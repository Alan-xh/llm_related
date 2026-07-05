"""
任务 5：实例分割（Instance Segmentation）
代表模型：Mask R-CNN（简化手写实现，不调用 torchvision.detection）
损失函数：分类 + 边框回归 + Mask 二分类损失
使用合成数据演示实例分割训练流程。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 超参数
BATCH_SIZE = 2
EPOCHS = 2
LR = 5e-3
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANCHOR_SCALES = [64, 128, 256]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
RPN_POS_THRESHOLD = 0.7
RPN_NEG_THRESHOLD = 0.3
ROI_POS_THRESHOLD = 0.5
ROI_NEG_THRESHOLD = 0.5
NMS_THRESHOLD = 0.7
NUM_POST_NMS = 128
ROI_POOL_SIZE = 7
MASK_SIZE = 14
STRIDE = 16


def box_iou(boxes1, boxes2):
    """计算两组框之间的 IoU。"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter
    return inter / union


def nms(boxes, scores, threshold):
    """非极大值抑制。"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        mask = ious <= threshold
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def generate_anchors(feature_h, feature_w, stride=STRIDE):
    """生成多尺度锚框。"""
    device = torch.device("cpu")
    shifts_x = torch.arange(0, feature_w, device=device) * stride
    shifts_y = torch.arange(0, feature_h, device=device) * stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    centers = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).float()

    anchors = []
    for scale in ANCHOR_SCALES:
        for ratio in ANCHOR_RATIOS:
            h = scale / torch.sqrt(torch.tensor(ratio))
            w = scale * torch.sqrt(torch.tensor(ratio))
            base = torch.tensor([-w / 2, -h / 2, w / 2, h / 2], device=device)
            anchors.append(centers + base)
    return torch.cat(anchors, dim=0)


def box_encode(reference, proposal):
    """GT 框相对于 proposal 的回归目标。"""
    px = (proposal[:, 0] + proposal[:, 2]) / 2
    py = (proposal[:, 1] + proposal[:, 3]) / 2
    pw = proposal[:, 2] - proposal[:, 0]
    ph = proposal[:, 3] - proposal[:, 1]

    gx = (reference[:, 0] + reference[:, 2]) / 2
    gy = (reference[:, 1] + reference[:, 3]) / 2
    gw = reference[:, 2] - reference[:, 0]
    gh = reference[:, 3] - reference[:, 1]

    return torch.stack([
        (gx - px) / pw,
        (gy - py) / ph,
        torch.log(gw / pw + 1e-6),
        torch.log(gh / ph + 1e-6),
    ], dim=1)


def box_decode(proposal, delta):
    """将回归目标解码为真实框坐标。"""
    px = (proposal[:, 0] + proposal[:, 2]) / 2
    py = (proposal[:, 1] + proposal[:, 3]) / 2
    pw = proposal[:, 2] - proposal[:, 0]
    ph = proposal[:, 3] - proposal[:, 1]

    gx = delta[:, 0] * pw + px
    gy = delta[:, 1] * ph + py
    gw = torch.exp(delta[:, 2]) * pw
    gh = torch.exp(delta[:, 3]) * ph

    return torch.stack([
        gx - gw / 2, gy - gh / 2,
        gx + gw / 2, gy + gh / 2,
    ], dim=1)


def roi_pool(feature, boxes, output_size=ROI_POOL_SIZE):
    """RoI 池化。"""
    if boxes.numel() == 0:
        return torch.zeros(
            (0, feature.size(0), output_size, output_size),
            device=feature.device,
        )
    boxes = boxes / STRIDE
    rois = []
    for box in boxes:
        x1, y1, x2, y2 = box.long()
        x1 = x1.clamp(0, feature.size(2) - 1)
        y1 = y1.clamp(0, feature.size(1) - 1)
        x2 = x2.clamp(x1 + 1, feature.size(2))
        y2 = y2.clamp(y1 + 1, feature.size(1))
        crop = feature[:, y1:y2, x1:x2]
        roi = F.adaptive_max_pool2d(crop, (output_size, output_size))
        rois.append(roi)
    return torch.stack(rois, dim=0)


def assign_labels(proposals, gt_boxes, gt_labels, pos_threshold, neg_threshold):
    """为候选框分配标签并采样。"""
    if gt_boxes.numel() == 0:
        labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
        return labels, None, torch.arange(len(proposals), device=proposals.device)

    ious = box_iou(proposals, gt_boxes)
    max_iou, matched_gt = ious.max(dim=1)
    _, best_anchor_per_gt = ious.max(dim=0)

    labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
    pos_mask = (max_iou >= pos_threshold)
    pos_mask[best_anchor_per_gt] = True
    labels[pos_mask] = gt_labels[matched_gt[pos_mask]]

    neg_mask = (~pos_mask) & (max_iou < neg_threshold)
    labels[neg_mask] = 0

    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    num_pos = min(64, len(pos_idx))
    if len(pos_idx) > num_pos:
        perm = torch.randperm(len(pos_idx), device=proposals.device)[:num_pos]
        pos_idx = pos_idx[perm]

    num_neg = min(256 - num_pos, len(neg_idx))
    if len(neg_idx) > num_neg:
        perm = torch.randperm(len(neg_idx), device=proposals.device)[:num_neg]
        neg_idx = neg_idx[perm]

    sampled = torch.cat([pos_idx, neg_idx])
    return labels, matched_gt, sampled


def mask_target(proposals, matched_gt, gt_masks, mask_size=MASK_SIZE):
    """为正样本 proposal 生成 mask 目标。"""
    targets = []
    for i, gt_idx in enumerate(matched_gt):
        box = proposals[i].long()
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gt_masks.size(2), x2), min(gt_masks.size(1), y2)
        if x2 <= x1 or y2 <= y1:
            targets.append(torch.zeros(mask_size, mask_size, device=proposals.device))
            continue
        crop = gt_masks[gt_idx, y1:y2, x1:x2].float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(crop, size=(mask_size, mask_size), mode="bilinear", align_corners=False)
        targets.append(resized.squeeze())
    return torch.stack(targets, dim=0)


class SimpleBackbone(nn.Module):
    """简单 CNN 骨干。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_block(64, 64, stride=1)
        self.layer2 = self._make_block(64, 128, stride=2)
        self.layer3 = self._make_block(128, 256, stride=2)

    def _make_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * 2, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(self, x):
        t = F.relu(self.conv(x))
        cls = self.cls_logits(t)
        bbox = self.bbox_pred(t)
        B, _, H, W = x.shape
        cls = cls.permute(0, 2, 3, 1).reshape(B, H * W * len(ANCHOR_SCALES) * len(ANCHOR_RATIOS), 2)
        bbox = bbox.permute(0, 2, 3, 1).reshape(B, H * W * len(ANCHOR_SCALES) * len(ANCHOR_RATIOS), 4)
        return cls, bbox


class MaskHead(nn.Module):
    """Mask R-CNN 的 mask 预测头。"""

    def __init__(self, in_channels, num_classes, mask_size=MASK_SIZE):
        super().__init__()
        self.mask_size = mask_size
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.predictor = nn.Conv2d(256, num_classes, 1)

    def forward(self, roi_features):
        x = F.relu(self.conv1(roi_features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.deconv(x))
        return self.predictor(x)  # (N, num_classes, mask_size*2, mask_size*2)


class MaskRCNN(nn.Module):
    """简化版 Mask R-CNN。"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self.rpn = RPNHead(256, self.num_anchors)

        self.fc1 = nn.Linear(256 * ROI_POOL_SIZE * ROI_POOL_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        self.mask_head = MaskHead(256, num_classes)

    def forward(self, images, targets=None):
        features = self.backbone(torch.stack(images, dim=0))
        B, _, H, W = features.shape
        device = features.device
        anchors = generate_anchors(H, W, STRIDE).to(device)

        rpn_cls, rpn_bbox = self.rpn(features)

        losses = {}
        for i in range(B):
            gt = targets[i]["boxes"].to(device)
            labels_gt = targets[i]["labels"].to(device)
            gt_masks = targets[i]["masks"].to(device)

            # RPN
            rpn_labels, matched, sampled = assign_labels(
                anchors, gt, torch.ones(len(gt), device=device),
                RPN_POS_THRESHOLD, RPN_NEG_THRESHOLD,
            )
            rpn_cls_i = rpn_cls[i]
            rpn_bbox_i = rpn_bbox[i]
            rpn_cls_loss = F.cross_entropy(rpn_cls_i[sampled], rpn_labels[sampled])

            sampled_pos = sampled[rpn_labels[sampled] == 1]
            if sampled_pos.numel() > 0:
                pos_anchors = anchors[sampled_pos]
                matched_gt = gt[matched[sampled_pos]]
                bbox_targets = box_encode(matched_gt, pos_anchors)
                bbox_preds = rpn_bbox_i[sampled_pos]
                rpn_box_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
            else:
                rpn_box_loss = torch.tensor(0.0, device=device)

            # Proposals
            scores = F.softmax(rpn_cls_i, dim=1)[:, 1]
            decoded = box_decode(anchors, rpn_bbox_i).detach()
            decoded[:, [0, 2]] = decoded[:, [0, 2]].clamp(0, images[i].shape[2])
            decoded[:, [1, 3]] = decoded[:, [1, 3]].clamp(0, images[i].shape[1])
            keep = nms(decoded, scores, NMS_THRESHOLD)
            proposals = decoded[keep[:NUM_POST_NMS]]

            # Detection head assignment
            det_labels, det_matched, det_sampled = assign_labels(
                proposals, gt, labels_gt,
                ROI_POS_THRESHOLD, ROI_NEG_THRESHOLD,
            )
            sampled_proposals = proposals[det_sampled]

            roi_features = roi_pool(features[i], sampled_proposals)
            flat = roi_features.view(roi_features.size(0), -1)
            h = F.relu(self.fc1(flat))
            h = F.relu(self.fc2(h))
            cls_logits = self.cls_score(h)
            bbox_deltas = self.bbox_pred(h)

            det_cls_loss = F.cross_entropy(cls_logits, det_labels[det_sampled])

            pos_det = det_labels[det_sampled] > 0
            det_box_loss = torch.tensor(0.0, device=device)
            mask_loss = torch.tensor(0.0, device=device)
            if pos_det.sum() > 0:
                pos_labels = det_labels[det_sampled][pos_det]
                pos_proposals = sampled_proposals[pos_det]
                pos_gt = gt[det_matched[det_sampled][pos_det]]
                pos_gt_masks = gt_masks[det_matched[det_sampled][pos_det]]

                bbox_targets = box_encode(pos_gt, pos_proposals)
                bbox_preds = bbox_deltas[pos_det].view(-1, NUM_CLASSES, 4)
                bbox_preds = bbox_preds[torch.arange(len(pos_labels)), pos_labels]
                det_box_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)

                # Mask head
                mask_roi_features = roi_pool(features[i], pos_proposals, output_size=ROI_POOL_SIZE)
                mask_logits = self.mask_head(mask_roi_features)
                mask_targets = mask_target(pos_proposals, det_matched[det_sampled][pos_det], gt_masks)
                mask_preds = mask_logits[torch.arange(len(pos_labels)), pos_labels]
                mask_loss = F.binary_cross_entropy_with_logits(mask_preds, mask_targets)

            for k, v in [
                ("loss_rpn_cls", rpn_cls_loss),
                ("loss_rpn_box_reg", rpn_box_loss),
                ("loss_classifier", det_cls_loss),
                ("loss_box_reg", det_box_loss),
                ("loss_mask", mask_loss),
            ]:
                losses[k] = losses.get(k, 0.0) + v

        for k in losses:
            losses[k] = losses[k] / B
        return losses


class SyntheticInstanceDataset(Dataset):
    """合成实例分割数据集。"""

    def __init__(self, num_samples=100, num_classes=NUM_CLASSES, size=128):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, self.size, self.size)
        num_objs = torch.randint(1, 4, (1,)).item()

        boxes = []
        labels = []
        masks = []
        for _ in range(num_objs):
            x1 = int(torch.rand(1).item() * (self.size - 40))
            y1 = int(torch.rand(1).item() * (self.size - 40))
            w = int(torch.rand(1).item() * 30 + 10)
            h = int(torch.rand(1).item() * 30 + 10)
            x2 = x1 + w
            y2 = y1 + h

            boxes.append([x1, y1, x2, y2])
            labels.append(torch.randint(1, self.num_classes, (1,)).item())

            mask = torch.zeros(self.size, self.size, dtype=torch.bool)
            mask[y1:y2, x1:x2] = True
            masks.append(mask)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.stack(masks),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(
                [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32
            ),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    dataset = SyntheticInstanceDataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = MaskRCNN(num_classes=NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)

    model.train()
    for epoch in range(EPOCHS):
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}]  "
                f"total_loss: {losses.item():.4f}  "
                f"cls: {loss_dict['loss_classifier'].item():.4f}  "
                f"box: {loss_dict['loss_box_reg'].item():.4f}  "
                f"mask: {loss_dict['loss_mask'].item():.4f}"
            )


if __name__ == "__main__":
    main()
