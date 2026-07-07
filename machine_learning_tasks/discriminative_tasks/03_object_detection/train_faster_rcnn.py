"""
任务 3：目标检测（Object Detection）
代表模型：Faster R-CNN（基于区域的更快卷积神经网络, 简化手写实现，不调用 torchvision.detection）
损失函数：RPN 分类/回归损失 + 检测头分类/回归损失
使用合成数据演示目标检测训练流程。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 超参数
BATCH_SIZE = 4
EPOCHS = 2
LR = 5e-3
NUM_CLASSES = 3  # 包含背景
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANCHOR_SCALES = [64, 128, 256]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]
RPN_POS_THRESHOLD = 0.7
RPN_NEG_THRESHOLD = 0.3
ROI_POS_THRESHOLD = 0.5
ROI_NEG_THRESHOLD = 0.5
NMS_THRESHOLD = 0.7
NUM_PRE_NMS = 2000
NUM_POST_NMS = 128
ROI_POOL_SIZE = 7
STRIDE = 16


def box_iou(boxes1, boxes2):
    """
    计算两组边界框之间的交并比（IoU, Intersection over Union），boxes 格式为 [x1, y1, x2, y2]

    boxes1: [N, 4], 格式: [x1, y1, x2, y2]
    boxes2: [M, 4], 格式: [x1, y1, x2, y2]

    输出: [N, M], 交并比
    """
    # 计算面积 [N], [M]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 两两对比计算交集区域左上角和右下角坐标 [N, M]
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0]) 
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    # 计算交集面积 [N, M]
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    # 计算并级面积
    union = area1[:, None] + area2[None, :] - inter
    return inter / union


def nms(boxes, scores, threshold):
    """
    非极大值抑制，去除冗余(交并比过高)检测框,并按分数从高到低的顺序输出
    
    boxes: [N, 4], 边界框坐标 [x1, y1, x2, y2]
    scores: [N,], 框得分
    threshold: 交并比阈值，去除重叠度过高的框
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True) # 降序索引
    keep = [] # 保留的最大框
    while order.numel() > 0:
        i = order[0] # 取分数最大的索引
        keep.append(i.item()) # 取概率最大框的索引
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0] # 取概率最大的 boxe 和其他框计算交并比
        mask = ious <= threshold # 筛选出 IoU 小于阈值的框的序号
        order = order[1:][mask] # 筛选出 IoU 小于阈值的框的索引
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def generate_anchors(feature_h, feature_w, stride=STRIDE):
    """
    为每个特征图位置生成多尺度锚框
    
    """
    device = torch.device("cpu")  # 后续会移到目标设备
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
            # [x1, y1, x2, y2] 相对于中心点的偏移
            base = torch.tensor([-w / 2, -h / 2, w / 2, h / 2], device=device)
            anchors.append(centers + base)
    anchors = torch.cat(anchors, dim=0)
    return anchors


def box_encode(reference, proposal):
    """将 GT 框编码为相对于 proposal 的回归目标。"""
    px = (proposal[:, 0] + proposal[:, 2]) / 2
    py = (proposal[:, 1] + proposal[:, 3]) / 2
    pw = proposal[:, 2] - proposal[:, 0]
    ph = proposal[:, 3] - proposal[:, 1]

    gx = (reference[:, 0] + reference[:, 2]) / 2
    gy = (reference[:, 1] + reference[:, 3]) / 2
    gw = reference[:, 2] - reference[:, 0]
    gh = reference[:, 3] - reference[:, 1]

    targets = torch.stack([
        (gx - px) / pw,
        (gy - py) / ph,
        torch.log(gw / pw + 1e-6),
        torch.log(gh / ph + 1e-6),
    ], dim=1)
    return targets


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
    """对单张图像特征图上的 RoI 进行固定尺寸池化。"""
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
    """为候选框分配标签并采样；同时把每个 GT 最大 IoU 的候选框强制设为正例。"""
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
    labels[neg_mask] = 0  # 背景

    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    # 采样：正样本最多 64，负样本补满 256
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


class SimpleBackbone(nn.Module):
    """简单 CNN 骨干，输出 1/16 输入尺寸的特征图。"""

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
    """区域提议网络头。"""

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


class FasterRCNN(nn.Module):
    """简化版 Faster R-CNN。"""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = SimpleBackbone()
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self.rpn = RPNHead(256, self.num_anchors)

        self.fc1 = nn.Linear(256 * ROI_POOL_SIZE * ROI_POOL_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(self, images, targets=None):
        features = self.backbone(torch.stack(images, dim=0))
        B, _, H, W = features.shape

        device = features.device
        anchors = generate_anchors(H, W, STRIDE).to(device)

        rpn_cls, rpn_bbox = self.rpn(features)

        losses = {}
        proposals_all = []
        if self.training:
            for i in range(B):
                gt = targets[i]["boxes"].to(device)
                labels_gt = targets[i]["labels"].to(device)

                # RPN 标签分配
                rpn_labels, matched, sampled = assign_labels(
                    anchors, gt, torch.ones(len(gt), device=device),
                    RPN_POS_THRESHOLD, RPN_NEG_THRESHOLD,
                )
                rpn_cls_i = rpn_cls[i]
                rpn_bbox_i = rpn_bbox[i]

                # 只保留采样后的正样本计算回归目标
                sampled_pos = sampled[rpn_labels[sampled] == 1]
                if sampled_pos.numel() > 0:
                    pos_anchors = anchors[sampled_pos]
                    matched_gt = gt[matched[sampled_pos]]
                    bbox_targets = box_encode(matched_gt, pos_anchors)
                    bbox_preds = rpn_bbox_i[sampled_pos]
                    rpn_box_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
                else:
                    rpn_box_loss = torch.tensor(0.0, device=device)

                rpn_cls_loss = F.cross_entropy(rpn_cls_i[sampled], rpn_labels[sampled])

                # 生成训练用 proposals（仅正样本 + 少量随机负样本）
                scores = F.softmax(rpn_cls_i, dim=1)[:, 1]
                decoded = box_decode(anchors, rpn_bbox_i).detach()
                decoded[:, [0, 2]] = decoded[:, [0, 2]].clamp(0, images[i].shape[2])
                decoded[:, [1, 3]] = decoded[:, [1, 3]].clamp(0, images[i].shape[1])

                keep = nms(decoded, scores, NMS_THRESHOLD)
                proposals = decoded[keep[:NUM_POST_NMS]]

                # 为检测头分配标签
                det_labels, det_matched, det_sampled = assign_labels(
                    proposals, gt, labels_gt,
                    ROI_POS_THRESHOLD, ROI_NEG_THRESHOLD,
                )
                proposals_all.append(proposals[det_sampled])

                roi_features = roi_pool(features[i], proposals_all[-1])
                flat = roi_features.view(roi_features.size(0), -1)
                h = F.relu(self.fc1(flat))
                h = F.relu(self.fc2(h))
                cls_logits = self.cls_score(h)
                bbox_deltas = self.bbox_pred(h)

                det_cls_loss = F.cross_entropy(cls_logits, det_labels[det_sampled])

                pos_det = det_labels[det_sampled] > 0
                if pos_det.sum() > 0:
                    pos_labels = det_labels[det_sampled][pos_det]
                    pos_proposals = proposals_all[-1][pos_det]
                    pos_gt = gt[det_matched[det_sampled][pos_det]]
                    bbox_targets = box_encode(pos_gt, pos_proposals)
                    # 选择对应类别的回归分支
                    bbox_preds = bbox_deltas[pos_det].view(-1, NUM_CLASSES, 4)
                    bbox_preds = bbox_preds[torch.arange(len(pos_labels)), pos_labels]
                    det_box_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
                else:
                    det_box_loss = torch.tensor(0.0, device=device)

                for k, v in [
                    ("loss_rpn_cls", rpn_cls_loss),
                    ("loss_rpn_box_reg", rpn_box_loss),
                    ("loss_classifier", det_cls_loss),
                    ("loss_box_reg", det_box_loss),
                ]:
                    losses[k] = losses.get(k, 0.0) + v

            for k in losses:
                losses[k] = losses[k] / B
            return losses
        else:
            # 推理模式：仅返回 proposals（本示例未使用）
            for i in range(B):
                scores = F.softmax(rpn_cls[i], dim=1)[:, 1]
                decoded = box_decode(anchors, rpn_bbox[i])
                keep = nms(decoded, scores, NMS_THRESHOLD)
                proposals_all.append(decoded[keep[:NUM_POST_NMS]])
            return proposals_all


class SyntheticDetectionDataset(Dataset):
    """合成目标检测数据集，返回 (image, target)。"""

    def __init__(self, num_samples=200, num_classes=NUM_CLASSES, size=256):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _idx):
        image = torch.rand(3, self.size, self.size)
        num_objs = torch.randint(1, 5, (1,)).item()

        boxes = []
        labels = []
        for _ in range(num_objs):
            x1 = torch.rand(1).item() * (self.size - 60)
            y1 = torch.rand(1).item() * (self.size - 60)
            w = torch.rand(1).item() * 40 + 20
            h = torch.rand(1).item() * 40 + 20
            x2 = x1 + w
            y2 = y1 + h
            boxes.append([x1, y1, x2, y2])
            labels.append(torch.randint(1, self.num_classes, (1,)).item())

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    dataset = SyntheticDetectionDataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = FasterRCNN(num_classes=NUM_CLASSES).to(DEVICE)
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
                f"rpn_cls: {loss_dict['loss_rpn_cls'].item():.4f}  "
                f"rpn_box: {loss_dict['loss_rpn_box_reg'].item():.4f}  "
                f"cls_loss: {loss_dict['loss_classifier'].item():.4f}  "
                f"box_loss: {loss_dict['loss_box_reg'].item():.4f}"
            )


if __name__ == "__main__":
    main()
