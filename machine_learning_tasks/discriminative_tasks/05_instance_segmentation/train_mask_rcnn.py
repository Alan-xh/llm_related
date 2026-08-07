"""
Task 5: Instance Segmentation (Mask R-CNN Pure PyTorch Implementation)
Architecture: Mask R-CNN (He et al., ICCV 2017)
Domain: Computer Vision / Instance Segmentation

Core Concept & Algorithm Flow:
    Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation
    masks on each Region of Interest (RoI), in parallel with the existing branch
    for classification and bounding box regression. The key innovation is RoIAlign
    (or exact feature extraction via Adaptive Max/Average Pooling) replacing RoIPool
    to preserve spatial pixel accuracy required for mask generation.

Pipeline Overview:
    1. Input Images [B, C, H, W] -> Feature Extraction via Backbone CNN -> Feature Map [B, C_feat, H_feat, W_feat]
    2. Feature Map -> Region Proposal Network (RPN) -> Anchor Scores [B, N_anchors, 2] & Delta Regressions [B, N_anchors, 4]
    3. Generate Anchors & Decode Proposals -> Apply Non-Maximum Suppression (NMS) -> Select Top-K Proposals
    4. Proposals & GT Assignment -> Assign Class Labels & Encode Targets
    5. RoI Feature Extraction (RoI Align/Pool) -> Sampled RoI Features [N_samples, C_feat, K_pool, K_pool]
    6. Two Parallel Heads:
        a) Fast R-CNN Box & Classification Head -> Linear FCs -> Cls Logits & Box Deltas
        b) Mask Head -> Fully Convolutional Transpose Network -> Class-specific Mask Predictions [N_pos, Num_Classes, Mask_H, Mask_W]

Mathematical Loss Formulations & Code Variable Mapping:
    Total Multi-task Loss Function:
        L_total = L_rpn_cls + L_rpn_box + L_cls + L_box + L_mask

    1. RPN Classification Loss (Binary Cross-Entropy):
        L_rpn_cls = - (1 / N_cls) * sum_i [ p_i^* log(p_i) + (1 - p_i^*) log(1 - p_i) ]
        - Code: F.cross_entropy(rpn_cls_logits[sampled], rpn_labels[sampled])
    
    2. Box Regression Loss (Smooth L1 Loss):
        SmoothL1(x) = 0.5 * x^2 if |x| < 1 else |x| - 0.5
        L_box = (1 / N_reg) * sum_i p_i^* * SmoothL1(t_i - t_i^*)
        - Code: F.smooth_l1_loss(bbox_preds, bbox_targets)
        - Target Parametrization (BBox Encoding):
            t_x = (g_x - p_x) / p_w,  t_y = (g_y - p_y) / p_h
            t_w = log(g_w / p_w),     t_h = log(g_h / p_h)

    3. Fast R-CNN Classification Loss (Categorical Cross-Entropy):
        L_cls = - log(p_{k^*})  where k^* is the ground truth class label.
        - Code: F.cross_entropy(cls_logits, det_labels[det_sampled])

    4. Mask Binary Cross-Entropy Loss:
        L_mask = - (1 / m^2) * sum_{1 <= i, j <= m} [ y_{ij} log(y_{ij}^*) + (1 - y_{ij}) log(1 - y_{ij}^*) ]
        - Code: F.binary_cross_entropy_with_logits(mask_preds, mask_targets)

Input / Output Specification:
    Input:
        - images: List[Tensor], length B, each tensor shape [3, H, W]
        - targets: List[Dict], length B, containing:
            * "boxes": Tensor [N_gt, 4] in (x1, y1, x2, y2) format
            * "labels": Tensor [N_gt] in range [1, NUM_CLASSES - 1] (0 is background)
            * "masks": Tensor [N_gt, H, W] binary masks
    Output:
        - Training: Dictionary of scalar loss tensors {'loss_rpn_cls', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg', 'loss_mask'}
        - Inference: Dict containing predicted boxes, labels, scores, and binary masks.
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# SECTION 3: Hyperparameters & Global Configuration
# ==============================================================================

BATCH_SIZE: int = 2
EPOCHS: int = 2
LR: float = 5e-3
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 5e-4
NUM_CLASSES: int = 3  # 0: Background, 1: Class_A, 2: Class_B
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anchor & RPN Hyperparameters
ANCHOR_SCALES: List[int] = [32, 64, 128]
ANCHOR_RATIOS: List[float] = [0.5, 1.0, 2.0]
RPN_POS_THRESHOLD: float = 0.7
RPN_NEG_THRESHOLD: float = 0.3
RPN_BATCH_SIZE: int = 256

# RoI & Proposal Hyperparameters
ROI_POS_THRESHOLD: float = 0.5
ROI_NEG_THRESHOLD: float = 0.5
ROI_BATCH_SIZE: int = 64
ROI_POS_RATIO: float = 0.25
NMS_THRESHOLD: float = 0.7
NUM_POST_NMS: int = 128
ROI_POOL_SIZE: int = 7
MASK_SIZE: int = 14
STRIDE: int = 16


# ==============================================================================
# SECTION 4: Data Processing, Geometry Utilities & Pipeline
# ==============================================================================

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): Bounding boxes set 1, shape: [N, 4] (x1, y1, x2, y2)
        boxes2 (Tensor): Bounding boxes set 2, shape: [M, 4] (x1, y1, x2, y2)

    Returns:
        iou (Tensor): Pairwise IoU matrix, shape: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)  # [M]

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # [N, M]
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])  # [N, M]
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])  # [N, M]
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])  # [N, M]

    inter_w = (inter_x2 - inter_x1).clamp(min=0)  # [N, M]
    inter_h = (inter_y2 - inter_y1).clamp(min=0)  # [N, M]
    inter = inter_w * inter_h                     # [N, M]

    union = area1[:, None] + area2[None, :] - inter  # [N, M]
    iou = inter / (union + 1e-6)                      # [N, M]
    return iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Performs Non-Maximum Suppression (NMS) on bounding boxes based on scores.

    Args:
        boxes (Tensor): Box coordinates, shape: [N, 4]
        scores (Tensor): Confidence scores, shape: [N]
        threshold (float): IoU overlap threshold for suppression.

    Returns:
        keep (Tensor): Indices of boxes retained after NMS, shape: [K]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)  # [N]
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]  # [N - 1]
        mask = ious <= threshold                             # [N - 1]
        order = order[1:][mask]                              # [N_remaining]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def generate_anchors(feature_h: int, feature_w: int, stride: int = STRIDE) -> torch.Tensor:
    """
    Generates multi-scale, multi-ratio anchor boxes across feature map coordinates.

    Mathematical Mapping:
        Center_x = (col + 0.5) * stride, Center_y = (row + 0.5) * stride
        w = scale * sqrt(ratio),         h = scale / sqrt(ratio)

    Args:
        feature_h (int): Height of feature map.
        feature_w (int): Width of feature map.
        stride (int): Stride of the backbone network.

    Outputs:
        anchors (Tensor): Anchor coordinates, shape: [H * W * A, 4] where A = len(scales) * len(ratios)
    """
    device = torch.device("cpu")
    shifts_x = torch.arange(0, feature_w, device=device) * stride + stride / 2.0  # [W]
    shifts_y = torch.arange(0, feature_h, device=device) * stride + stride / 2.0  # [H]
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")         # [H, W], [H, W]
    shift_x = shift_x.reshape(-1)                                                 # [H * W]
    shift_y = shift_y.reshape(-1)                                                 # [H * W]
    centers = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).float()   # [H * W, 4]

    base_anchors = []
    for scale in ANCHOR_SCALES:
        for ratio in ANCHOR_RATIOS:
            h = scale / math.sqrt(ratio)
            w = scale * math.sqrt(ratio)
            base_anchors.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])

    base_anchors = torch.tensor(base_anchors, device=device).float()  # [A, 4]
    A = base_anchors.size(0)
    K = centers.size(0)

    # Broadcast addition: [K, 1, 4] + [1, A, 4] -> [K, A, 4]
    anchors = centers.view(K, 1, 4) + base_anchors.view(1, A, 4)      # [K, A, 4]
    return anchors.reshape(-1, 4)                                     # [H * W * A, 4]


def box_encode(reference: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    """
    Encodes ground-truth box offsets relative to proposal/anchor bounding boxes.

    Mathematical Mapping:
        dx = (g_x - p_x) / p_w,   dy = (g_y - p_y) / p_h
        dw = log(g_w / p_w),      dh = log(g_h / p_h)

    Args:
        reference (Tensor): Ground truth boxes, shape: [N, 4]
        proposal (Tensor): Anchor or proposal boxes, shape: [N, 4]

    Outputs:
        targets (Tensor): Encoded regression deltas, shape: [N, 4]
    """
    px = (proposal[:, 0] + proposal[:, 2]) * 0.5  # [N]
    py = (proposal[:, 1] + proposal[:, 3]) * 0.5  # [N]
    pw = proposal[:, 2] - proposal[:, 0]          # [N]
    ph = proposal[:, 3] - proposal[:, 1]          # [N]

    gx = (reference[:, 0] + reference[:, 2]) * 0.5  # [N]
    gy = (reference[:, 1] + reference[:, 3]) * 0.5  # [N]
    gw = reference[:, 2] - reference[:, 0]          # [N]
    gh = reference[:, 3] - reference[:, 1]          # [N]

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / (pw + 1e-6) + 1e-6)
    dh = torch.log(gh / (ph + 1e-6) + 1e-6)

    return torch.stack([dx, dy, dw, dh], dim=1)     # [N, 4]


def box_decode(proposal: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Decodes predicted bounding box regression deltas back to absolute spatial coordinates.

    Mathematical Mapping:
        g_x = p_x + dx * p_w,     g_y = p_y + dy * p_h
        g_w = p_w * exp(dw),      g_h = p_h * exp(dh)

    Args:
        proposal (Tensor): Anchor or proposal boxes, shape: [N, 4]
        delta (Tensor): Predicted regression deltas, shape: [N, 4]

    Outputs:
        decoded_boxes (Tensor): Coordinates (x1, y1, x2, y2), shape: [N, 4]
    """
    px = (proposal[:, 0] + proposal[:, 2]) * 0.5  # [N]
    py = (proposal[:, 1] + proposal[:, 3]) * 0.5  # [N]
    pw = proposal[:, 2] - proposal[:, 0]          # [N]
    ph = proposal[:, 3] - proposal[:, 1]          # [N]

    gx = delta[:, 0] * pw + px
    gy = delta[:, 1] * ph + py
    gw = torch.exp(delta[:, 2].clamp(max=5.0)) * pw
    gh = torch.exp(delta[:, 3].clamp(max=5.0)) * ph

    return torch.stack([
        gx - gw * 0.5, gy - gh * 0.5,
        gx + gw * 0.5, gy + gh * 0.5
    ], dim=1)                                    # [N, 4]


def roi_pool(feature: torch.Tensor, boxes: torch.Tensor, output_size: int = ROI_POOL_SIZE) -> torch.Tensor:
    """
    Extracts fixed-size spatial feature maps for bounding box proposals via Adaptive Max Pooling.

    Args:
        feature (Tensor): Feature map from backbone, shape: [C, H_feat, W_feat]
        boxes (Tensor): Spatial RoI bounding boxes in image scale, shape: [N, 4]
        output_size (int): Spatial dimensions of extracted RoI features.

    Outputs:
        rois (Tensor): Pooled RoI feature maps, shape: [N, C, output_size, output_size]
    """
    if boxes.numel() == 0:
        return torch.zeros((0, feature.size(0), output_size, output_size), device=feature.device)

    boxes_scaled = boxes / STRIDE  # Convert to feature map scale [N, 4]
    rois = []
    for box in boxes_scaled:
        x1, y1, x2, y2 = box.long()
        x1 = x1.clamp(0, feature.size(2) - 1)
        y1 = y1.clamp(0, feature.size(1) - 1)
        x2 = x2.clamp(x1 + 1, feature.size(2))
        y2 = y2.clamp(y1 + 1, feature.size(1))

        crop = feature[:, y1:y2, x1:x2]  # [C, H_crop, W_crop]
        roi = F.adaptive_max_pool2d(crop, (output_size, output_size))  # [C, output_size, output_size]
        rois.append(roi)

    return torch.stack(rois, dim=0)     # [N, C, output_size, output_size]


def assign_labels(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pos_threshold: float,
    neg_threshold: float,
    max_samples: int = 256,
    pos_fraction: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assigns ground-truth class labels to proposal candidates and performs balanced sampling.

    Args:
        proposals (Tensor): Candidate proposals, shape: [N, 4]
        gt_boxes (Tensor): Ground truth boxes, shape: [M, 4]
        gt_labels (Tensor): Ground truth class labels, shape: [M]
        pos_threshold (float): Minimum IoU for positive label.
        neg_threshold (float): Maximum IoU for negative label.
        max_samples (int): Total number of proposals to sample.
        pos_fraction (float): Target ratio of positive proposals.

    Outputs:
        labels (Tensor): Class label per proposal, shape: [N]
        matched_gt (Tensor): GT index matched to each proposal, shape: [N]
        sampled_indices (Tensor): Filtered indices of sampled proposals, shape: [K]
    """
    if gt_boxes.numel() == 0:
        labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
        matched_gt = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
        sampled = torch.arange(min(len(proposals), max_samples), device=proposals.device)
        return labels, matched_gt, sampled

    ious = box_iou(proposals, gt_boxes)                  # [N, M]
    max_iou, matched_gt = ious.max(dim=1)                # [N], [N]
    _, best_anchor_per_gt = ious.max(dim=0)             # [M]

    labels = torch.full((len(proposals),), -1, dtype=torch.long, device=proposals.device)  # [N]
    
    # Assign positive and negative masks
    pos_mask = (max_iou >= pos_threshold)                # [N]
    pos_mask[best_anchor_per_gt] = True
    labels[pos_mask] = gt_labels[matched_gt[pos_mask]]

    neg_mask = (~pos_mask) & (max_iou < neg_threshold)    # [N]
    labels[neg_mask] = 0  # Background

    pos_idx = torch.where(labels > 0)[0]                 # [N_pos]
    neg_idx = torch.where(labels == 0)[0]                # [N_neg]

    num_pos = min(int(max_samples * pos_fraction), len(pos_idx))
    if len(pos_idx) > num_pos:
        perm = torch.randperm(len(pos_idx), device=proposals.device)[:num_pos]
        pos_idx = pos_idx[perm]

    num_neg = min(max_samples - len(pos_idx), len(neg_idx))
    if len(neg_idx) > num_neg:
        perm = torch.randperm(len(neg_idx), device=proposals.device)[:num_neg]
        neg_idx = neg_idx[perm]

    sampled_indices = torch.cat([pos_idx, neg_idx])      # [K]
    return labels, matched_gt, sampled_indices


def mask_target(
    proposals: torch.Tensor,
    matched_gt: torch.Tensor,
    gt_masks: torch.Tensor,
    mask_size: int = MASK_SIZE
) -> torch.Tensor:
    """
    Extracts and resizes ground-truth binary mask regions corresponding to positive proposals.

    Args:
        proposals (Tensor): Positive proposal bounding boxes, shape: [N_pos, 4]
        matched_gt (Tensor): Matched ground truth indices, shape: [N_pos]
        gt_masks (Tensor): Ground truth binary masks, shape: [M_gt, H, W]
        mask_size (int): Target spatial output resolution (e.g., 14 or 28).

    Outputs:
        targets (Tensor): Cropped & resized binary mask targets, shape: [N_pos, mask_size, mask_size]
    """
    targets = []
    for i, gt_idx in enumerate(matched_gt):
        box = proposals[i].long()
        x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gt_masks.size(2), x2), min(gt_masks.size(1), y2)

        if x2 <= x1 or y2 <= y1:
            targets.append(torch.zeros((mask_size, mask_size), device=proposals.device))
            continue

        crop = gt_masks[gt_idx, y1:y2, x1:x2].float().unsqueeze(0).unsqueeze(0)  # [1, 1, h_crop, w_crop]
        resized = F.interpolate(crop, size=(mask_size, mask_size), mode="bilinear", align_corners=False)  # [1, 1, mask_size, mask_size]
        targets.append(resized.squeeze(0).squeeze(0))

    if len(targets) == 0:
        return torch.zeros((0, mask_size, mask_size), device=proposals.device)

    return torch.stack(targets, dim=0)  # [N_pos, mask_size, mask_size]


class SyntheticInstanceDataset(Dataset):
    """
    Synthetic Dataset for generating random images with bounding boxes, labels, and binary masks.

    Args:
        num_samples (int): Total number of dataset samples.
        num_classes (int): Total number of target classes including background.
        size (int): Spatial dimensions of generated images (size x size).
    """
    def __init__(self, num_samples: int = 100, num_classes: int = NUM_CLASSES, size: int = 128):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.size = size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            "area": torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32),
        }
        return image, target


def collate_fn(batch: List) -> Tuple:
    return tuple(zip(*batch))


# ==============================================================================
# SECTION 5: Core Sub-modules (Backbone, RPN Head, Mask Head)
# ==============================================================================

class SimpleBackbone(nn.Module):
    """
    Simple ResNet-style Convolutional Backbone Network for Feature Extraction.

    Inputs:
        x (Tensor): Input images, shape: [B, 3, H, W]

    Outputs:
        feat (Tensor): Output feature map, shape: [B, 256, H/16, W/16]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # [B, 64, H/2, W/2]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                 # [B, 64, H/4, W/4]

        self.layer1 = self._make_block(64, 64, stride=1)                                # [B, 64, H/4, W/4]
        self.layer2 = self._make_block(64, 128, stride=2)                               # [B, 128, H/8, W/8]
        self.layer3 = self._make_block(128, 256, stride=2)                              # [B, 256, H/16, W/16]

    def _make_block(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B, 256, H/16, W/16]


class RPNHead(nn.Module):
    """
    Region Proposal Network (RPN) Head for anchor classification and box regression.

    Args:
        in_channels (int): Channels of input feature map.
        num_anchors (int): Number of anchor boxes per spatial location.

    Inputs:
        x (Tensor): Feature map, shape: [B, C_in, H_feat, W_feat]

    Outputs:
        cls_logits (Tensor): Objectness logits, shape: [B, H_feat * W_feat * A, 2]
        bbox_pred (Tensor): Anchor regression offsets, shape: [B, H_feat * W_feat * A, 4]
    """
    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        t = self.act(self.conv(x))                           # [B, C_in, H, W]
        cls = self.cls_logits(t)                             # [B, A*2, H, W]
        bbox = self.bbox_pred(t)                             # [B, A*4, H, W]

        # Reshape to [B, H*W*A, 2] and [B, H*W*A, 4]
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, 2)     # [B, H*W*A, 2]
        bbox = bbox.permute(0, 2, 3, 1).reshape(B, -1, 4)   # [B, H*W*A, 4]
        return cls, bbox


class MaskHead(nn.Module):
    """
    Fully Convolutional Mask Prediction Head for generating pixel-level instance masks.

    Args:
        in_channels (int): Input feature channel count.
        num_classes (int): Total target classes.
        mask_size (int): Target spatial base dimension.

    Inputs:
        roi_features (Tensor): Sampled RoI feature maps, shape: [N_pos, C_in, K, K]

    Outputs:
        mask_logits (Tensor): Class-specific mask logits, shape: [N_pos, Num_Classes, K*2, K*2]
    """
    def __init__(self, in_channels: int, num_classes: int, mask_size: int = MASK_SIZE):
        super().__init__()
        self.mask_size = mask_size
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.predictor = nn.Conv2d(256, num_classes, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(roi_features))               # [N_pos, 256, K, K]
        x = self.act(self.conv2(x))                          # [N_pos, 256, K, K]
        x = self.act(self.conv3(x))                          # [N_pos, 256, K, K]
        x = self.act(self.deconv(x))                         # [N_pos, 256, K*2, K*2]
        return self.predictor(x)                             # [N_pos, Num_Classes, K*2, K*2]


# ==============================================================================
# SECTION 6: Top-level Architecture (Mask R-CNN Model)
# ==============================================================================

class MaskRCNN(nn.Module):
    """
    Top-Level Mask R-CNN Architecture encompassing Backbone, RPN, Fast R-CNN Detection Head, and Mask Head.

    Args:
        num_classes (int): Number of target object categories (including background = 0).
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = SimpleBackbone()
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self.rpn = RPNHead(256, self.num_anchors)

        # Fast R-CNN Head
        self.fc1 = nn.Linear(256 * ROI_POOL_SIZE * ROI_POOL_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # Mask Head
        self.mask_head = MaskHead(256, num_classes)
        self.act = nn.GELU()

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward Pass for Training & Loss Computation.

        Inputs:
            images (List[Tensor]): List of input image tensors [3, H, W].
            targets (List[Dict], optional): List of target metadata dicts.

        Outputs:
            losses (Dict[str, Tensor]): Dictionary of calculated individual and total losses.
        """
        img_batch = torch.stack(images, dim=0)                 # [B, 3, H, W]
        features = self.backbone(img_batch)                    # [B, 256, H_feat, W_feat]
        B, _, H, W = features.shape
        device = features.device

        anchors = generate_anchors(H, W, STRIDE).to(device)    # [N_anchors, 4]
        rpn_cls, rpn_bbox = self.rpn(features)                 # [B, N_anchors, 2], [B, N_anchors, 4]

        losses = {
            "loss_rpn_cls": torch.tensor(0.0, device=device),
            "loss_rpn_box_reg": torch.tensor(0.0, device=device),
            "loss_classifier": torch.tensor(0.0, device=device),
            "loss_box_reg": torch.tensor(0.0, device=device),
            "loss_mask": torch.tensor(0.0, device=device),
        }

        for i in range(B):
            gt = targets[i]["boxes"].to(device)                # [N_gt, 4]
            labels_gt = targets[i]["labels"].to(device)        # [N_gt]
            gt_masks = targets[i]["masks"].to(device)          # [N_gt, H, W]

            # ------------------------------------------------------------------
            # Step 1: RPN Target Assignment & Losses
            # ------------------------------------------------------------------
            rpn_labels, matched, sampled = assign_labels(
                anchors, gt, torch.ones(len(gt), dtype=torch.long, device=device),
                RPN_POS_THRESHOLD, RPN_NEG_THRESHOLD, max_samples=RPN_BATCH_SIZE
            )
            rpn_cls_i = rpn_cls[i]                             # [N_anchors, 2]
            rpn_bbox_i = rpn_bbox[i]                           # [N_anchors, 4]

            valid_mask = rpn_labels[sampled] >= 0
            if valid_mask.sum() > 0:
                rpn_cls_loss = F.cross_entropy(rpn_cls_i[sampled[valid_mask]], rpn_labels[sampled[valid_mask]])
            else:
                rpn_cls_loss = torch.tensor(0.0, device=device)

            sampled_pos = sampled[rpn_labels[sampled] == 1]
            if sampled_pos.numel() > 0:
                pos_anchors = anchors[sampled_pos]             # [N_pos_rpn, 4]
                matched_gt_rpn = gt[matched[sampled_pos]]      # [N_pos_rpn, 4]
                bbox_targets_rpn = box_encode(matched_gt_rpn, pos_anchors)  # [N_pos_rpn, 4]
                bbox_preds_rpn = rpn_bbox_i[sampled_pos]       # [N_pos_rpn, 4]
                rpn_box_loss = F.smooth_l1_loss(bbox_preds_rpn, bbox_targets_rpn)
            else:
                rpn_box_loss = torch.tensor(0.0, device=device)

            # ------------------------------------------------------------------
            # Step 2: Generate Proposals & Apply NMS
            # ------------------------------------------------------------------
            scores = F.softmax(rpn_cls_i, dim=1)[:, 1]          # [N_anchors]
            decoded_boxes = box_decode(anchors, rpn_bbox_i).detach()  # [N_anchors, 4]
            decoded_boxes[:, [0, 2]] = decoded_boxes[:, [0, 2]].clamp(0, images[i].shape[2])
            decoded_boxes[:, [1, 3]] = decoded_boxes[:, [1, 3]].clamp(0, images[i].shape[1])

            keep_idx = nms(decoded_boxes, scores, NMS_THRESHOLD)
            proposals = decoded_boxes[keep_idx[:NUM_POST_NMS]] # [N_proposals, 4]

            # ------------------------------------------------------------------
            # Step 3: Fast R-CNN Detection Head Target Assignment
            # ------------------------------------------------------------------
            det_labels, det_matched, det_sampled = assign_labels(
                proposals, gt, labels_gt,
                ROI_POS_THRESHOLD, ROI_NEG_THRESHOLD, max_samples=ROI_BATCH_SIZE, pos_fraction=ROI_POS_RATIO
            )
            sampled_proposals = proposals[det_sampled]          # [N_roi, 4]

            # RoI Pooling & Classification/Bounding Box Head
            roi_features = roi_pool(features[i], sampled_proposals, output_size=ROI_POOL_SIZE)  # [N_roi, 256, 7, 7]
            flat = roi_features.view(roi_features.size(0), -1)  # [N_roi, 256*7*7]
            h = self.act(self.fc1(flat))                       # [N_roi, 1024]
            h = self.act(self.fc2(h))                          # [N_roi, 1024]
            cls_logits = self.cls_score(h)                     # [N_roi, Num_Classes]
            bbox_deltas = self.bbox_pred(h)                    # [N_roi, Num_Classes * 4]

            det_cls_loss = F.cross_entropy(cls_logits, det_labels[det_sampled])

            pos_mask_det = det_labels[det_sampled] > 0
            det_box_loss = torch.tensor(0.0, device=device)
            mask_loss = torch.tensor(0.0, device=device)

            if pos_mask_det.sum() > 0:
                pos_labels = det_labels[det_sampled][pos_mask_det]  # [N_pos_det]
                pos_proposals = sampled_proposals[pos_mask_det]     # [N_pos_det, 4]
                pos_gt_boxes = gt[det_matched[det_sampled][pos_mask_det]]  # [N_pos_det, 4]

                bbox_targets_det = box_encode(pos_gt_boxes, pos_proposals) # [N_pos_det, 4]
                bbox_preds_det = bbox_deltas[pos_mask_det].view(-1, self.num_classes, 4)  # [N_pos_det, Num_Classes, 4]
                bbox_preds_det = bbox_preds_det[torch.arange(len(pos_labels)), pos_labels]  # [N_pos_det, 4]
                det_box_loss = F.smooth_l1_loss(bbox_preds_det, bbox_targets_det)

                # --------------------------------------------------------------
                # Step 4: Mask Head Branch
                # --------------------------------------------------------------
                mask_roi_feats = roi_pool(features[i], pos_proposals, output_size=ROI_POOL_SIZE)  # [N_pos_det, 256, 7, 7]
                mask_logits = self.mask_head(mask_roi_feats)                                       # [N_pos_det, Num_Classes, 28, 28]
                
                mask_targets_gt = mask_target(pos_proposals, det_matched[det_sampled][pos_mask_det], gt_masks, mask_size=MASK_SIZE * 2)  # [N_pos_det, 28, 28]
                mask_preds_pos = mask_logits[torch.arange(len(pos_labels)), pos_labels]           # [N_pos_det, 28, 28]
                mask_loss = F.binary_cross_entropy_with_logits(mask_preds_pos, mask_targets_gt)

            losses["loss_rpn_cls"] += rpn_cls_loss
            losses["loss_rpn_box_reg"] += rpn_box_loss
            losses["loss_classifier"] += det_cls_loss
            losses["loss_box_reg"] += det_box_loss
            losses["loss_mask"] += mask_loss

        # Average losses over batch dimension
        for k in losses:
            losses[k] = losses[k] / B

        return losses


# ==============================================================================
# SECTION 8: Execution Entry Point (Training Loop)
# ==============================================================================

def main():
    print(f"Initializing Synthetic Instance Segmentation Dataset...")
    dataset = SyntheticInstanceDataset(num_samples=20, num_classes=NUM_CLASSES, size=128)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(f"Initializing Mask R-CNN Model on {DEVICE}...")
    model = MaskRCNN(num_classes=NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    print("Starting Training Execution Loop...\n")
    model.train()
    for epoch in range(EPOCHS):
        for step, (images, targets) in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            total_loss.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | Step [{step + 1}/{len(loader)}] | "
                f"Total Loss: {total_loss.item():.4f} | "
                f"RPN Cls: {loss_dict['loss_rpn_cls'].item():.4f} | "
                f"RPN Box: {loss_dict['loss_rpn_box_reg'].item():.4f} | "
                f"Det Cls: {loss_dict['loss_classifier'].item():.4f} | "
                f"Det Box: {loss_dict['loss_box_reg'].item():.4f} | "
                f"Mask: {loss_dict['loss_mask'].item():.4f}"
            )

    print("\nTraining execution completed successfully.")


if __name__ == "__main__":
    main()