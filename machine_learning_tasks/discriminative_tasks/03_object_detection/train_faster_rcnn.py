"""
任务编号与名称: Task 03 - 目标检测 (Object Detection)
代表架构/算法: Faster R-CNN (Two-Stage Detector, 纯 PyTorch 实现，不依赖 torchvision.detection)
领域分类: 计算机视觉 / 目标检测 (Computer Vision / Object Detection)

1. 核心思想与机制:
   - Faster R-CNN 采用两阶段 (Two-Stage) 架构：
     a) Backbone: 利用深度卷积网络 (ResNet-style) 提取输入图像的特征图 (Feature Map)。
     b) Region Proposal Network (RPN): 在特征图上以 1/16 步长生成多尺度/多宽高比 Anchor，
        通过 1x1 卷积预测 Anchor 的二分类 (前景/背景) 分数及边界框偏移量 (BBox Offsets)，
        经由 NMS 筛选出高质量的候选区域 (Proposals)。
     c) RoI Pooling & Classification/Regression Head: 利用 RoI 池化将变长 Proposals 
        映射为固定尺寸 (7x7) 特征，送入全连接网络预测具体的类别 (Class Head) 与精细框坐标 (Box Head)。

2. 数学公式/目标函数:
   - 总体损失函数 (Multi-task Loss):
     L({p_i}, {t_i}) = L_cls(RPN) + λ1 * L_reg(RPN) + L_cls(Fast R-CNN) + λ2 * L_reg(Fast R-CNN)

   - 边界框编码/解码 (Box Bounding Box Regression Target):
     t_x = (x - x_a) / w_a,          t_y = (y - y_a) / h_a
     t_w = log(w / w_a),             t_h = log(h / h_a)
     代码变量对应:
     gx, gy, gw, gh <-> 真实框/目标框坐标 (GT Boxes / References)
     px, py, pw, ph <-> 锚框/候选框坐标 (Anchors / Proposals)
     delta [t_x, t_y, t_w, t_h] <-> 偏移量预测 (Offsets)

   - 损失函数:
     - 分类损失: Cross Entropy Loss -> L_cls = -log(p_i)
     - 回归损失: Smooth L1 Loss -> SmoothL1(x) = 0.5 * x^2 if |x| < 1 else |x| - 0.5

3. 数据输入输出规范:
   - 输入 (Inputs):
     - images: List[Tensor], 长度 B, 单个张量 Shape [C, H, W] (C=3, H=256, W=256)
     - targets: List[Dict], 包含 'boxes' [N_gt, 4] 与 'labels' [N_gt]
   - 输出 (Outputs):
     - Training: Dict[str, Tensor], 包含 4 项 Loss ('loss_rpn_cls', 'loss_rpn_box_reg', 'loss_classifier', 'loss_box_reg')
     - Inference: List[Tensor], 每张图像预测得到的候选框 [N_post_nms, 4]
"""

import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===================================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ===================================================================================
BATCH_SIZE: int = 4
EPOCHS: int = 2
LR: float = 5e-3
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 5e-4
NUM_CLASSES: int = 3  # 包括背景 (0: Background, 1: Class_1, 2: Class_2)
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anchor & RPN 配置
ANCHOR_SCALES: List[int] = [64, 128, 256]
ANCHOR_RATIOS: List[float] = [0.5, 1.0, 2.0]
STRIDE: int = 16  # 下采样倍率
RPN_POS_THRESHOLD: float = 0.7
RPN_NEG_THRESHOLD: float = 0.3

# RoI & Fast R-CNN 配置
ROI_POS_THRESHOLD: float = 0.5
ROI_NEG_THRESHOLD: float = 0.5
NMS_THRESHOLD: float = 0.7
NUM_POST_NMS: int = 128
ROI_POOL_SIZE: int = 7


# ===================================================================================
# 4. 数据处理与 Utils 工具函数 (Data Pipeline & Utils)
# ===================================================================================
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两组边界框之间的交并比 (IoU, Intersection over Union)。

    数学原理:
        IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)

    Args:
        boxes1 (Tensor): 第一组边界框，shape: [N, 4], 格式为 [x1, y1, x2, y2]
        boxes2 (Tensor): 第二组边界框，shape: [M, 4], 格式为 [x1, y1, x2, y2]

    Inputs:
        boxes1: [N, 4]
        boxes2: [M, 4]

    Outputs:
        ious (Tensor): 交并比矩阵，shape: [N, M]
    """
    # area1: [N], area2: [M]
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # 广播计算交集区域坐标: [N, M]
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # [N, M]
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])  # [N, M]
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])  # [N, M]
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])  # [N, M]

    inter_w = (inter_x2 - inter_x1).clamp(min=0)                  # [N, M]
    inter_h = (inter_y2 - inter_y1).clamp(min=0)                  # [N, M]
    inter = inter_w * inter_h                                     # [N, M]

    union = area1[:, None] + area2[None, :] - inter               # [N, M]
    return inter / (union + 1e-6)                                 # [N, M]


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    非极大值抑制 (Non-Maximum Suppression, NMS)，去除重叠度过高的候选框。

    Args:
        boxes (Tensor): 边界框坐标，shape: [N, 4]
        scores (Tensor): 各框的前景得分，shape: [N]
        threshold (float): IoU 重叠度阈值

    Inputs:
        boxes: [N, 4]
        scores: [N]

    Outputs:
        keep (Tensor): 保留框的索引向量，shape: [K] (K <= N)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)  # [N] 降序排列索引
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        
        # 计算当前得分最高框与其余框的 IoU
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]  # [len(order)-1]
        mask = ious <= threshold                             # [len(order)-1]
        order = order[1:][mask]                              # 过滤掉高重叠框

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def generate_anchors(feature_h: int, feature_w: int, stride: int = STRIDE) -> torch.Tensor:
    """
    基于特征图网格生成多尺度、多宽高比的 Anchor 集合。

    Args:
        feature_h (int): 特征图高度 H_feat
        feature_w (int): 特征图宽度 W_feat
        stride (int): 下采样步长，默认 16

    Outputs:
        anchors (Tensor): 全图 Anchor 坐标，shape: [A, 4], 其中 A = H_feat * W_feat * K (K=9)
    """
    device = torch.device("cpu")
    shifts_x = torch.arange(0, feature_w, device=device) * stride  # [W_feat]
    shifts_y = torch.arange(0, feature_h, device=device) * stride  # [H_feat]
    
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij") # [H_feat, W_feat]
    shift_x = shift_x.reshape(-1)  # [H_feat * W_feat]
    shift_y = shift_y.reshape(-1)  # [H_feat * W_feat]
    
    # 中心点坐标 [x_ctr, y_ctr, x_ctr, y_ctr]
    centers = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).float() # [N_grid, 4]

    anchors = []
    for scale in ANCHOR_SCALES:
        for ratio in ANCHOR_RATIOS:
            h = scale / math.sqrt(ratio)
            w = scale * math.sqrt(ratio)
            # 基础偏移量 [x1, y1, x2, y2]
            base = torch.tensor([-w / 2, -h / 2, w / 2, h / 2], device=device)
            anchors.append(centers + base)

    anchors = torch.cat(anchors, dim=0)  # [K * N_grid, 4]
    return anchors


def box_encode(reference: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
    """
    将 GT 框坐标编码为相对于 Proposal / Anchor 的偏移量回归目标。

    数学公式映射:
        t_x = (g_x - p_x) / p_w
        t_y = (g_y - p_y) / p_h
        t_w = log(g_w / p_w)
        t_h = log(g_h / p_h)

    Args:
        reference (Tensor): 真实目标框 GT Boxes，shape: [N, 4]
        proposal (Tensor): 候选框 Proposals / Anchors，shape: [N, 4]

    Outputs:
        targets (Tensor): 编码后的回归目标，shape: [N, 4]
    """
    px = (proposal[:, 0] + proposal[:, 2]) / 2.0
    py = (proposal[:, 1] + proposal[:, 3]) / 2.0
    pw = proposal[:, 2] - proposal[:, 0]
    ph = proposal[:, 3] - proposal[:, 1]

    gx = (reference[:, 0] + reference[:, 2]) / 2.0
    gy = (reference[:, 1] + reference[:, 3]) / 2.0
    gw = reference[:, 2] - reference[:, 0]
    gh = reference[:, 3] - reference[:, 1]

    targets = torch.stack([
        (gx - px) / pw,
        (gy - py) / ph,
        torch.log(gw / (pw + 1e-6) + 1e-6),
        torch.log(gh / (ph + 1e-6) + 1e-6),
    ], dim=1)  # [N, 4]
    return targets


def box_decode(proposal: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    根据预测的 delta 偏移量将 Proposal / Anchor 解码为真实的边界框坐标。

    数学公式映射:
        g_x = t_x * p_w + p_x
        g_y = t_y * p_h + p_y
        g_w = exp(t_w) * p_w
        g_h = exp(t_h) * p_h

    Args:
        proposal (Tensor): 候选框/锚框坐标，shape: [N, 4]
        delta (Tensor): 预测偏移量，shape: [N, 4]

    Outputs:
        boxes (Tensor): 解码后的预测框坐标，shape: [N, 4]
    """
    px = (proposal[:, 0] + proposal[:, 2]) / 2.0
    py = (proposal[:, 1] + proposal[:, 3]) / 2.0
    pw = proposal[:, 2] - proposal[:, 0]
    ph = proposal[:, 3] - proposal[:, 1]

    gx = delta[:, 0] * pw + px
    gy = delta[:, 1] * ph + py
    gw = torch.exp(delta[:, 2]) * pw
    gh = torch.exp(delta[:, 3]) * ph

    return torch.stack([
        gx - gw / 2.0, gy - gh / 2.0,
        gx + gw / 2.0, gy + gh / 2.0,
    ], dim=1)  # [N, 4]


def roi_pool(feature: torch.Tensor, boxes: torch.Tensor, output_size: int = ROI_POOL_SIZE) -> torch.Tensor:
    """
    对单张图像特征图上的变长 RoI 区域执行自适应最大池化，输出固定维度特征。

    Args:
        feature (Tensor): 单张图特征图，shape: [C, H_feat, W_feat]
        boxes (Tensor): 映射到原图尺度的 RoI 候选框，shape: [N, 4]
        output_size (int): 池化目标尺寸，默认 7

    Outputs:
        rois (Tensor): 池化后的固定尺度特征，shape: [N, C, output_size, output_size]
    """
    if boxes.numel() == 0:
        return torch.zeros((0, feature.size(0), output_size, output_size), device=feature.device)

    boxes_scaled = boxes / STRIDE  # 缩放到特征图尺度 [N, 4]
    rois = []
    
    for box in boxes_scaled:
        x1, y1, x2, y2 = box.long()
        x1 = x1.clamp(0, feature.size(2) - 1)
        y1 = y1.clamp(0, feature.size(1) - 1)
        x2 = x2.clamp(x1 + 1, feature.size(2))
        y2 = y2.clamp(y1 + 1, feature.size(1))
        
        crop = feature[:, y1:y2, x1:x2]  # [C, h_crop, w_crop]
        roi = F.adaptive_max_pool2d(crop, (output_size, output_size))  # [C, 7, 7]
        rois.append(roi)

    return torch.stack(rois, dim=0)  # [N, C, 7, 7]


def assign_labels(
    proposals: torch.Tensor, 
    gt_boxes: torch.Tensor, 
    gt_labels: torch.Tensor, 
    pos_threshold: float, 
    neg_threshold: float
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    为 Proposals/Anchors 分配正负样本标签，并按照固定比例进行平衡采样。

    Args:
        proposals (Tensor): 候选框，shape: [N_prop, 4]
        gt_boxes (Tensor): 真实框，shape: [N_gt, 4]
        gt_labels (Tensor): 真实类别标签，shape: [N_gt]
        pos_threshold (float): 正样本 IoU 阈值
        neg_threshold (float): 负样本 IoU 阈值

    Outputs:
        labels (Tensor): 分配后的类别标签向量，shape: [N_prop]
        matched_gt (Tensor): 每个 Proposal 匹配到的 GT 框索引，shape: [N_prop]
        sampled (Tensor): 被选中的采样索引向量，shape: [N_sampled]
    """
    if gt_boxes.numel() == 0:
        labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
        return labels, None, torch.arange(len(proposals), device=proposals.device)

    ious = box_iou(proposals, gt_boxes)  # [N_prop, N_gt]
    max_iou, matched_gt = ious.max(dim=1)  # [N_prop], [N_prop]
    _, best_anchor_per_gt = ious.max(dim=0)  # [N_gt]强制保留每个 GT 最大 IoU 框

    labels = torch.zeros(len(proposals), dtype=torch.long, device=proposals.device)
    pos_mask = (max_iou >= pos_threshold)
    pos_mask[best_anchor_per_gt] = True
    labels[pos_mask] = gt_labels[matched_gt[pos_mask]]

    neg_mask = (~pos_mask) & (max_iou < neg_threshold)
    labels[neg_mask] = 0  # 0 标记为背景

    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    # 正负样本均衡采样 (最多 64 正样本，总计 256)
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


# ===================================================================================
# 5. 核心子模块 (Sub-components: Backbone & Heads)
# ===================================================================================
class SimpleBackbone(nn.Module):
    """
    轻量级卷积骨干网络 (CNN Backbone)，负责提取输入图像的高层抽象特征。

    结构设计:
        Input [B, 3, 256, 256] -> Conv1 -> Layer1 -> Layer2 -> Layer3 -> Output [B, 256, 16, 16]

    Outputs:
        x (Tensor): 缩放到 1/16 尺寸的特征图，shape: [B, 256, H/16, W/16]
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # [B, 64, 128, 128]
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.SiLU(inplace=True)  # 换用现代高效激活函数 SiLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)               # [B, 64, 64, 64]

        self.layer1 = self._make_block(64, 64, stride=1)                                # [B, 64, 64, 64]
        self.layer2 = self._make_block(64, 128, stride=2)                               # [B, 128, 32, 32]
        self.layer3 = self._make_block(128, 256, stride=2)                              # [B, 256, 16, 16]

    def _make_block(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x (Tensor): 输入图像张量, shape: [B, 3, H, W]
        Outputs:
            feat (Tensor): 特征图张量, shape: [B, 256, H/16, W/16]
        """
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class RPNHead(nn.Module):
    """
    区域提议网络头 (Region Proposal Network Head)。

    数学原理:
        对特征图每个位置应用 3x3 卷积后，分别分支预测：
        1) 2 分类 logits (前景/背景): [B, H*W*K, 2]
        2) 4 参数边界框偏移 delta: [B, H*W*K, 4]

    Args:
        in_channels (int): 输入通道数，例如 256
        num_anchors (int): 每个网格点的 Anchor 数量，K = len(SCALES) * len(RATIOS) = 9
    """
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors * 2, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x (Tensor): 特征图，shape: [B, C, H_feat, W_feat]

        Outputs:
            cls (Tensor): RPN 二分类 logits，shape: [B, H_feat * W_feat * K, 2]
            bbox (Tensor): RPN 框偏移量，shape: [B, H_feat * W_feat * K, 4]
        """
        B, _, H, W = x.shape
        t = F.silu(self.conv(x))                             # [B, C, H, W]
        cls = self.cls_logits(t)                             # [B, K*2, H, W]
        bbox = self.bbox_pred(t)                             # [B, K*4, H, W]

        # 调整维度以适配后续采样与损失计算
        cls = cls.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, 2)  # [B, Total_Anchors, 2]
        bbox = bbox.permute(0, 2, 3, 1).reshape(B, H * W * self.num_anchors, 4) # [B, Total_Anchors, 4]
        return cls, bbox


# ===================================================================================
# 6. 顶层模型主体 (Top-level Model Architecture)
# ===================================================================================
class FasterRCNN(nn.Module):
    """
    Faster R-CNN 端到端目标检测框架。

    Args:
        num_classes (int): 类别数（含背景）
    """
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = SimpleBackbone()
        self.num_anchors = len(ANCHOR_SCALES) * len(ANCHOR_RATIOS)
        self.rpn = RPNHead(256, self.num_anchors)

        # 检测头全连接层 (Fast R-CNN Head)
        self.fc1 = nn.Linear(256 * ROI_POOL_SIZE * ROI_POOL_SIZE, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

    def forward(
        self, 
        images: List[torch.Tensor], 
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ):
        """
        Inputs:
            images (List[Tensor]): 图像列表，每张 Shape [3, H, W]
            targets (Optional[List[Dict]]): 训练目标列表，包含 'boxes' [N_gt, 4] 和 'labels' [N_gt]

        Outputs:
            training: Dict[str, Tensor] -> 4 项 Multi-task Loss 字典
            eval: List[Tensor] -> 预测 Proposal 结果列表
        """
        img_batch = torch.stack(images, dim=0)               # [B, 3, H, W]
        features = self.backbone(img_batch)                  # [B, 256, H_feat, W_feat]
        B, _, H, W = features.shape

        device = features.device
        anchors = generate_anchors(H, W, STRIDE).to(device)  # [A, 4]

        rpn_cls, rpn_bbox = self.rpn(features)               # [B, A, 2], [B, A, 4]

        losses = {}
        proposals_all = []

        if self.training and targets is not None:
            for i in range(B):
                gt = targets[i]["boxes"].to(device)
                labels_gt = targets[i]["labels"].to(device)

                # -------------------------------------------------------------
                # Stage 1: RPN 损失计算 & Proposal 生成
                # -------------------------------------------------------------
                rpn_labels, matched, sampled = assign_labels(
                    anchors, gt, torch.ones(len(gt), device=device, dtype=torch.long),
                    RPN_POS_THRESHOLD, RPN_NEG_THRESHOLD,
                )
                rpn_cls_i = rpn_cls[i]                        # [A, 2]
                rpn_bbox_i = rpn_bbox[i]                      # [A, 4]

                # RPN 框回归 Loss
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

                # 根据 RPN 预测结果生成 Proposals 送入 Stage 2
                scores = F.softmax(rpn_cls_i, dim=1)[:, 1]
                decoded = box_decode(anchors, rpn_bbox_i).detach()
                decoded[:, [0, 2]] = decoded[:, [0, 2]].clamp(0, images[i].shape[2])
                decoded[:, [1, 3]] = decoded[:, [1, 3]].clamp(0, images[i].shape[1])

                keep = nms(decoded, scores, NMS_THRESHOLD)
                proposals = decoded[keep[:NUM_POST_NMS]]       # [N_prop, 4]

                # -------------------------------------------------------------
                # Stage 2: Fast R-CNN 检测头标签分配与损失计算
                # -------------------------------------------------------------
                det_labels, det_matched, det_sampled = assign_labels(
                    proposals, gt, labels_gt,
                    ROI_POS_THRESHOLD, ROI_NEG_THRESHOLD,
                )
                proposals_all.append(proposals[det_sampled])  # [N_sampled_roi, 4]

                # RoI 池化提取特征
                roi_features = roi_pool(features[i], proposals_all[-1]) # [N_roi, 256, 7, 7]
                flat = roi_features.view(roi_features.size(0), -1)      # [N_roi, 256*7*7]
                h = F.silu(self.fc1(flat))                              # [N_roi, 1024]
                h = F.silu(self.fc2(h))                                 # [N_roi, 1024]
                cls_logits = self.cls_score(h)                          # [N_roi, NUM_CLASSES]
                bbox_deltas = self.bbox_pred(h)                         # [N_roi, NUM_CLASSES * 4]

                det_cls_loss = F.cross_entropy(cls_logits, det_labels[det_sampled])

                pos_det = det_labels[det_sampled] > 0
                if pos_det.sum() > 0:
                    pos_labels = det_labels[det_sampled][pos_det]
                    pos_proposals = proposals_all[-1][pos_det]
                    pos_gt = gt[det_matched[det_sampled][pos_det]]
                    bbox_targets = box_encode(pos_gt, pos_proposals)
                    
                    bbox_preds = bbox_deltas[pos_det].view(-1, self.num_classes, 4)
                    bbox_preds = bbox_preds[torch.arange(len(pos_labels)), pos_labels]
                    det_box_loss = F.smooth_l1_loss(bbox_preds, bbox_targets)
                else:
                    det_box_loss = torch.tensor(0.0, device=device)

                # 累加 Loss
                for k, v in [
                    ("loss_rpn_cls", rpn_cls_loss),
                    ("loss_rpn_box_reg", rpn_box_loss),
                    ("loss_classifier", det_cls_loss),
                    ("loss_box_reg", det_box_loss),
                ]:
                    losses[k] = losses.get(k, torch.tensor(0.0, device=device)) + v

            for k in losses:
                losses[k] = losses[k] / B
            return losses
        else:
            # 推理阶段: 提取 proposals
            for i in range(B):
                scores = F.softmax(rpn_cls[i], dim=1)[:, 1]
                decoded = box_decode(anchors, rpn_bbox[i])
                keep = nms(decoded, scores, NMS_THRESHOLD)
                proposals_all.append(decoded[keep[:NUM_POST_NMS]])
            return proposals_all


# ===================================================================================
# 7. 合成数据集 (Synthetic Dataset)
# ===================================================================================
class SyntheticDetectionDataset(Dataset):
    """
    合成目标检测数据集，用于功能验证与流水线测试。
    """
    def __init__(self, num_samples: int = 100, num_classes: int = NUM_CLASSES, size: int = 256) -> None:
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


# ===================================================================================
# 8. 训练/推理逻辑与入口 (Training Pipeline & Execution)
# ===================================================================================
def main() -> None:
    print(f"[Init] Initializing Faster R-CNN Detection Model on Device: {DEVICE}")
    dataset = SyntheticDetectionDataset(num_samples=16)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = FasterRCNN(num_classes=NUM_CLASSES).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    model.train()
    print("[Start] Training Loop Started...")
    for epoch in range(EPOCHS):
        for step, (images, targets) in enumerate(loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] Step [{step + 1}/{len(loader)}] | "
                f"Total Loss: {total_loss.item():.4f} | "
                f"RPN Cls: {loss_dict['loss_rpn_cls'].item():.4f} | "
                f"RPN Box: {loss_dict['loss_rpn_box_reg'].item():.4f} | "
                f"Head Cls: {loss_dict['loss_classifier'].item():.4f} | "
                f"Head Box: {loss_dict['loss_box_reg'].item():.4f}"
            )

    print("[Finished] Model Training Pipeline Executed Successfully!")


if __name__ == "__main__":
    main()