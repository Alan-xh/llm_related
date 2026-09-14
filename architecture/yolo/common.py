"""Shared, small PyTorch components used by the YOLO teaching models.

The code in this module intentionally favors readable tensor contracts over
production throughput.  It is sufficient for shape experiments and tiny
synthetic training runs, but it is not a replacement for the official
Ultralytics or Darknet implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DetectorConfig:
    num_classes: int = 3
    width: int = 16
    image_size: int = 64
    strides: tuple[int, ...] = (8, 16, 32)


class ConvBNAct(nn.Module):
    """Convolution, normalization, and activation used throughout the models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        if activation == "leaky":
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = max(8, int(channels * expansion))
        self.block = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, channels, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class CSPBlock(nn.Module):
    """A compact CSP/C3-like block with a partial residual path."""

    def __init__(self, channels: int, depth: int = 1) -> None:
        super().__init__()
        hidden = max(8, channels // 2)
        self.left = ConvBNAct(channels, hidden, 1)
        self.right = ConvBNAct(channels, hidden, 1)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden) for _ in range(depth)))
        self.out = ConvBNAct(hidden * 2, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(torch.cat((self.blocks(self.left(x)), self.right(x)), dim=1))


class SPPF(nn.Module):
    """Fast spatial pyramid pooling used by YOLOv5-style models."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 2)
        self.reduce = ConvBNAct(channels, hidden, 1)
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.expand = ConvBNAct(hidden * 4, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.expand(torch.cat((x, y1, y2, y3), dim=1))


class TinyBackbone(nn.Module):
    """Three-level CNN feature pyramid with strides 8, 16, and 32."""

    def __init__(self, width: int = 16, style: str = "plain") -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = width, width * 2, width * 4, width * 8, width * 8
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.down1 = ConvBNAct(c1, c2, 3, 2)
        self.block1 = CSPBlock(c2, 1 if style in {"csp", "sppf"} else 0)
        self.down2 = ConvBNAct(c2, c3, 3, 2)
        self.block2 = CSPBlock(c3, 2 if style in {"csp", "sppf"} else 1)
        self.down3 = ConvBNAct(c3, c4, 3, 2)
        self.block3 = CSPBlock(c4, 2 if style in {"csp", "sppf"} else 1)
        self.down4 = ConvBNAct(c4, c5, 3, 2)
        self.block4 = SPPF(c5) if style == "sppf" else CSPBlock(c5, 1)
        self.out_channels = (c3, c4, c5)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        x = self.block1(self.down1(x))
        p3 = self.block2(self.down2(x))
        p4 = self.block3(self.down3(p3))
        p5 = self.block4(self.down4(p4))
        return [p3, p4, p5]


class FeaturePyramidNeck(nn.Module):
    """Small top-down/lateral neck, sufficient to show multi-scale fusion."""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        c3, c4, c5 = channels
        self.lateral4 = ConvBNAct(c5, c4, 1)
        self.lateral3 = ConvBNAct(c4, c3, 1)
        self.out4 = ConvBNAct(c4, c4, 3)
        self.out3 = ConvBNAct(c3, c3, 3)

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        p3, p4, p5 = features
        p4 = self.out4(p4 + F.interpolate(self.lateral4(p5), size=p4.shape[-2:], mode="nearest"))
        p3 = self.out3(p3 + F.interpolate(self.lateral3(p4), size=p3.shape[-2:], mode="nearest"))
        return [p3, p4, p5]


class AnchorFreeHead(nn.Module):
    """Decoupled objectness/classification and box regression head."""

    def __init__(self, channels: int, num_classes: int, regression_bins: int = 1) -> None:
        super().__init__()
        self.stem = ConvBNAct(channels, channels, 3)
        self.box = nn.Conv2d(channels, 4 * regression_bins, 1)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.classification = nn.Conv2d(channels, num_classes, 1)
        self.regression_bins = regression_bins

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        return torch.cat((self.box(x), self.objectness(x), self.classification(x)), dim=1)


class MultiScaleDetector(nn.Module):
    """Anchor-free detector returning [B, 5+C, H_i, W_i] tensors."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        backbone_style: str = "plain",
        use_neck: bool = True,
        regression_bins: int = 1,
    ) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        self.backbone = TinyBackbone(self.config.width, backbone_style)
        self.neck = FeaturePyramidNeck(self.backbone.out_channels) if use_neck else nn.Identity()
        self.heads = nn.ModuleList(
            AnchorFreeHead(channels, self.config.num_classes, regression_bins)
            for channels in self.backbone.out_channels
        )
        self.regression_bins = regression_bins

    def forward(self, images: Tensor) -> list[Tensor]:
        features = self.backbone(images)
        features = self.neck(features)
        return [head(feature) for head, feature in zip(self.heads, features)]


class YoloV1Detector(nn.Module):
    """The original single-grid YOLOv1-style detector."""

    def __init__(self, num_classes: int = 3, grid_size: int = 7, width: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.boxes_per_cell = 2
        self.backbone = nn.Sequential(
            ConvBNAct(3, width, 3, 2, "leaky"),
            ConvBNAct(width, width * 2, 3, 2, "leaky"),
            ConvBNAct(width * 2, width * 4, 3, 2, "leaky"),
            ConvBNAct(width * 4, width * 8, 3, 2, "leaky"),
            ConvBNAct(width * 8, width * 8, 3, 2, "leaky"),
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.head = nn.Conv2d(width * 8, num_classes + self.boxes_per_cell * 5, 1)

    def forward(self, images: Tensor) -> Tensor:
        output = self.head(self.pool(self.backbone(images)))
        return output.permute(0, 2, 3, 1).contiguous()


class DualHeadDetector(nn.Module):
    """A compact one-to-many plus one-to-one detector for YOLOv10/26 lessons."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        backbone_style: str = "csp",
        progressive_loss: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or DetectorConfig()
        self.backbone = TinyBackbone(self.config.width, backbone_style)
        self.neck = FeaturePyramidNeck(self.backbone.out_channels)
        self.one_to_many = nn.ModuleList(
            AnchorFreeHead(c, self.config.num_classes) for c in self.backbone.out_channels
        )
        self.one_to_one = nn.ModuleList(
            AnchorFreeHead(c, self.config.num_classes) for c in self.backbone.out_channels
        )
        self.progressive_loss = progressive_loss

    def forward(self, images: Tensor) -> dict[str, list[Tensor]]:
        features = self.neck(self.backbone(images))
        return {
            "one_to_many": [head(x) for head, x in zip(self.one_to_many, features)],
            "one_to_one": [head(x) for head, x in zip(self.one_to_one, features)],
        }


def _split_anchor_free(prediction: Tensor, regression_bins: int = 1) -> tuple[Tensor, Tensor, Tensor]:
    box_end = 4 * regression_bins
    return prediction[:, :box_end], prediction[:, box_end : box_end + 1], prediction[:, box_end + 1 :]


def _distribution_to_distance(box: Tensor, regression_bins: int) -> Tensor:
    if regression_bins == 1:
        return F.softplus(box)
    values = torch.arange(regression_bins, device=box.device, dtype=box.dtype)
    batch, _, height, width = box.shape
    distribution = box.reshape(batch, 4, regression_bins, height, width).softmax(2)
    values = values.view(1, 1, regression_bins, 1, 1)
    return (distribution * values).sum(2)


def _flatten_predictions(
    predictions: Sequence[Tensor],
    strides: Sequence[int],
    regression_bins: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    boxes, objectness, classes = [], [], []
    for prediction, stride in zip(predictions, strides):
        box, obj, cls = _split_anchor_free(prediction, regression_bins)
        box = _distribution_to_distance(box, regression_bins)
        batch, _, height, width = box.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=box.device),
            torch.arange(width, device=box.device),
            indexing="ij",
        )
        centers = torch.stack((xx + 0.5, yy + 0.5), dim=-1).reshape(1, -1, 2)
        centers = centers * stride
        distance = box.permute(0, 2, 3, 1).reshape(batch, -1, 4) * stride
        xyxy = torch.cat(
            (centers - distance[..., :2], centers + distance[..., 2:]),
            dim=-1,
        )
        boxes.append(xyxy)
        objectness.append(obj.sigmoid().flatten(2).transpose(1, 2))
        classes.append(cls.sigmoid().flatten(2).transpose(1, 2))
    return torch.cat(boxes, 1), torch.cat(objectness, 1), torch.cat(classes, 1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-6)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    keep: list[Tensor] = []
    order = scores.argsort(descending=True)
    while order.numel():
        first = order[:1]
        keep.append(first)
        if order.numel() == 1:
            break
        overlaps = box_iou(boxes[first], boxes[order[1:]]).squeeze(0)
        order = order[1:][overlaps <= iou_threshold]
    return torch.cat(keep) if keep else order


@torch.no_grad()
def decode_predictions(
    predictions: Sequence[Tensor],
    image_size: tuple[int, int] = (64, 64),
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    strides: Sequence[int] = (8, 16, 32),
    regression_bins: int = 1,
    apply_nms: bool = True,
    max_detections: int = 100,
) -> list[Tensor]:
    """Decode predictions into per-image [x1, y1, x2, y2, score, class] tensors."""
    boxes, objectness, classes = _flatten_predictions(predictions, strides, regression_bins)
    image_h, image_w = image_size
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, image_w)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, image_h)
    results: list[Tensor] = []
    for image_boxes, image_objectness, image_classes in zip(boxes, objectness, classes):
        scores, class_ids = (image_objectness * image_classes).max(-1)
        keep = scores >= confidence_threshold
        image_boxes, scores, class_ids = image_boxes[keep], scores[keep], class_ids[keep]
        if image_boxes.numel() == 0:
            results.append(boxes.new_zeros((0, 6)))
            continue
        if apply_nms:
            kept_indices: list[Tensor] = []
            for class_id in class_ids.unique():
                indices = torch.where(class_ids == class_id)[0]
                kept_indices.append(indices[nms(image_boxes[indices], scores[indices], iou_threshold)])
            keep_indices = torch.cat(kept_indices) if kept_indices else scores.argsort(descending=True)[:0]
        else:
            keep_indices = scores.argsort(descending=True)[:max_detections]
        keep_indices = keep_indices[scores[keep_indices].argsort(descending=True)[:max_detections]]
        results.append(
            torch.cat(
                (image_boxes[keep_indices], scores[keep_indices, None], class_ids[keep_indices, None].float()),
                dim=1,
            )
        )
    return results


def yolo_loss(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
    box_weight: float = 5.0,
    objectness_weight: float = 1.0,
    class_weight: float = 1.0,
    regression_bins: int = 1,
) -> Tensor:
    """Loss for the teaching target format [tx, ty, tw, th, obj, one-hot classes]."""
    total = predictions[0].new_zeros(())
    for prediction, target in zip(predictions, targets):
        box_pred, objectness_pred, class_pred = _split_anchor_free(prediction, regression_bins)
        box_pred = _distribution_to_distance(box_pred, regression_bins)
        box_target = target[:, :4]
        objectness_target = target[:, 4:5]
        class_target = target[:, 5:]
        object_mask = objectness_target.bool()
        box_loss = F.smooth_l1_loss(box_pred, box_target, reduction="none").sum(1, keepdim=True)
        box_loss = (box_loss * objectness_target).mean()
        obj_loss = F.binary_cross_entropy_with_logits(objectness_pred, objectness_target)
        class_loss = F.binary_cross_entropy_with_logits(
            class_pred, class_target, reduction="none"
        )
        class_loss = class_loss.masked_select(object_mask.expand_as(class_loss))
        class_loss = class_loss.mean() if class_loss.numel() else class_pred.new_zeros(())
        total = total + box_weight * box_loss + objectness_weight * obj_loss + class_weight * class_loss
    return total / max(1, len(predictions))


def yolo_v1_loss(
    predictions: Tensor,
    targets: Tensor,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
) -> Tensor:
    """Simplified YOLOv1 loss for [B, S, S, C + 10] predictions."""
    batch, grid, _, channels = predictions.shape
    classes = channels - 10
    pred_classes = predictions[..., :classes]
    pred_boxes = predictions[..., classes:].reshape(batch, grid, grid, 2, 5)
    target_classes = targets[..., :classes]
    target_boxes = targets[..., classes:].reshape(batch, grid, grid, 2, 5)
    coord = F.mse_loss(pred_boxes[..., :4], target_boxes[..., :4], reduction="none").sum(-1, keepdim=True)
    object_loss = F.mse_loss(torch.sigmoid(pred_boxes[..., 4:5]), target_boxes[..., 4:5], reduction="none")
    no_object = (1 - target_boxes[..., 4:5]) * object_loss
    class_loss = F.mse_loss(pred_classes, target_classes, reduction="none").sum(-1, keepdim=True)
    return lambda_coord * (coord * target_boxes[..., 4:5]).mean() + object_loss.mean() + lambda_noobj * no_object.mean() + class_loss.mean()


def build_single_target(
    predictions: Sequence[Tensor],
    num_classes: int,
    image_size: int = 64,
    step: int = 0,
) -> list[Tensor]:
    """Create deterministic one-box targets for a smoke-test training loop."""
    targets: list[Tensor] = []
    for level, prediction in enumerate(predictions):
        batch, _, height, width = prediction.shape
        target = prediction.new_zeros((batch, 5 + num_classes, height, width))
        cell_y = (step + level) % height
        cell_x = (2 * step + level) % width
        batch_index = torch.arange(batch, device=prediction.device)
        target[batch_index, 0, cell_y, cell_x] = 0.5
        target[batch_index, 1, cell_y, cell_x] = 0.5
        target[batch_index, 2, cell_y, cell_x] = 0.5
        target[batch_index, 3, cell_y, cell_x] = 0.5
        target[batch_index, 4, cell_y, cell_x] = 1.0
        target[batch_index, 5 + (step + level) % num_classes, cell_y, cell_x] = 1.0
        targets.append(target)
    return targets


def build_yolo_v1_target(
    batch_size: int, grid_size: int, num_classes: int, device: torch.device, step: int = 0
) -> Tensor:
    target = torch.zeros(
        batch_size, grid_size, grid_size, num_classes + 10, device=device
    )
    y, x = step % grid_size, (2 * step) % grid_size
    target[:, y, x, num_classes : num_classes + 4] = 0.5
    target[:, y, x, num_classes + 4] = 1.0
    target[:, y, x, num_classes + 9] = 1.0
    return target


def toy_images(batch_size: int, image_size: int, device: torch.device, step: int = 0) -> Tensor:
    """Generate simple colored rectangles without requiring a dataset download."""
    images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    size = max(4, image_size // 4)
    for index in range(batch_size):
        top = (step * 3 + index * 5) % max(1, image_size - size)
        left = (step * 2 + index * 7) % max(1, image_size - size)
        channel = (step + index) % 3
        images[index, channel, top : top + size, left : left + size] = 1.0
    return images


def train_detector(
    build_model: Callable[[], nn.Module],
    *,
    steps: int = 5,
    batch_size: int = 2,
    image_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint: str | None = None,
) -> None:
    """Shared CLI loop for the version directories."""
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        images = toy_images(batch_size, image_size, torch.device(device), step)
        output = model(images)
        if isinstance(output, dict):
            predictions = output["one_to_many"]
        elif isinstance(output, Tensor):
            target = build_yolo_v1_target(
                batch_size, output.shape[1], model.num_classes, images.device, step
            )
            loss = yolo_v1_loss(output, target)
            predictions = None
        else:
            predictions = output
        if predictions is not None:
            targets = build_single_target(predictions, model.config.num_classes, image_size, step)
            loss = yolo_loss(
                predictions,
                targets,
                regression_bins=getattr(model, "regression_bins", 1),
            )
            if isinstance(output, dict) and getattr(model, "progressive_loss", False):
                auxiliary = yolo_loss(
                    output["one_to_one"],
                    targets,
                    regression_bins=getattr(model, "regression_bins", 1),
                )
                loss = loss + 0.25 * auxiliary
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"step {step + 1:03d}/{steps}: loss={loss.item():.4f}")
    if checkpoint:
        torch.save({"model": model.state_dict()}, checkpoint)
        print(f"saved checkpoint to {checkpoint}")


@torch.no_grad()
def infer_detector(
    build_model: Callable[[], nn.Module],
    *,
    checkpoint: str | None = None,
    image_size: int = 64,
    confidence: float = 0.25,
    nms_free: bool = False,
    device: str = "cpu",
) -> list[Tensor]:
    model = build_model().to(device).eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model", state))
    images = torch.rand(1, 3, image_size, image_size, device=device)
    output = model(images)
    if isinstance(output, Tensor):
        raise ValueError("YOLOv1 uses a grid tensor; use its model output directly for decoding.")
    predictions = output["one_to_one"] if nms_free and isinstance(output, dict) else output
    strides = getattr(model.config, "strides", (8, 16, 32))
    bins = getattr(model, "regression_bins", 1)
    return decode_predictions(
        predictions,
        (image_size, image_size),
        confidence,
        strides=strides,
        regression_bins=bins,
        apply_nms=not nms_free,
    )
