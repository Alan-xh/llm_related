"""Shared, compact PyTorch building blocks for the SAM teaching examples.

The implementations in this package intentionally use small dimensions and
synthetic data.  They mirror the tensor contracts and the important ideas of
the public SAM family, but they are not checkpoint-compatible reproductions of
the official models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class SAMConfig:
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 64
    encoder_depth: int = 1
    num_heads: int = 4
    num_multimask_outputs: int = 3
    max_memory_frames: int = 4

    @property
    def embedding_size(self) -> tuple[int, int]:
        return (self.image_size // self.patch_size, self.image_size // self.patch_size)


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier features used by SAM for 2-D coordinate embeddings."""

    def __init__(self, num_pos_feats: int, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "gaussian_matrix",
            scale * torch.randn(2, num_pos_feats),
            persistent=False,
        )

    def _encode(self, coords: Tensor) -> Tensor:
        projected = (2.0 * coords - 1.0) @ self.gaussian_matrix
        projected = 2.0 * math.pi * projected
        return torch.cat((projected.sin(), projected.cos()), dim=-1)

    def forward(self, size: tuple[int, int], device: Optional[torch.device] = None) -> Tensor:
        h, w = size
        device = device or self.gaussian_matrix.device
        y, x = torch.meshgrid(
            (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h,
            (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w,
            indexing="ij",
        )
        return self._encode(torch.stack((x, y), dim=-1)).permute(2, 0, 1)

    def forward_with_coords(self, coords: Tensor, image_size: tuple[int, int]) -> Tensor:
        scale = coords.new_tensor([image_size[1], image_size[0]])
        return self._encode((coords + 0.5) / scale)


class TinyImageEncoder(nn.Module):
    """Small ViT-like encoder producing a stride-``patch_size`` feature map."""

    def __init__(self, config: SAMConfig) -> None:
        super().__init__()
        if config.embed_dim % config.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.config = config
        self.patch_embed = nn.Conv2d(
            3, config.embed_dim, config.patch_size, config.patch_size
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.encoder_depth)
        self.neck = nn.Sequential(
            nn.Conv2d(config.embed_dim, config.embed_dim, 1, bias=False),
            nn.GroupNorm(1, config.embed_dim),
            nn.GELU(),
        )
        self.position_encoding = PositionEmbeddingRandom(config.embed_dim // 2)

    def forward(self, images: Tensor) -> Tensor:
        x = self.patch_embed(images)
        batch, channels, height, width = x.shape
        position = self.position_encoding((height, width), x.device)
        tokens = (x + position.unsqueeze(0)).flatten(2).transpose(1, 2)
        tokens = self.blocks(tokens)
        return self.neck(tokens.transpose(1, 2).reshape(batch, channels, height, width))


class TinyMobileImageEncoder(nn.Module):
    """Depthwise-separable encoder used to illustrate MobileSAM's student."""

    def __init__(self, config: SAMConfig) -> None:
        super().__init__()
        hidden = max(16, config.embed_dim // 2)
        self.patch_size = config.patch_size
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 2, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 2, 1, groups=hidden, bias=False),
            nn.Conv2d(hidden, config.embed_dim, 1, bias=False),
            nn.BatchNorm2d(config.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.embed_dim, config.embed_dim, 3, 2, 1, groups=config.embed_dim),
            nn.Conv2d(config.embed_dim, config.embed_dim, 1),
        )
        self.neck = nn.Sequential(nn.GroupNorm(1, config.embed_dim), nn.GELU())

    def forward(self, images: Tensor) -> Tensor:
        return self.neck(self.net(images))


class PromptEncoder(nn.Module):
    """Encodes point, box, and dense mask prompts into SAM-style embeddings."""

    def __init__(self, config: SAMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.image_size = (config.image_size, config.image_size)
        self.image_embedding_size = config.embedding_size
        self.pe_layer = PositionEmbeddingRandom(config.embed_dim // 2)
        self.point_embeddings = nn.Embedding(2, config.embed_dim)
        self.not_a_point = nn.Embedding(1, config.embed_dim)
        self.box_corner_embeddings = nn.Embedding(2, config.embed_dim)
        self.mask_downscaler = nn.Sequential(
            nn.Conv2d(1, config.embed_dim // 4, 2, 2),
            nn.GELU(),
            nn.Conv2d(config.embed_dim // 4, config.embed_dim, 1),
        )

    def _embed_points(self, coords: Tensor, labels: Tensor) -> Tensor:
        embeddings = self.pe_layer.forward_with_coords(coords, self.image_size)
        labels = labels.to(torch.long)
        valid = labels.clamp(0, 1)
        embeddings = embeddings + self.point_embeddings(valid)
        embeddings = torch.where(
            (labels < 0).unsqueeze(-1),
            self.not_a_point.weight.view(1, 1, -1),
            embeddings,
        )
        return embeddings

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        corners = torch.stack(
            (
                boxes[..., [0, 1]],
                boxes[..., [2, 3]],
            ),
            dim=-2,
        )
        embeddings = self.pe_layer.forward_with_coords(corners, self.image_size)
        embeddings = embeddings + self.box_corner_embeddings.weight.view(1, 1, 2, -1)
        return embeddings.reshape(boxes.shape[0], -1, self.embed_dim)

    def forward(
        self,
        points: Optional[tuple[Tensor, Tensor]] = None,
        boxes: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        if points is not None:
            batch_size = points[0].shape[0]
            device = points[0].device
        elif boxes is not None:
            batch_size = boxes.shape[0]
            device = boxes.device
        elif masks is not None:
            batch_size = masks.shape[0]
            device = masks.device
        else:
            batch_size = batch_size or 1
            device = self.point_embeddings.weight.device

        sparse: list[Tensor] = []
        if points is not None:
            sparse.append(self._embed_points(*points))
        if boxes is not None:
            sparse.append(self._embed_boxes(boxes).flatten(-2))
        sparse_embeddings = (
            torch.cat(sparse, dim=1)
            if sparse
            else torch.zeros(batch_size, 0, self.embed_dim, device=device)
        )
        if masks is None:
            dense_embeddings = torch.zeros(
                batch_size,
                self.embed_dim,
                *self.image_embedding_size,
                device=device,
            )
        else:
            dense_embeddings = self.mask_downscaler(masks.float())
            dense_embeddings = F.interpolate(
                dense_embeddings, size=self.image_embedding_size, mode="bilinear", align_corners=False
            )
        return sparse_embeddings, dense_embeddings


class TwoWayAttentionBlock(nn.Module):
    """One readable approximation of SAM's token/image bidirectional block."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.token_to_image = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.image_to_token = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(4))
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(
        self, tokens: Tensor, image: Tensor, token_pe: Tensor, image_pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        query = tokens + token_pe
        attended, _ = self.self_attn(query, query, tokens)
        tokens = self.norms[0](tokens + attended)
        attended, _ = self.token_to_image(
            tokens + token_pe, image + image_pe, image
        )
        tokens = self.norms[1](tokens + attended)
        tokens = self.norms[2](tokens + self.mlp(tokens))
        attended, _ = self.image_to_token(
            image + image_pe, tokens + token_pe, tokens
        )
        image = self.norms[3](image + attended)
        return tokens, image


class TwoWayTransformer(nn.Module):
    def __init__(self, config: SAMConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TwoWayAttentionBlock(config.embed_dim, config.num_heads)
            for _ in range(config.encoder_depth)
        )
        self.final_attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, batch_first=True
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self, tokens: Tensor, image: Tensor, token_pe: Tensor, image_pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            tokens, image = block(tokens, image, token_pe, image_pe)
        attended, _ = self.final_attn(tokens + token_pe, image + image_pe, image)
        return self.final_norm(tokens + attended), image


class MaskDecoder(nn.Module):
    """Dynamic-mask-token decoder with quality prediction."""

    def __init__(self, config: SAMConfig, high_quality: bool = False) -> None:
        super().__init__()
        self.config = config
        self.high_quality = high_quality
        self.num_masks = config.num_multimask_outputs
        self.num_mask_tokens = self.num_masks + 1
        self.iou_token = nn.Embedding(1, config.embed_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, config.embed_dim)
        self.transformer = TwoWayTransformer(config)
        mask_channels = max(8, config.embed_dim // 4)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(config.embed_dim, mask_channels, 2, 2),
            nn.GroupNorm(1, mask_channels),
            nn.GELU(),
            nn.ConvTranspose2d(mask_channels, mask_channels, 2, 2),
            nn.GELU(),
        )
        self.hq_projection = (
            nn.Conv2d(config.embed_dim, mask_channels, 1)
            if high_quality
            else None
        )
        hyper_hidden = max(8, config.embed_dim // 4)
        self.hypernets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.embed_dim, hyper_hidden),
                    nn.GELU(),
                    nn.Linear(hyper_hidden, mask_channels),
                )
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_head = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, self.num_mask_tokens),
        )

    def forward(
        self,
        image_embeddings: Tensor,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        image_size: tuple[int, int],
        multimask_output: bool = True,
        high_res_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = image_embeddings.shape
        output_tokens = torch.cat(
            (self.iou_token.weight, self.mask_tokens.weight), dim=0
        ).unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        image = (image_embeddings + dense_prompt_embeddings).flatten(2).transpose(1, 2)
        image_pe_tokens = image_pe.flatten(2).transpose(1, 2).expand(batch, -1, -1)
        tokens, image = self.transformer(tokens, image, tokens, image_pe_tokens)
        iou_embedding = tokens[:, 0]
        mask_embeddings = tokens[:, 1 : 1 + self.num_mask_tokens]
        image = image.transpose(1, 2).reshape(batch, channels, height, width)
        upscaled = self.output_upscaling(image)
        if self.hq_projection is not None and high_res_features is not None:
            high_res = F.interpolate(
                high_res_features, size=upscaled.shape[-2:], mode="bilinear", align_corners=False
            )
            upscaled = upscaled + self.hq_projection(high_res)
        hyper_in = torch.stack(
            [head(mask_embeddings[:, index]) for index, head in enumerate(self.hypernets)],
            dim=1,
        )
        masks = torch.einsum("bkc,bchw->bkhw", hyper_in, upscaled)
        masks = masks[:, 1:] if multimask_output else masks[:, :1]
        scores = self.iou_head(iou_embedding)
        scores = scores[:, 1:] if multimask_output else scores[:, :1]
        masks = F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)
        return masks, scores.sigmoid()


class PromptableSAM(nn.Module):
    """A compact SAM-style image model with the standard point/box interface."""

    def __init__(
        self,
        config: Optional[SAMConfig] = None,
        image_encoder: Optional[nn.Module] = None,
        high_quality: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or SAMConfig()
        self.image_encoder = image_encoder or TinyImageEncoder(self.config)
        self.prompt_encoder = PromptEncoder(self.config)
        self.mask_decoder = MaskDecoder(self.config, high_quality=high_quality)

    def encode_image(self, images: Tensor) -> Tensor:
        return self.image_encoder(images)

    def forward(
        self,
        images: Tensor,
        point_coords: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
        boxes: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        multimask_output: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if point_coords is not None and point_labels is None:
            raise ValueError("point_labels is required when point_coords is provided")
        image_embeddings = self.encode_image(images)
        sparse, dense = self.prompt_encoder(
            points=None if point_coords is None else (point_coords, point_labels),
            boxes=boxes,
            masks=masks,
            batch_size=images.shape[0],
        )
        image_pe = self.prompt_encoder.pe_layer(
            image_embeddings.shape[-2:], image_embeddings.device
        ).unsqueeze(0)
        return self.mask_decoder(
            image_embeddings,
            image_pe,
            sparse,
            dense,
            images.shape[-2:],
            multimask_output,
            high_res_features=image_embeddings,
        )


def mask_loss(
    pred_masks: Tensor,
    target_masks: Tensor,
    pred_iou: Tensor,
    bce_weight: float = 20.0,
    dice_weight: float = 1.0,
    iou_weight: float = 1.0,
) -> Tensor:
    """BCE + soft Dice + detached target IoU loss used by the demos."""
    if target_masks.shape[-2:] != pred_masks.shape[-2:]:
        target_masks = F.interpolate(
            target_masks.float(), size=pred_masks.shape[-2:], mode="nearest"
        )
    target_masks = target_masks.float()
    target_masks = target_masks.expand(-1, pred_masks.shape[1], -1, -1)
    bce = F.binary_cross_entropy_with_logits(pred_masks, target_masks)
    probabilities = pred_masks.sigmoid()
    intersection = (probabilities * target_masks).sum(dim=(-2, -1))
    union = probabilities.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) - intersection
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        probabilities.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) + 1e-6
    )
    true_iou = (intersection + 1e-6) / (union + 1e-6)
    quality = F.mse_loss(pred_iou, true_iou.detach())
    return bce_weight * bce + dice_weight * dice.mean() + iou_weight * quality


sam_loss = mask_loss


def synthetic_segmentation_batch(
    batch_size: int,
    image_size: int,
    device: torch.device,
    step: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create deterministic rectangle images, masks, and positive/negative points."""
    images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    masks = torch.zeros(batch_size, 1, image_size, image_size, device=device)
    points = torch.zeros(batch_size, 2, 2, device=device)
    labels = torch.tensor([[1, 0]], device=device).expand(batch_size, -1).clone()
    size = max(8, image_size // 3)
    for index in range(batch_size):
        top = (step * 3 + index * 5) % max(1, image_size - size)
        left = (step * 2 + index * 7) % max(1, image_size - size)
        bottom, right = top + size, left + size
        channel = (index + step) % 3
        images[index, channel, top:bottom, left:right] = 1.0
        masks[index, 0, top:bottom, left:right] = 1.0
        points[index, 0] = torch.tensor(
            [left + size / 2, top + size / 2], device=device
        )
        points[index, 1] = torch.tensor(
            [max(0, left - 2), max(0, top - 2)], device=device
        )
    return images, points, labels, masks


def train_segmenter(
    build_model,
    *,
    steps: int = 5,
    batch_size: int = 2,
    image_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint: str = "sam_tiny.pt",
) -> None:
    target_device = torch.device(device)
    model = build_model(image_size=image_size).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        images, points, labels, target = synthetic_segmentation_batch(
            batch_size, image_size, target_device, step
        )
        masks, scores = model(images, points, labels)
        loss = mask_loss(masks, target, scores)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"step {step + 1:03d}/{steps}: loss={loss.item():.4f}")
    torch.save({"model": model.state_dict()}, checkpoint)
    print(f"saved checkpoint to {checkpoint}")


@dataclass
class VideoMemory:
    """Bounded streaming state for SAM 2 style frame propagation."""

    max_frames: int = 4
    keys: list[Tensor] = field(default_factory=list)
    masks: list[Tensor] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

    def append(self, key: Tensor, mask: Tensor, frame_index: int) -> None:
        self.keys.append(key.detach())
        self.masks.append(mask.detach())
        self.frame_indices.append(frame_index)
        if len(self.keys) > self.max_frames:
            self.keys.pop(0)
            self.masks.pop(0)
            self.frame_indices.pop(0)

    def stacked_keys(self) -> Optional[Tensor]:
        return torch.cat(self.keys, dim=1) if self.keys else None


class SAM2Model(PromptableSAM):
    """PromptableSAM plus bounded memory attention for video frames."""

    def __init__(self, config: Optional[SAMConfig] = None, improved: bool = False) -> None:
        config = config or SAMConfig()
        super().__init__(config)
        self.improved = improved
        self.memory_attention = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(config.embed_dim)

    def attend_memory(self, image_embeddings: Tensor, memory: Optional[VideoMemory]) -> Tensor:
        if memory is None or memory.stacked_keys() is None:
            return image_embeddings
        batch, channels, height, width = image_embeddings.shape
        query = image_embeddings.flatten(2).transpose(1, 2)
        keys = memory.stacked_keys().to(query.device)
        if keys.shape[0] == 1 and batch > 1:
            keys = keys.expand(batch, -1, -1)
        attended, _ = self.memory_attention(query, keys, keys)
        updated = self.memory_norm(query + attended)
        return updated.transpose(1, 2).reshape(batch, channels, height, width)

    def predict_frame(
        self,
        images: Tensor,
        point_coords: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
        memory: Optional[VideoMemory] = None,
        frame_index: int = 0,
    ) -> tuple[Tensor, Tensor, VideoMemory]:
        image_embeddings = self.attend_memory(self.encode_image(images), memory)
        sparse, dense = self.prompt_encoder(
            None if point_coords is None else (point_coords, point_labels),
            batch_size=images.shape[0],
        )
        image_pe = self.prompt_encoder.pe_layer(
            image_embeddings.shape[-2:], image_embeddings.device
        ).unsqueeze(0)
        masks, scores = self.mask_decoder(
            image_embeddings, image_pe, sparse, dense, images.shape[-2:]
        )
        state = memory or VideoMemory(self.config.max_memory_frames)
        state.append(
            image_embeddings.flatten(2).transpose(1, 2).mean(dim=1, keepdim=True),
            masks[:, :1].flatten(2),
            frame_index,
        )
        return masks, scores, state

    def forward(self, images: Tensor, point_coords=None, point_labels=None, **kwargs):
        masks, scores, _ = self.predict_frame(
            images, point_coords, point_labels, kwargs.get("memory")
        )
        return masks, scores


class TextPromptEncoder(nn.Module):
    """Dependency-free text prompt encoder for the SAM 3 teaching model."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, texts: Sequence[str], device: torch.device, batch_size: int
    ) -> Tensor:
        if len(texts) == 1 and batch_size > 1:
            texts = list(texts) * batch_size
        if len(texts) != batch_size:
            raise ValueError("one text prompt is required per batch element")
        rows = []
        for text in texts:
            token_ids = torch.tensor(list(text.encode("utf-8")) or [0], device=device)
            rows.append(self.embedding(token_ids).mean(dim=0))
        return self.norm(torch.stack(rows, dim=0)).unsqueeze(1)


class SAM3Model(PromptableSAM):
    """Concept segmentation with text, exemplar, and visual prompts."""

    def __init__(self, config: Optional[SAMConfig] = None) -> None:
        super().__init__(config)
        self.text_encoder = TextPromptEncoder(self.config.embed_dim)
        self.presence_head = nn.Linear(self.config.embed_dim, 1)
        self.concept_projection = nn.Linear(self.config.embed_dim, self.config.embed_dim)

    def forward(
        self,
        images: Tensor,
        text_prompts: Optional[Sequence[str]] = None,
        exemplar_boxes: Optional[Tensor] = None,
        point_coords: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        image_embeddings = self.encode_image(images)
        sparse, dense = self.prompt_encoder(
            None if point_coords is None else (point_coords, point_labels),
            boxes=exemplar_boxes,
            batch_size=images.shape[0],
        )
        if text_prompts is not None:
            text_tokens = self.text_encoder(text_prompts, images.device, images.shape[0])
            sparse = torch.cat((self.concept_projection(text_tokens), sparse), dim=1)
        image_pe = self.prompt_encoder.pe_layer(
            image_embeddings.shape[-2:], image_embeddings.device
        ).unsqueeze(0)
        masks, scores = self.mask_decoder(
            image_embeddings, image_pe, sparse, dense, images.shape[-2:]
        )
        pooled = image_embeddings.mean(dim=(-2, -1))
        presence = self.presence_head(pooled)
        return {
            "masks": masks,
            "iou_scores": scores,
            "presence_logits": presence,
            "image_embeddings": image_embeddings,
        }


class ObjectMultiplex:
    """Pack a set of object tokens and decode them against one image feature map."""

    def __init__(self, bucket_size: int = 8) -> None:
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        self.bucket_size = bucket_size

    def pack(self, tokens: Sequence[Tensor]) -> tuple[Tensor, list[int]]:
        if not tokens:
            raise ValueError("at least one object token is required")
        lengths = [token.shape[1] for token in tokens]
        packed = torch.cat(tokens, dim=1)
        padded_length = self.bucket_size * math.ceil(packed.shape[1] / self.bucket_size)
        if padded_length > packed.shape[1]:
            packed = torch.cat(
                (
                    packed,
                    packed.new_zeros(packed.shape[0], padded_length - packed.shape[1], packed.shape[2]),
                ),
                dim=1,
            )
        return packed, lengths

    @staticmethod
    def unpack(tokens: Tensor, lengths: Sequence[int]) -> list[Tensor]:
        return list(tokens.split(list(lengths), dim=1))


class SAM31Model(SAM3Model):
    """SAM 3 concept model with a shared image pass for many object prompts."""

    def multiplex(
        self,
        images: Tensor,
        text_prompts: Sequence[str],
        bucket_size: int = 8,
    ) -> dict[str, Tensor]:
        if images.shape[0] != 1:
            raise ValueError("the teaching multiplex path currently expects batch size 1")
        image_embeddings = self.encode_image(images)
        tokens = [
            self.text_encoder([prompt], images.device, 1)
            for prompt in text_prompts
        ]
        packer = ObjectMultiplex(bucket_size)
        _, lengths = packer.pack(tokens)
        object_count = len(tokens)
        expanded_image = image_embeddings.expand(object_count, -1, -1, -1)
        sparse = self.concept_projection(torch.cat(tokens, dim=0))
        dense = torch.zeros_like(expanded_image)
        image_pe = self.prompt_encoder.pe_layer(
            image_embeddings.shape[-2:], image_embeddings.device
        ).unsqueeze(0).expand(object_count, -1, -1, -1)
        masks, scores = self.mask_decoder(
            expanded_image,
            image_pe,
            sparse,
            dense,
            images.shape[-2:],
        )
        presence = self.presence_head(image_embeddings.mean(dim=(-2, -1))).expand(object_count, -1)
        return {
            "masks": masks.view(1, object_count, masks.shape[1], *masks.shape[-2:]),
            "iou_scores": scores.view(1, object_count, -1),
            "presence_logits": presence.view(1, object_count, -1),
            "object_token_lengths": torch.tensor(lengths, device=images.device),
        }


def load_checkpoint(model: nn.Module, path: Optional[str], device: torch.device) -> None:
    if not path:
        return
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))
