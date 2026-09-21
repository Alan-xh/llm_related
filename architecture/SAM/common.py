"""Shared building blocks for the compact Segment Anything teaching models.

Task:
    Promptable image/video segmentation. Inputs are images ``[B, 3, H, W]``
    plus optional point, box, mask, text, or memory prompts. The standard
    output is mask logits ``[B, K, H, W]`` and quality scores ``[B, K]``.

Representative architectures:
    SAM, MobileSAM, FastSAM, SAM-HQ, SAM 2, SAM 2.1, SAM 3, and SAM 3.1.
    These are deliberately small, readable approximations rather than
    checkpoint-compatible reproductions of the official implementations.

Core formulas:
    PE(x) = [sin(2*pi*B*x), cos(2*pi*B*x)]
    M_k(h,w) = hypernet_k(mask_token_k) dot upscaled_feature(h,w)
    L = 20 * BCE(mask, target) + Dice(mask, target)
        + MSE(predicted_iou, stop_gradient(true_iou))
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
    """Shared hyperparameters and spatial contracts for SAM teaching models."""

    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 64
    encoder_depth: int = 1
    num_heads: int = 4
    num_multimask_outputs: int = 3
    max_memory_frames: int = 4

    @property
    def embedding_size(self) -> tuple[int, int]:
        """Return the image embedding size ``(H/patch_size, W/patch_size)``."""
        return (self.image_size // self.patch_size, self.image_size // self.patch_size)


class PositionEmbeddingRandom(nn.Module):
    """Encode 2-D coordinates with random Fourier features.

    For ``coords[..., 2]``, the registered matrix ``B`` has shape
    ``[2, C/2]`` and the result has shape ``[..., C]``:
    ``[sin(2*pi*coords*B), cos(2*pi*coords*B)]``.
    """

    def __init__(self, num_pos_feats: int, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "gaussian_matrix",
            scale * torch.randn(2, num_pos_feats),
            persistent=False,
        )

    def _encode(self, coords: Tensor) -> Tensor:
        """Map normalized coordinates ``[..., 2]`` to Fourier features ``[..., C]``."""
        projected = (2.0 * coords - 1.0) @ self.gaussian_matrix
        projected = 2.0 * math.pi * projected
        # [.., C/2] + [.., C/2] -> [.., C].
        return torch.cat((projected.sin(), projected.cos()), dim=-1)

    def forward(self, size: tuple[int, int], device: Optional[torch.device] = None) -> Tensor:
        """Create a dense positional map with shape ``[C, H, W]``."""
        h, w = size
        device = device or self.gaussian_matrix.device
        y, x = torch.meshgrid(
            (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h,
            (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w,
            indexing="ij",
        )
        coords = torch.stack((x, y), dim=-1)  # [H, W, 2].
        return self._encode(coords).permute(2, 0, 1)  # [H, W, C] -> [C, H, W].

    def forward_with_coords(self, coords: Tensor, image_size: tuple[int, int]) -> Tensor:
        """Encode pixel coordinates ``[..., 2]`` as embeddings ``[..., C]``."""
        scale = coords.new_tensor([image_size[1], image_size[0]])
        return self._encode((coords + 0.5) / scale)


class TinyImageEncoder(nn.Module):
    """Small ViT-like encoder producing ``[B, C, H/P, W/P]`` features."""

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
        """Encode images ``[B, 3, H, W]`` into ``[B, C, H/P, W/P]``."""
        x = self.patch_embed(images)  # [B, 3, H, W] -> [B, C, H/P, W/P].
        batch, channels, height, width = x.shape
        position = self.position_encoding((height, width), x.device)  # [C, h, w].
        tokens = (x + position.unsqueeze(0)).flatten(2).transpose(1, 2)
        # [B, C, h, w] -> [B, h*w, C].
        tokens = self.blocks(tokens)
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        # [B, h*w, C] -> [B, C, h, w].
        return self.neck(features)


class TinyMobileImageEncoder(nn.Module):
    """Depthwise-separable student encoder with output ``[B, C, H/8, W/8]``."""

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
        """Encode images ``[B, 3, H, W]`` into lightweight features ``[B, C, H/8, W/8]``."""
        features = self.net(images)  # [B, 3, H, W] -> [B, C, H/8, W/8].
        return self.neck(features)


class PromptEncoder(nn.Module):
    """Encode point, box, and dense mask prompts into sparse/dense embeddings.

    Outputs:
        sparse embeddings ``[B, N_prompt, C]``;
        dense embeddings ``[B, C, H/P, W/P]``.
    """

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
        """Encode points ``[B, N, 2]`` and labels ``[B, N]`` as ``[B, N, C]``."""
        embeddings = self.pe_layer.forward_with_coords(coords, self.image_size)
        labels = labels.to(torch.long)
        valid = labels.clamp(0, 1)
        embeddings = embeddings + self.point_embeddings(valid)
        # Label -1 marks padding; replace its coordinate embedding with a learned token.
        embeddings = torch.where(
            (labels < 0).unsqueeze(-1),
            self.not_a_point.weight.view(1, 1, -1),
            embeddings,
        )
        return embeddings

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        """Encode boxes ``[B, M, 4]`` as two corner tokens ``[B, 2*M, C]``."""
        corners = torch.stack(
            (
                boxes[..., [0, 1]],
                boxes[..., [2, 3]],
            ),
            dim=-2,
        )
        embeddings = self.pe_layer.forward_with_coords(corners, self.image_size)
        embeddings = embeddings + self.box_corner_embeddings.weight.view(1, 1, 2, -1)
        # [B, M, 2, C] -> [B, 2*M, C].
        return embeddings.reshape(boxes.shape[0], -1, self.embed_dim)

    def forward(
        self,
        points: Optional[tuple[Tensor, Tensor]] = None,
        boxes: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return sparse ``[B, N_prompt, C]`` and dense ``[B, C, h, w]`` prompts.

        Point inputs are ``([B, N, 2], [B, N])`` in pixel ``(x, y)`` format;
        boxes are ``[B, M, 4]`` in ``xyxy`` format; masks are ``[B, 1, H, W]``.
        Missing prompts produce zero embeddings with the same batch contract.
        """
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
        # Concatenation changes [B, N_i, C] -> [B, sum(N_i), C].
        if masks is None:
            dense_embeddings = torch.zeros(
                batch_size,
                self.embed_dim,
                *self.image_embedding_size,
                device=device,
            )
        else:
            dense_embeddings = self.mask_downscaler(masks.float())
            # [B, 1, H, W] -> [B, C, H/4, W/4], then resize to [B, C, h, w].
            dense_embeddings = F.interpolate(
                dense_embeddings, size=self.image_embedding_size, mode="bilinear", align_corners=False
            )
        return sparse_embeddings, dense_embeddings


class TwoWayAttentionBlock(nn.Module):
    """Update prompt tokens and image tokens with bidirectional attention.

    Tokens have shape ``[B, N, C]`` and image tokens have shape ``[B, HW, C]``.
    The block applies token self-attention, token-to-image attention, an MLP,
    and image-to-token attention.
    """

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
        """Return updated tokens ``[B, N, C]`` and image tokens ``[B, HW, C]``."""
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
    """Stack bidirectional attention blocks over prompt and image tokens."""

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
        """Transform tokens ``[B, N, C]`` and image tokens ``[B, HW, C]``."""
        for block in self.blocks:
            tokens, image = block(tokens, image, token_pe, image_pe)
        attended, _ = self.final_attn(tokens + token_pe, image + image_pe, image)
        return self.final_norm(tokens + attended), image


class MaskDecoder(nn.Module):
    """Decode prompt/image tokens into dynamic masks and IoU-quality scores.

    The hypernetwork maps each mask token to a channel vector. The einsum then
    implements ``M_k(h,w) = hyper_k dot U(h,w)``.
    """

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
        """Return mask logits ``[B, K, H, W]`` and scores ``[B, K]``.

        ``image_embeddings`` and ``dense_prompt_embeddings`` are ``[B, C, h, w]``;
        ``sparse_prompt_embeddings`` is ``[B, N_prompt, C]``; ``image_pe`` is
        ``[1, C, h, w]`` or a broadcastable equivalent.
        """
        batch, channels, height, width = image_embeddings.shape
        output_tokens = torch.cat(
            (self.iou_token.weight, self.mask_tokens.weight), dim=0
        ).unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # [B, 1+K, C] + [B, N_prompt, C] -> [B, 1+K+N_prompt, C].
        image = (image_embeddings + dense_prompt_embeddings).flatten(2).transpose(1, 2)
        # [B, C, h, w] -> [B, h*w, C].
        image_pe_tokens = image_pe.flatten(2).transpose(1, 2).expand(batch, -1, -1)
        tokens, image = self.transformer(tokens, image, tokens, image_pe_tokens)
        iou_embedding = tokens[:, 0]
        mask_embeddings = tokens[:, 1 : 1 + self.num_mask_tokens]
        image = image.transpose(1, 2).reshape(batch, channels, height, width)
        # [B, h*w, C] -> [B, C, h, w] -> [B, C_mask, 4h, 4w].
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
        # [B, K, C_mask] x [B, C_mask, H', W'] -> [B, K, H', W'].
        masks = torch.einsum("bkc,bchw->bkhw", hyper_in, upscaled)
        masks = masks[:, 1:] if multimask_output else masks[:, :1]
        scores = self.iou_head(iou_embedding)
        scores = scores[:, 1:] if multimask_output else scores[:, :1]
        masks = F.interpolate(masks, size=image_size, mode="bilinear", align_corners=False)
        return masks, scores.sigmoid()


class PromptableSAM(nn.Module):
    """Compact SAM model with point, box, and mask prompt interfaces.

    Image inputs use ``[B, 3, H, W]``. The forward output is
    ``(masks=[B, K, H, W], scores=[B, K])``.
    """

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
        """Encode images ``[B, 3, H, W]`` as ``[B, C, H/P, W/P]``."""
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
        """Run image/prompt encoding and dynamic mask decoding.

        ``point_coords`` is ``[B, N, 2]`` and ``point_labels`` is ``[B, N]``;
        ``boxes`` is ``[B, M, 4]``; ``masks`` is ``[B, 1, H, W]``.
        """
        if point_coords is not None and point_labels is None:
            raise ValueError("point_labels is required when point_coords is provided")
        image_embeddings = self.encode_image(images)
        # [B, 3, H, W] -> [B, C, H/P, W/P].
        sparse, dense = self.prompt_encoder(
            points=None if point_coords is None else (point_coords, point_labels),
            boxes=boxes,
            masks=masks,
            batch_size=images.shape[0],
        )
        image_pe = self.prompt_encoder.pe_layer(
            image_embeddings.shape[-2:], image_embeddings.device
        ).unsqueeze(0)
        # [C, h, w] -> [1, C, h, w], broadcast over the batch.
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
    """Compute ``20*BCE + Dice + IoU-MSE`` for masks and quality scores.

    Args:
        pred_masks: mask logits ``[B, K, H, W]``.
        target_masks: binary masks ``[B, 1, H, W]`` or ``[B, K, H, W]``.
        pred_iou: predicted quality scores ``[B, K]``.

    The target IoU is computed from sigmoid probabilities and detached before
    the MSE term, so the quality head does not backpropagate through masks.
    """
    if target_masks.shape[-2:] != pred_masks.shape[-2:]:
        target_masks = F.interpolate(
            target_masks.float(), size=pred_masks.shape[-2:], mode="nearest"
        )
    target_masks = target_masks.float()
    target_masks = target_masks.expand(-1, pred_masks.shape[1], -1, -1)
    bce = F.binary_cross_entropy_with_logits(pred_masks, target_masks)
    probabilities = pred_masks.sigmoid()  # [B, K, H, W].
    intersection = (probabilities * target_masks).sum(dim=(-2, -1))
    union = probabilities.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) - intersection
    # Dice = 1 - 2*|P intersection T| / (|P| + |T|).
    dice = 1.0 - (2.0 * intersection + 1e-6) / (
        probabilities.sum(dim=(-2, -1)) + target_masks.sum(dim=(-2, -1)) + 1e-6
    )
    true_iou = (intersection + 1e-6) / (union + 1e-6)
    quality = F.mse_loss(pred_iou, true_iou.detach())  # both are [B, K].
    return bce_weight * bce + dice_weight * dice.mean() + iou_weight * quality


sam_loss = mask_loss


def synthetic_segmentation_batch(
    batch_size: int,
    image_size: int,
    device: torch.device,
    step: int = 0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create a synthetic segmentation batch with explicit tensor contracts.

    Returns ``images [B,3,H,W]``, ``points [B,2,2]`` in ``xy`` pixels,
    ``labels [B,2]`` with positive/negative values, and ``targets [B,1,H,W]``.
    """
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
        images[index, channel, top:bottom, left:right] = 1.0  # rectangle image.
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
    """Run a small decoupled training pipeline and save model weights.

    The model receives images ``[B,3,H,W]`` and point prompts
    ``(points [B,2,2], labels [B,2])``; it returns masks ``[B,K,H,W]`` and
    scores ``[B,K]`` consumed by :func:`mask_loss`.
    """
    target_device = torch.device(device)
    model = build_model(image_size=image_size).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        images, points, labels, target = synthetic_segmentation_batch(
            batch_size, image_size, target_device, step
        )
        masks, scores = model(images, points, labels)
        # masks: [B,K,H,W], scores: [B,K], target: [B,1,H,W].
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
    """Bounded streaming state for SAM 2 style frame propagation.

    ``keys`` stores entries shaped ``[B, T_i, C]`` and ``masks`` stores the
    corresponding flattened mask features. Oldest entries are discarded when
    ``max_frames`` is exceeded.
    """

    max_frames: int = 4
    keys: list[Tensor] = field(default_factory=list)
    masks: list[Tensor] = field(default_factory=list)
    frame_indices: list[int] = field(default_factory=list)

    def append(self, key: Tensor, mask: Tensor, frame_index: int) -> None:
        """Append one detached frame entry and enforce the memory limit."""
        self.keys.append(key.detach())
        self.masks.append(mask.detach())
        self.frame_indices.append(frame_index)
        if len(self.keys) > self.max_frames:
            self.keys.pop(0)
            self.masks.pop(0)
            self.frame_indices.pop(0)

    def stacked_keys(self) -> Optional[Tensor]:
        """Concatenate keys along time: ``[B,T_i,C] -> [B,sum(T_i),C]``."""
        return torch.cat(self.keys, dim=1) if self.keys else None


class SAM2Model(PromptableSAM):
    """PromptableSAM plus bounded memory attention for video frames.

    Current image tokens query concatenated memory keys using
    ``feature' = LayerNorm(feature + Attention(feature, memory, memory))``.
    """

    def __init__(self, config: Optional[SAMConfig] = None, improved: bool = False) -> None:
        config = config or SAMConfig()
        super().__init__(config)
        self.improved = improved
        self.memory_attention = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(config.embed_dim)

    def attend_memory(self, image_embeddings: Tensor, memory: Optional[VideoMemory]) -> Tensor:
        """Fuse ``[B,C,h,w]`` features with memory while preserving shape."""
        if memory is None or memory.stacked_keys() is None:
            return image_embeddings
        batch, channels, height, width = image_embeddings.shape
        query = image_embeddings.flatten(2).transpose(1, 2)
        # [B, C, h, w] -> [B, h*w, C].
        keys = memory.stacked_keys().to(query.device)
        if keys.shape[0] == 1 and batch > 1:
            keys = keys.expand(batch, -1, -1)
        attended, _ = self.memory_attention(query, keys, keys)
        updated = self.memory_norm(query + attended)
        # [B, h*w, C] -> [B, C, h, w].
        return updated.transpose(1, 2).reshape(batch, channels, height, width)

    def predict_frame(
        self,
        images: Tensor,
        point_coords: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
        memory: Optional[VideoMemory] = None,
        frame_index: int = 0,
    ) -> tuple[Tensor, Tensor, VideoMemory]:
        """Predict one frame and return masks ``[B,K,H,W]``, scores ``[B,K]``,
        and the updated bounded memory state.
        """
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
        # Stored key is [B,1,C]; stored mask is [B,1,H*W].
        return masks, scores, state

    def forward(self, images: Tensor, point_coords=None, point_labels=None, **kwargs):
        """Compatibility forward returning masks and quality scores."""
        masks, scores, _ = self.predict_frame(
            images, point_coords, point_labels, kwargs.get("memory")
        )
        return masks, scores


class TextPromptEncoder(nn.Module):
    """Dependency-free byte-level text encoder producing ``[B, 1, C]`` tokens."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, texts: Sequence[str], device: torch.device, batch_size: int
    ) -> Tensor:
        """Encode one UTF-8 text prompt per batch item as ``[B, 1, C]``."""
        if len(texts) == 1 and batch_size > 1:
            texts = list(texts) * batch_size
        if len(texts) != batch_size:
            raise ValueError("one text prompt is required per batch element")
        rows = []
        for text in texts:
            token_ids = torch.tensor(list(text.encode("utf-8")) or [0], device=device)
            rows.append(self.embedding(token_ids).mean(dim=0))
        return self.norm(torch.stack(rows, dim=0)).unsqueeze(1)  # [B,C] -> [B,1,C].


class SAM3Model(PromptableSAM):
    """Concept segmentation with text, exemplar-box, and visual prompts.

    In addition to masks and IoU scores, the model predicts a concept-presence
    logit from globally pooled image features.
    """

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
        """Return masks ``[B,K,H,W]``, scores ``[B,K]``, presence ``[B,1]``,
        and image embeddings ``[B,C,h,w]``.
        """
        image_embeddings = self.encode_image(images)
        sparse, dense = self.prompt_encoder(
            None if point_coords is None else (point_coords, point_labels),
            boxes=exemplar_boxes,
            batch_size=images.shape[0],
        )
        if text_prompts is not None:
            text_tokens = self.text_encoder(text_prompts, images.device, images.shape[0])
            sparse = torch.cat((self.concept_projection(text_tokens), sparse), dim=1)
            # [B,1,C] + [B,N,C] -> [B,N+1,C].
        image_pe = self.prompt_encoder.pe_layer(
            image_embeddings.shape[-2:], image_embeddings.device
        ).unsqueeze(0)
        masks, scores = self.mask_decoder(
            image_embeddings, image_pe, sparse, dense, images.shape[-2:]
        )
        pooled = image_embeddings.mean(dim=(-2, -1))  # [B,C,h,w] -> [B,C].
        presence = self.presence_head(pooled)
        return {
            "masks": masks,
            "iou_scores": scores,
            "presence_logits": presence,
            "image_embeddings": image_embeddings,
        }


class ObjectMultiplex:
    """Pack object tokens into buckets for one shared image-encoder pass."""

    def __init__(self, bucket_size: int = 8) -> None:
        if bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        self.bucket_size = bucket_size

    def pack(self, tokens: Sequence[Tensor]) -> tuple[Tensor, list[int]]:
        """Concatenate ``[B,N_i,C]`` tokens and return packed ``[B,N,C]``."""
        if not tokens:
            raise ValueError("at least one object token is required")
        lengths = [token.shape[1] for token in tokens]
        packed = torch.cat(tokens, dim=1)
        # [B,N_1,C] + ... -> [B,sum(N_i),C].
        padded_length = self.bucket_size * math.ceil(packed.shape[1] / self.bucket_size)
        if padded_length > packed.shape[1]:
            packed = torch.cat(
                (
                    packed,
                    packed.new_zeros(packed.shape[0], padded_length - packed.shape[1], packed.shape[2]),
                ),
                dim=1,
            )
            # Pad the token axis to a multiple of bucket_size.
        return packed, lengths

    @staticmethod
    def unpack(tokens: Tensor, lengths: Sequence[int]) -> list[Tensor]:
        """Split packed ``[B,N,C]`` tokens back into the original groups."""
        return list(tokens.split(list(lengths), dim=1))


class SAM31Model(SAM3Model):
    """SAM 3 concept model with shared image features for many object prompts."""

    def multiplex(
        self,
        images: Tensor,
        text_prompts: Sequence[str],
        bucket_size: int = 8,
    ) -> dict[str, Tensor]:
        """Decode ``O`` text prompts and return masks ``[1,O,K,H,W]``.

        The image is encoded once as ``[1,C,h,w]`` and expanded to
        ``[O,C,h,w]`` only for the prompt-conditioned decoder batch.
        """
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
        # [1,C,h,w] -> [O,C,h,w] for parallel decoder evaluation.
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
            # [O,K,H,W] -> [1,O,K,H,W], restoring the object axis.
            "masks": masks.view(1, object_count, masks.shape[1], *masks.shape[-2:]),
            "iou_scores": scores.view(1, object_count, -1),  # [O,K] -> [1,O,K].
            "presence_logits": presence.view(1, object_count, -1),  # [O,1] -> [1,O,1].
            "object_token_lengths": torch.tensor(lengths, device=images.device),
        }


def load_checkpoint(model: nn.Module, path: Optional[str], device: torch.device) -> None:
    """Load either a raw state dict or a ``{"model": state_dict}`` checkpoint."""
    if not path:
        return
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))
