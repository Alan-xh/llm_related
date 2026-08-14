"""
12_MultiModal.py
================
vLLM 多模态(VLM)推理:Vision Tower + Projector + LLM,把图像 patch 转成 token 输入。

架构(LLaVA 风格):
    image -> Vision Tower (CLIP/SigLIP) -> image features [N_patches, D]
         -> Projector (MLP/Q-Former) -> image embeddings [N_patches, hidden]
         -> 替换 prompt 中的 <image> placeholder -> LLM forward

关键挑战:
    1. Image token 数量动态(不同分辨率 patch 数不同)
    2. 多图像输入
    3. Patch embedding 与 text token 在同一序列中

vLLM 实现:
    - MultiModalRegistry:注册不同模态的 processor
    - MultiModalInputs:统一数据结构
    - 在 prefill 时把 image features 插入对应位置
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Vision Tower(模拟 CLIP/SigLIP)
# ============================================================

class VisionTower(nn.Module):
    """
    模拟 ViT-based vision encoder。
    输入:image [B, 3, H, W]
    输出:patch features [B, N_patches, vision_dim]
    """

    def __init__(self, vision_dim: int = 768, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        # patch embedding (conv2d 把图像切成 patch)
        self.patch_embed = nn.Conv2d(3, vision_dim,
                                     kernel_size=patch_size,
                                     stride=patch_size)
        # 简化的 transformer(实际有 12-32 层)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=vision_dim, nhead=12,
                                       dim_feedforward=vision_dim*4,
                                       batch_first=True),
            num_layers=4
        )
        self.ln_post = nn.LayerNorm(vision_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        return: [B, N_patches, vision_dim]
        """
        B, C, H, W = images.shape
        patches = self.patch_embed(images)  # [B, vision_dim, H/p, W/p]
        N = patches.shape[2] * patches.shape[3]
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, vision_dim]

        # 加 cls token(简化:省略)
        features = self.transformer(patches)
        return self.ln_post(features)


# ============================================================
# 2. Projector(LLaVA 风格 MLP)
# ============================================================

class MLPProjector(nn.Module):
    """把 vision_dim 映射到 LLM hidden_size"""

    def __init__(self, vision_dim: int = 768, hidden_size: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


# ============================================================
# 3. LLM(简化 transformer)
# ============================================================

class SimpleLLM(nn.Module):
    """简化 LLM:embedding + 几层 transformer + LM head"""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 1024,
                 num_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8,
                                       dim_feedforward=hidden_size*4,
                                       batch_first=True),
            num_layers=num_layers
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """接受 embedding 而非 token id(支持混合 image/text embedding)"""
        h = self.transformer(input_embeddings)
        return self.head(self.ln(h))


# ============================================================
# 4. MultiModal Processor:拼接 image features 和 text embedding
# ============================================================

IMAGE_TOKEN_ID = -100  # prompt 中 image 占位符的标记


class MultiModalProcessor:
    """
    处理混合输入:把 prompt 中的 <image> 占位替换为 image features。
    """

    def __init__(self, llm: SimpleLLM, projector: MLPProjector):
        self.llm = llm
        self.projector = projector

    def build_input_embeddings(self,
                               prompt_token_ids: List[int],
                               image_features: Optional[torch.Tensor] = None
                               ) -> torch.Tensor:
        """
        prompt_token_ids: List[int],其中 IMAGE_TOKEN_ID 标记 image 位置
        image_features: [N_patches, vision_dim]
        return: [1, seq_len, hidden_size]
        """
        if image_features is not None:
            # 投影 image features
            image_emb = self.projector(image_features)  # [N_patches, hidden]
            n_image_tokens = image_emb.shape[0]
        else:
            n_image_tokens = 0

        # 找到 image placeholder 位置
        text_ids = [t for t in prompt_token_ids if t != IMAGE_TOKEN_ID]
        text_emb = self.llm.embed(torch.tensor(text_ids))  # [N_text, hidden]

        if image_features is None:
            return text_emb.unsqueeze(0)

        # 拼接:text_emb 前 + image_emb + text_emb 后
        placeholder_idx = prompt_token_ids.index(IMAGE_TOKEN_ID)
        before = self.llm.embed(torch.tensor(prompt_token_ids[:placeholder_idx]))
        after = self.llm.embed(torch.tensor(prompt_token_ids[placeholder_idx+1:]))

        full_emb = torch.cat([before, image_emb, after], dim=0)  # [seq_len, hidden]
        return full_emb.unsqueeze(0)


# ============================================================
# 5. 完整 VLM 推理
# ============================================================

class LLaVA(nn.Module):
    """LLaVA 风格 VLM"""

    def __init__(self, vision_dim=768, hidden_size=1024, vocab_size=32000):
        super().__init__()
        self.vision_tower = VisionTower(vision_dim=vision_dim)
        self.projector = MLPProjector(vision_dim, hidden_size)
        self.llm = SimpleLLM(vocab_size=vocab_size, hidden_size=hidden_size)
        self.processor = MultiModalProcessor(self.llm, self.projector)

    def forward(self, images: Optional[torch.Tensor],
                prompt_token_ids: List[int]) -> torch.Tensor:
        # 1. 提取 image features
        if images is not None:
            img_feats = self.vision_tower(images)  # [B, N, D]
            # 假设 batch=1
            img_feats = img_feats[0]
        else:
            img_feats = None

        # 2. 构建混合 embedding
        emb = self.processor.build_input_embeddings(prompt_token_ids, img_feats)

        # 3. LLM forward
        logits = self.llm(emb)
        return logits


# ============================================================
# 6. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    model = LLaVA(vision_dim=128, hidden_size=128, vocab_size=1000)
    model.eval()

    # 模拟图像 [1, 3, 32, 32] -> 4 个 patch
    image = torch.randn(1, 3, 32, 32)

    # prompt: "Describe <image> in detail"
    # 用 -100 标记 image 位置
    prompt = [10, 20, 30, IMAGE_TOKEN_ID, 40, 50]

    with torch.no_grad():
        logits = model(image, prompt)

    print(f"Image shape: {image.shape}")
    print(f"Prompt tokens: {prompt}")
    print(f"  (其中 -100 是 image placeholder,会被替换为 image features)")
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"  seq_len = before(3) + image_patches(4) + after(2) = 9")
    print(f"  expected seq_len: {3 + 4 + 2}")

    # 多图像场景(简化)
    print("\n--- Multi-image (省略 LLM,只演示 vision tower) ---")
    multi_images = torch.randn(2, 3, 32, 32)
    feats = model.vision_tower(multi_images)
    print(f"2 images -> features: {feats.shape}  (B=2, N_patches=4, D=128)")


if __name__ == "__main__":
    demo()
