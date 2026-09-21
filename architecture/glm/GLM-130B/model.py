"""GLM-130B 教学版 blank-infilling 模型定义。

任务定义:
    任务编号: GLM-130B；领域: 自回归空白填充语言建模。输入 token ids
    shape 为 [B, T]，二维位置 ids shape 为 [B, 2, T]，输出 logits shape
    为 [B, T, V]。

核心机制:
    样本布局为 ``[BOS] + prefix + <BLANK> + suffix + target``。prefix 和
    suffix 通过 prefix-visible attention 形成 context，loss 仅监督 target；
    第 0 行 position ids 进入 RoPE，第 1 行进入 block-position embedding。

数学映射:
    ``M(q,k)=causal(q,k) OR (q<prefix AND k<prefix)``；
    ``L_lm=CrossEntropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)``。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from architecture.glm.common import GLMConfig, GLMCausalLM
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.glm.common import GLMConfig, GLMCausalLM


@dataclass
class GLM130BConfig(GLMConfig):
    """GLM-130B 风格的小型配置，启用二维位置和 blank-infilling 接口。"""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    use_2d_position_ids: bool = True
    max_block_position_embeddings: int = 128
    rope_theta: float = 10_000.0


class GLM130BForCausalLM(GLMCausalLM):
    """可运行的 GLM 教学 decoder，不是官方 130B checkpoint 兼容复刻。"""


def build_model() -> GLM130BForCausalLM:
    """按默认 GLM-130B 教学配置构造模型。"""
    return GLM130BForCausalLM(GLM130BConfig())


if __name__ == "__main__":
    model = build_model()
    # 参数量用于确认当前是缩小后的可运行示例。
    print(f"GLM-130B parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("2-D position ids: enabled; blank-infilling training: enabled")
