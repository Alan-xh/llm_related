# MobileSAM

## 来源

- 论文：Zhang et al., *Faster Segment Anything: Towards Lightweight SAM*，2023，<https://arxiv.org/abs/2308.16184>
- 作者实现：<https://github.com/ChaoningZhang/MobileSAM>

## 数据流

```text
image [B,3,H,W] → depthwise-separable student → embedding [B,C,H/8,W/8]
prompt → shared SAM PromptEncoder
embedding + prompt → shared SAM MaskDecoder → masks [B,K,H,W], scores [B,K]
```

## Shape 与核心公式

默认配置下，输入 `[B,3,64,64]` 经过轻量编码器得到 `[B,64,8,8]`。prompt 和 decoder 的输出与 SAM v1 一致。

蒸馏目标对齐归一化后的全局 embedding：

```text
L_distill = MSE(
    normalize(student_embedding.flatten(1)),
    normalize(stop_gradient(teacher_embedding.flatten(1)))
)
```

`distillation_loss` 可直接接入 teacher/student 训练；最小训练循环优化 mask BCE、Dice 和 IoU 质量损失。

## 运行

```bash
python architecture/SAM/MobileSAM/model.py
python architecture/SAM/MobileSAM/train.py --steps 5 --checkpoint mobile_sam_tiny.pt
python architecture/SAM/MobileSAM/inference.py --checkpoint mobile_sam_tiny.pt
```

教学实现只替换 image encoder，保留 SAM 的 prompt/mask 接口，便于比较参数量和边缘设备延迟。
