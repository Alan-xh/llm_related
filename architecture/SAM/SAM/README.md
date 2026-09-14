# SAM

## 来源

- 论文：Kirillov et al., *Segment Anything*, ICCV 2023，<https://arxiv.org/abs/2304.02643>
- 官方实现：<https://github.com/facebookresearch/segment-anything>

## 数据流

```text
image [B,3,H,W] → TinyImageEncoder → image embedding [B,C,H/8,W/8]
point/box prompt → PromptEncoder → sparse [B,N,C], dense [B,C,H/8,W/8]
image + prompt → TwoWayTransformer → mask/IoU tokens
mask tokens → hypernetwork × upscaled image feature → masks + scores
```

## Shape 表

| 节点 | 默认输入 | 默认输出 |
| --- | --- | --- |
| Patch embedding | `[B,3,64,64]` | `[B,64,8,8]` |
| Point prompt | `[B,N,2]`, `[B,N]` | `[B,N,64]` |
| Decoder tokens | image + prompt | `[B,4+N,64]` |
| Dynamic masks | decoder feature | `[B,3,64,64]` |
| IoU head | IoU token | `[B,3]` |

## 核心公式

随机 Fourier 坐标编码：

```text
PE(x) = [sin(2πBx), cos(2πBx)]
```

动态 mask：

```text
mask_k(h,w) = hypernet_k(mask_token_k) · upscaled_image_feature(h,w)
```

教学损失为：

```text
L = 20 * BCEWithLogits(mask, target) + Dice(mask, target)
    + MSE(predicted_iou, detached_true_iou)
```

## 运行

```bash
python architecture/SAM/SAM/model.py
python architecture/SAM/SAM/train.py --steps 5 --checkpoint sam_tiny.pt
python architecture/SAM/SAM/inference.py --checkpoint sam_tiny.pt
```

实现使用小型 ViT 和合成矩形数据，仅用于阅读 tensor contract、prompt 编码和 mask decoder。
