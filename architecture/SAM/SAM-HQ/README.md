# SAM-HQ

## 来源

- 论文：Ke et al., *Segment Anything in High Quality*，2023，<https://arxiv.org/abs/2306.01567>
- 作者实现：<https://github.com/SysCV/SAM-HQ>

## 数据流

```text
image → SAM image encoder → compressed feature ─────┐
                                      └→ high-res projection
prompt → SAM prompt encoder ────────────────────────┤
                                                    ↓
                         Two-Way Transformer + HQ feature fusion
                                                    ↓
                                  dynamic masks + IoU scores
```

## Shape 表

输入 `[B,3,64,64]`，压缩图像特征为 `[B,64,8,8]`。decoder 的上采样特征为 `[B,16,32,32]`，HQ projection 插值到同一空间后相加，最终输出 `[B,3,64,64]` 与 `[B,3]`。

## 核心公式

原始动态 mask：

```text
M_base = Hyper(mask_token) · U(image_feature)
```

HQ 分支在相同 mask embedding 空间注入高质量特征：

```text
U_hq = U(image_feature) + Conv1x1(Interpolate(high_res_feature))
M_hq = Hyper(mask_token) · U_hq
```

训练仍使用：

```text
L = 20 * BCE + Dice + IoU-MSE
```

## 运行

```bash
python architecture/SAM/SAM-HQ/model.py
python architecture/SAM/SAM-HQ/train.py --steps 5 --checkpoint sam_hq_tiny.pt
python architecture/SAM/SAM-HQ/inference.py --checkpoint sam_hq_tiny.pt
```
