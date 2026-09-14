# DN-DETR

## 来源

- 论文：[DN-DETR: Accelerate DETR Training by Introducing Query DeNoising](https://arxiv.org/abs/2203.01305)
- 代码参考：[IDEA-Research/DN-DETR](https://github.com/IDEA-Research/DN-DETR)

## 数据流与 Shape

```text
image [B,3,H,W] -> DETR encoder -> memory [L,B,d]
clean object queries [B,N,d]
noisy labels/boxes -> denoising queries [B,G*M,d]
concat queries -> decoder
split output:
  normal logits/boxes [B,N,K+1], [B,N,4]
  dn logits/boxes [B,G*M,K+1], [B,G*M,4]
```

`G` 是 denoising group 数，`M` 是当前 batch 中的最大目标数；推理时不构造
denoising queries。

## 核心公式

```text
y_tilde = corrupt(y)
L_DN = CE(cls_dn, cls) + lambda_L1 * L1(box_dn, box)
L = L_set(normal) + L_DN
```

噪声标签和框只作为 decoder 的训练提示，主分支仍通过 Hungarian matching
学习一对一集合预测。

## 教学实现边界

本实现展示 noisy label/box 构造、query 拼接、训练/推理分支和 denoising loss；
使用固定 group 数与 compact Transformer，不包含官方 attention mask、COCO
增强、分布式训练及完整辅助层损失。

## 运行

```bash
python architecture/detr/DN-DETR/model.py
python architecture/detr/DN-DETR/train.py --steps 2 --checkpoint dn_detr_tiny.pt
python architecture/detr/DN-DETR/inference.py --checkpoint dn_detr_tiny.pt
```
