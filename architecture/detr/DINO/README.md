# DINO

## 来源

- 论文：[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)
- 代码参考：[IDEA-Research/DINO](https://github.com/IDEA-Research/DINO)

## 数据流与 Shape

```text
image [B,3,H,W] -> encoder memory [B,L,d]
encoder class/box proposals -> top-k query selection [B,N,d], [B,N,4]
contrastive denoising labels/boxes -> [B,G*M,d]
decoder -> normal logits/boxes [B,N,K+1], [B,N,4]
        -> denoising logits/boxes [B,G*M,K+1], [B,G*M,4]
```

模型先从 encoder proposal 中选择 query，再用 box 位置编码初始化 decoder；
训练时同时加入带正负扰动的 denoising query。

## 核心公式

```text
q = TopK(foreground_score(encoder(memory)))
L = L_set(q) + L_DN^contrastive
L_DN = CE(noisy_cls, clean_cls) + lambda_L1 * L1(noisy_box, clean_box)
```

## 教学实现边界

本实现保留 two-stage query selection、anchor box positional encoding 和
contrastive denoising 的接口；使用 compact encoder、固定 query 数和合成数据，
不复刻官方多尺度 backbone、完整 contrastive group mask、COCO recipe 和评测工具。

## 运行

```bash
python architecture/detr/DINO/model.py
python architecture/detr/DINO/train.py --steps 2 --checkpoint dino_tiny.pt
python architecture/detr/DINO/inference.py --checkpoint dino_tiny.pt
```
