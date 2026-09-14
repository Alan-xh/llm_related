# Grounding DINO

## 来源

- 论文：[Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)
- 官方代码：[IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

## 数据流与 Shape

```text
image [B,3,H,W] -> visual encoder memory [B,L,d]
text token ids [B,T] -> text encoder -> text features [B,T,d]
visual <-> text cross attention -> fused visual memory [B,L,d]
object queries -> decoder hidden [B,N,d]
hidden x text features -> token logits [B,N,T]
hidden -> objectness [B,N,1] and boxes [B,N,4]
```

预测类别不是固定的 `K` 个分类器，而是 query 与输入文本 token 的相似度；
因此同一个模型可以用不同短语查询不同目标。

## 核心公式

```text
s_{q,t} = exp(alpha) * cosine(h_q, e_t)
p_{q,t} = sigmoid(s_{q,t})
L = L_objectness + L_token + lambda_L1 * L_box + lambda_giou * L_giou
```

匹配代价由 token positive score、框 L1 和 GIoU 组成。推理时将最高 token
分数与 objectness 相乘并返回对应文本 token。

## 教学实现边界

本实现展示文本 embedding、视觉到文本 cross attention、token-level logits、
开放词汇解码和集合匹配；词表、分词器、预训练对齐、短语级 span、区域增强
和官方 CUDA/评测 pipeline 均使用 compact 教学替代。

## 运行

```bash
python architecture/detr/Grounding-DINO/model.py
python architecture/detr/Grounding-DINO/train.py --steps 2 --checkpoint grounding_dino_tiny.pt
python architecture/detr/Grounding-DINO/inference.py --checkpoint grounding_dino_tiny.pt
```
