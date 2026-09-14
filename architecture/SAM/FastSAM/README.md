# FastSAM

## 来源

- 论文：Zhao et al., *Fast Segment Anything*，2023，<https://arxiv.org/abs/2308.05960>
- 作者实现：<https://github.com/CASIA-IVA-Lab/FastSAM>

## 数据流

```text
image → real-time encoder → N candidate masks + candidate scores
point/box prompt → sample/IoU/distance matching → top-K candidates
```

与 SAM 的主要区别是 image encoder 只运行一次，prompt 主要负责候选筛选。

## Shape 表与公式

| 节点 | Shape |
| --- | --- |
| Feature map | `[B,64,8,8]` |
| All candidates | `[B,N,64,64]` |
| Candidate scores | `[B,N]` |
| Selected output | `[B,3,64,64]`, `[B,3]` |

点 prompt 的简化排序分数为：

```text
score = 0.5 * model_score + 0.5 * prompt_agreement
prompt_agreement = mean(mask(point)) for positive points
                   + mean(1-mask(point)) for negative points
```

`candidate_iou` 提供候选框之间的 pairwise IoU，用于扩展框 prompt 的筛选逻辑。

## 运行

```bash
python architecture/SAM/FastSAM/model.py
python architecture/SAM/FastSAM/train.py --steps 5 --checkpoint fast_sam_tiny.pt
python architecture/SAM/FastSAM/inference.py --all-candidates
python architecture/SAM/FastSAM/inference.py --checkpoint fast_sam_tiny.pt
```

这里的候选生成器是教学版，不包含官方 YOLO 实例分割 backbone、NMS 和文本检测器。
