# SAM 3

## 来源

- 公开资料与代码入口：<https://github.com/facebookresearch/sam3>
- 任务方向：promptable concept segmentation，将文本、示例图和视觉 prompt 统一到概念检测、分割与跟踪接口。

本实现将文本 prompt 编码为一个 concept token，并保留点/框 prompt；它是可运行的架构教学版本，不依赖外部 tokenizer、检测器或官方 checkpoint。

## 数据流

```text
image/video → shared visual encoder → image feature [B,C,H/8,W/8]
text prompt → TextPromptEncoder → concept token [B,1,C]
exemplar box / click → PromptEncoder → visual tokens
all tokens + image feature → mask decoder
image feature pool → presence head
```

## Shape 表

| 节点 | Shape |
| --- | --- |
| Image | `[B,3,64,64]` |
| Image feature | `[B,64,8,8]` |
| Text token | `[B,1,64]` |
| Masks | `[B,3,64,64]` |
| IoU scores | `[B,3]` |
| Presence logits | `[B,1]` |

## 核心公式

概念存在性使用：

```text
p_concept = sigmoid(Linear(mean(image_feature)))
```

掩码部分沿用动态 hypernetwork：

```text
mask_k = hypernet_k(mask_token_k) · upscaled_image_feature
```

## 运行

```bash
python architecture/SAM/SAM3/model.py
python architecture/SAM/SAM3/train.py --steps 5 --checkpoint sam3_tiny.pt
python architecture/SAM/SAM3/inference.py --prompt rectangle --checkpoint sam3_tiny.pt
```
