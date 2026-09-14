# SAM 3.1

## 来源

- 版本说明与代码入口：沿用本目录 SAM 3 概念分割约定，<https://github.com/facebookresearch/sam3>
- 版本定位：面向多目标视频推理的 shared-memory Object Multiplex 教学实现。

本目录把对象 prompt 打包后共享一次 image encoder，再按对象维度解码；它用于解释吞吐优化的 tensor 组织方式，不是官方权重或完整视频 tracker 的复刻。

## 数据流

```text
image → one shared image encoder → feature [1,C,H/8,W/8]
O text/object prompts → object tokens [O,1,C]
pack into buckets → shared decoder forward
unpack by object identity → masks [1,O,K,H,W]
```

## Shape 表

| 节点 | Shape |
| --- | --- |
| Shared image feature | `[1,64,8,8]` |
| Object tokens | `[O,1,64]` |
| Decoder batch | `[O,64,8,8]` |
| Multiplex masks | `[1,O,3,64,64]` |
| Multiplex scores | `[1,O,3]` |

## 核心公式

对象 token 的打包/解包保持每个对象的长度：

```text
packed = concat(token_1, ..., token_O)
tokens_i = split(packed, lengths)
```

多目标路径的主要收益来自 image encoder 从 `O` 次变为 1 次；decoder 仍按对象 batch 并行计算。

## 运行

```bash
python architecture/SAM/SAM3.1/model.py
python architecture/SAM/SAM3.1/train.py --steps 5 --checkpoint sam31_tiny.pt
python architecture/SAM/SAM3.1/inference.py --prompts rectangle object shape
```

`multiplex` 当前要求 batch size 为 1，以便清楚展示“共享一张图、多个对象 token”的数据流。
