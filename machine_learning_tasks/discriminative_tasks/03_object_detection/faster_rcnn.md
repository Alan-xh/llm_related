# Faster R-CNN 目标检测架构与接口文档

## 1. 架构总览

Faster R-CNN 是一种典型的两阶段 (Two-Stage) 目标检测神经网络，整体数据处理流程如下所示：

```text
  [Input Image] : [B, 3, 256, 256]
        │
        ▼
  [SimpleBackbone] (Downsample x16)
        │
        ├───> Feature Map : [B, 256, 16, 16]
        │          │
        │          ▼
        │    [RPN Head] ──────> 预测 Cls Logits [B, A, 2] & Box Deltas [B, A, 4]
        │          │
        │          ▼
        │    [Box Decode & NMS Filter]
        │          │
        │          ▼
        │    Selected Proposals : [N_roi, 4]
        │          │
        └──────────┼─────────────────────┐
                   ▼                     ▼
          [RoI Pooling Layer]  (Crop & Resize)
                   │
                   ▼
          Pooled Features : [N_roi, 256, 7, 7]
                   │
                   ▼
          [FC Layers & Classification Head]
                   │
                   ├───> Class Scores : [N_roi, NUM_CLASSES]
                   └───> Bounding Box Deltas : [N_roi, NUM_CLASSES * 4]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Image** | `[B, 3, 256, 256]` | - | 原始 RGB 图像 Batch |
| **Backbone Conv1+Pool** | `[B, 3, 256, 256]` | `[B, 64, 64, 64]` | Stride=2 卷积 + MaxPool 下采样 4 倍 |
| **Backbone Layer1-3** | `[B, 64, 64, 64]` | `[B, 256, 16, 16]` | 残差块下采样，最终输出 1/16 尺寸特征图 |
| **Anchor Generation** | Feature Size `(16, 16)` | `[2304, 4]` | 16x16 格点 × 9 个 Anchor (3 scale × 3 ratio) |
| **RPN Head (Conv)** | `[B, 256, 16, 16]` | `[B, 256, 16, 16]` | 3x3 卷积进行特征融合 |
| **RPN Cls / Box Logits** | `[B, 256, 16, 16]` | Cls: `[B, 2304, 2]`<br>

<br>Box: `[B, 2304, 4]` | 1x1 卷积展开预测二分类及 4 参数偏移量 |
| **NMS & Proposals** | `[2304, 4]` & Scores | `[128, 4]` | 经过解码和 NMS 抑制后保留得分最高的 Proposal |
| **RoI Pooling** | Feat: `[256, 16, 16]`<br>

<br>RoI: `[128, 4]` | `[128, 256, 7, 7]` | 截取对应区域特征并自适应最大池化为 7x7 |
| **FC Head Linear 1 & 2** | `[128, 12544]` | `[128, 1024]` | 展平特征向量送入双全连接层提取特征 |
| **Final Cls Score** | `[128, 1024]` | `[128, NUM_CLASSES]` | 最终多分类概率预测 |
| **Final Box Pred** | `[128, 1024]` | `[128, NUM_CLASSES * 4]` | 各类别专属的精细边界框偏移量预测 |

---

## 3. 核心公式与代码映射

| 数学含义 / 表达公式 | 代码函数/变量 | 映射说明 |
| --- | --- | --- |
| $IoU(A, B) = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)}$ | `box_iou()` | 计算坐标格式为 $[x1, y1, x2, y2]$ 的两组框交并比 |
| $t_x = \frac{g_x - p_x}{p_w}, t_y = \frac{g_y - p_y}{p_h}$ | `box_encode()` 中的 `gx - px / pw` | 将真实框坐标归一化编码为相对中心偏移量 |
| $t_w = \log(\frac{g_w}{p_w}), t_h = \log(\frac{g_h}{p_h})$ | `box_encode()` 中的 `torch.log(gw / pw)` | 缩放比例取对数映射 |
| $g_x = t_x \cdot p_w + p_x, g_w = \exp(t_w) \cdot p_w$ | `box_decode()` | 根据网络预测的偏移量复原真实预测框坐标 |
| $\text{Smooth}_{L1}(x) = \begin{cases} 0.5 x^2 & \text{if } \Vert{}x\Vert{} < 1 \\ \Vert{}x\Vert{} - 0.5 & \text{otherwise} \end{cases}$ | `F.smooth_l1_loss()` | 用于 RPN 与 Fast R-CNN 边界框回归鲁棒损失计算 |