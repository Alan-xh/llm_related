# Mask R-CNN 实例分割架构与接口技术文档

## 1. 架构总览与数据流图

Mask R-CNN 是在 Faster R-CNN 基础上的扩展，增加了一个用于预测像素级 Mask 的全卷积分支。

```
                       [ Input Image: B x 3 x H x W ]
                                     │
                                     ▼
                      [ Backbone Network (ResNet-like) ]
                                     │
                                     ▼
                    [ Feature Map: B x 256 x H/16 x W/16 ]
                                     │
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
   [ Region Proposal Network ]                     [ Anchor Generation ]
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     ▼
                         [ Decode & Apply NMS ]
                                     │
                                     ▼
                          [ Top-K Proposals ]
                                     │
                                     ▼
                      [ RoI Pooling / Alignment ]
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
[ Fast R-CNN Head (FC Layers) ]                   [ Mask Head (FCN Branch) ]
            │                                                 │
   ┌────────┴────────┐                                        │
   ▼                 ▼                                        ▼
[Cls Logits]   [Box Deltas]                          [Class-Specific Masks]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

假设输入图像尺寸为 `[B, 3, 128, 128]`，批次大小 $B = 2$：

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Images** | `[B, 3, 128, 128]` | - | 原始 RGB 图像输入 |
| **Backbone Stage 1** | `[B, 3, 128, 128]` | `[B, 64, 32, 32]` | Conv7x7 (stride=2) + MaxPool2d (stride=2) 下采样 4 倍 |
| **Backbone Stage 2 & 3** | `[B, 64, 32, 32]` | `[B, 256, 8, 8]` | 残差卷积组进一步下采样 4 倍，总 Stride = 16 |
| **Anchors Generation** | `[8, 8]` (Feature HW) | `[576, 4]` | 每个特征点生成 $3 \times 3 = 9$ 个锚框，共 $8 \times 8 \times 9 = 576$ 个 |
| **RPN Head Cls / Bbox** | `[B, 256, 8, 8]` | `[B, 576, 2]`, `[B, 576, 4]` | 1x1 卷积预测每个 Anchor 的前景/背景概率与 4 维偏移量 |
| **NMS Proposals** | `[576, 4]` | `[N_prop, 4]` (e.g. `128, 4`) | 解码后根据分数过滤，并保留 Top-K (例如 128) 个候选框 |
| **RoI Sampled Proposals** | `[N_prop, 4]` | `[N_roi, 4]` (e.g. `64, 4`) | 与 GT 框计算 IoU 采样正负样本，采样总数为 `ROI_BATCH_SIZE` |
| **RoI Pooling** | `[256, 8, 8]`, `[N_roi, 4]` | `[N_roi, 256, 7, 7]` | 将不同尺度的 BBox 区域池化为统一的 $7 \times 7$ 特征块 |
| **Fast R-CNN FC1 & FC2** | `[N_roi, 256*7*7]` | `[N_roi, 1024]` | 展平特征后经过两层全连接隐藏层 |
| **Cls Score & Bbox Pred** | `[N_roi, 1024]` | `[N_roi, Num_Classes]`, `[N_roi, Num_Classes*4]` | 预测分类 Logits 与类特定的框回归增量 |
| **Mask RoI Features** | `[256, 8, 8]`, `[N_pos, 4]` | `[N_pos, 256, 7, 7]` | 仅针对 **正样本 (Positive RoIs)** 重新提取 RoI 特征 |
| **Mask ConvTranspose2d** | `[N_pos, 256, 7, 7]` | `[N_pos, 256, 14, 14]` | 反卷积层将特征图空间尺寸上采样 2 倍 |
| **Mask Predictor Output** | `[N_pos, 256, 14, 14]` | `[N_pos, Num_Classes, 28, 28]` | 1x1 卷积输出类别特定的像素级分割 Logits |

---

## 3. 核心公式与代码映射

| 数学含义 / 算法模块 | 标准公式表示 | 代码变量 / 函数实现 |
| --- | --- | --- |
| **Bounding Box Center** | $p_x = \frac{x_1 + x_2}{2}, p_y = \frac{y_1 + y_2}{2}$ | `px = (proposal[:, 0] + proposal[:, 2]) * 0.5` |
| **Box Dimension** | $p_w = x_2 - x_1, p_h = y_2 - y_1$ | `pw = proposal[:, 2] - proposal[:, 0]` |
| **BBox Encoding Translation** | $d_x = (g_x - p_x) / p_w, d_y = (g_y - p_y) / p_h$ | `dx = (gx - px) / pw`, `dy = (gy - py) / ph` |
| **BBox Encoding Scaling** | $d_w = \log(g_w / p_w), d_h = \log(g_h / p_h)$ | `dw = torch.log(gw / pw)`, `dh = torch.log(gh / ph)` |
| **BBox Decoding Center** | $g_x = p_x + d_x \cdot p_w, g_y = p_y + d_y \cdot p_h$ | `gx = delta[:, 0] * pw + px`, `gy = delta[:, 1] * ph + py` |
| **BBox Decoding Dim** | $g_w = p_w \cdot \exp(d_w), g_h = p_h \cdot \exp(d_h)$ | `gw = torch.exp(delta[:, 2]) * pw`, `gh = torch.exp(delta[:, 3]) * ph` |
| **Smooth L1 Loss** | $L_{\text{smooth1}}(x) = \begin{cases} 0.5 x^2 & \text{if } \Vert{}x\Vert{} < 1 \\ \Vert{}x\Vert{} - 0.5 & \text{otherwise} \end{cases}$ | `F.smooth_l1_loss(bbox_preds, bbox_targets)` |
| **Mask Binary CE Loss** | $-\frac{1}{m^2}\sum [y_{ij}\log \hat{y}_{ij} + (1-y_{ij})\log(1-\hat{y}_{ij})]$ | `F.binary_cross_entropy_with_logits(mask_preds, mask_targets)` |