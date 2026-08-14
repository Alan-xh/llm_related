# CNN面试题

## 1.CNN 基础类

### Q1：卷积运算的输出尺寸公式？

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

### Q2：为什么 CNN 比全连接网络更适合图像？

1. **参数高效**：权值共享让参数量从 $O(HWC \cdot K^2)$ 降为 $O(K^2 C)$；
2. **保留空间结构**：不展平，保留 H/W 维度；
3. **平移等变**：目标在图像中移动，特征同步移动。

### Q3：CNN 的三大特性？

1. **局部感受野**：每神经元只看局部；
2. **权值共享**：同一核在整图滑动，参数共享；
3. **平移等变/不变**：卷积是等变，池化提供不变性。

### Q4：感受野的计算公式？

$$
RF_l = RF_{l-1} + (k_l - 1) \cdot \prod_{i=1}^{l-1} s_i
$$

### Q5：1×1 卷积有什么用？

1. 通道升降维；
2. 跨通道融合；
3. 增加非线性；
4. 降参。

### Q6：空洞卷积的作用？缺点？

- 作用：不增加参数和计算量地扩大感受野；
- 缺点：网格效应（gridding artifact），可通过 HDC（混合不同 rate）缓解。

### Q7：深度可分离卷积省了多少参数？

3×3 卷积时约为 $1/C_{out} + 1/9$，通常 **1/8 ~ 1/9**。

### Q8：分组卷积的作用？

1. 参数量降为 $1/g$；
2. 不同组可学习不同特征；
3. AlexNet 双 GPU 历史起源，ResNeXt 系统化研究。

### Q9：可变形卷积是什么？解决什么问题？**

为每个位置学习一个偏移量，使采样点不再固定为正方形网格，适应物体几何变形（旋转、形变）。

### Q10：转置卷积为什么会产生棋盘效应？怎么解决？

核大小不能被步长整除时，相邻输出位置由不同数量输入点贡献，产生不均匀。解决：

1. 选核大小为步长倍数；
2. 用 `resize-conv`（先 upsample 再 conv）替代；
3. sub-pixel convolution（PixelShuffle）。

## 2.经典网络类

### Q11：为什么用小卷积核（3×3）替代大卷积核？

1. 参数更少：2 个 3×3 = 18C² vs 1 个 5×5 = 25C²；
2. 非线性更强；
3. 感受野相同。

### Q12：AlexNet 的三大创新？

1. **ReLU** 替代 tanh，训练加速 6 倍；
2. **Dropout** 防过拟合；
3. **GPU 训练** + 数据增强。

### Q13：VGG 参数量为什么这么大？

FC6 输入 7×7×512 = 25088，输出 4096，单层 1 亿参数；FC 层占总参数 90%+。

### Q14：GoogLeNet Inception 模块的 1×1 卷积有什么用？

降维。先 1×1 把通道数降下来再做 3×3/5×5 卷积，参数量可减少数倍。

### Q15：GoogLeNet 为什么用 GAP 替代 FC？

- FC 参数量大；
- GAP 无参数、保留空间结构、抗过拟合；
- 适合高分辨率 feature map。

## 3.残差网络类

### Q16：ResNet 解决了什么问题？怎么解决的？

**退化问题**：深层网络训练误差反而升高。通过**残差连接**让网络学 $F(x) = H(x) - x$，"什么都不做"（恒等映射）成为容易学的解，shortcut 提供梯度高速公路。

### Q17：残差连接为什么能缓解梯度消失？

反向传播时 $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial H} (\frac{\partial F}{\partial x} + 1)$，加法中的 1 保证梯度可无衰减传到浅层。

### Q18：ResNet-50 和 ResNet-34 的区别？

- 34 用 BasicBlock（2 层 3×3）；50 用 Bottleneck（1×1 → 3×3 → 1×1）；
- 50 更深但参数量更少（Bottleneck 用 1×1 降维）；
- 50 性能更好。

### Q19：Bottleneck 为什么用 1×1 降维？

3×3 卷积在低维空间计算，参数和 FLOPs 大幅减少，相同计算预算下能堆更深。

### Q20：shortcut 何时需要 1×1 投影？

输入输出通道数或 stride 不匹配时，需 1×1 卷积把 shortcut 变换到与主路径输出一致。

### Q21：ResNet-v2 pre-activation 的好处？

1. 最后 BN 不在主路径，信息流动更直接；
2. 训练更稳定，能训 1000 层；
3. 正则化更强。

### Q22：ResNet 各 stage 输出尺寸？

输入 224 -> conv1 (112) -> maxpool (56) -> conv2 (56) -> conv3 (28) -> conv4 (14) -> conv5 (7)。

### Q23：FPN 解决什么问题？

多尺度目标检测：浅层分辨率高但语义弱，深层语义强但分辨率低。FPN 通过自顶向下 + 横向连接融合两者。

## 4.轻量化网络类

### Q24：深度可分离卷积为什么省参数？

把 $k \cdot k \cdot C_{in} \cdot C_{out}$ 拆为 $k \cdot k \cdot C_{in} + C_{in} \cdot C_{out}$，3×3 时约 1/9。

### Q25：MobileNetV2 倒残差与 ResNet 残差的区别？

| 维度 | ResNet | MobileNetV2 |
| --- | --- | --- |
| 形状 | 宽->窄->宽 | 窄->宽->窄 |
| shortcut 位置 | 宽 | 窄 |
| 最后激活 | ReLU | 线性 |

### Q26：MobileNetV2 最后为什么不用激活？

低维特征经 ReLU 会丢失大量信息（负半轴置零）。实验证明加 ReLU6 掉 2~3%。

### Q27：ShuffleNet 的 channel shuffle 解决什么？

分组卷积使信息只能在组内流动。shuffle 重排通道，使下一层分组卷积能接收不同组通道，实现跨组交流。

### Q28：ShuffleNetV2 的 4 条原则？

1. 输入输出通道数相等时 MAC 最小；
2. 过度分组增加 MAC；
3. 网络碎片化降低并行度；
4. element-wise 操作不可忽视。

### Q29：RepVGG 的重参数化怎么做？

1. 1×1 卷积 padding 成 3×3；
2. BN 融合到卷积：$W' = W \cdot \gamma / \sqrt{\sigma^2 + \epsilon}, b' = \beta - \mu \cdot \gamma / \sqrt{\sigma^2 + \epsilon}$；
3. 恒等映射视为特殊 1×1 卷积；
4. 3 个 3×3 卷积核 + bias 相加得到单个 3×3 卷积。

### Q30：FLOPs 低就一定快吗？

不一定。实际速度受**内存访问开销、并行度、硬件特性**影响。DW 卷积 FLOPs 极低但实际不快，多分支网络 FLOPs 低但内存访问多。应以**硬件 latency** 为准。

## 5.注意力机制类

### Q31：SENet 的工作原理？

- Squeeze：GAP 把每通道 $H \times W$ 压成 1 个标量；
- Excitation：FC-ReLU-FC-Sigmoid 得到通道权重；
- Reweighting：原 feature map 通道维乘权重。

### Q32：SE 为什么 FC 降维再升维？

1. 减少参数；
2. 增加非线性；
3. 让通道间信息交互。

### Q33：ECA 为什么比 SE 好？

- 不降维，避免信息损失；
- 1D 卷积让每通道只与相邻通道交互，更高效；
- 参数极少。

### Q34：CBAM 的空间注意力怎么做的？

沿通道维 AvgPool + MaxPool 得到 2 个 $H \times W$ 图，concat 后 7×7 卷积降为 1 通道，Sigmoid 得到空间权重。

### Q35：Non-local 与 Transformer 自注意力区别？

- Non-local：单头、无位置编码、是 CNN 插件；
- Transformer：多头、有位置编码、有 FFN 和残差、是端到端架构。

## 6.综合题

### Q36：CNN 浅层学到什么特征？为什么？

学到**边缘、角点、颜色块**等低层特征。原因：

1. 自然图像统计特性决定这些特征高频、强响应；
2. 梯度下降偏好高响应特征；
3. 浅层感受野小，只能学局部特征。

### Q37：为什么 ResNet 之后用 stride=2 卷积替代池化？

卷积有可学习参数，能更好地保留下采样所需信息；池化是固定操作（max/avg），不可学习。

### Q38：BN 和 LN 在 CV 中怎么选？

- BN：在 batch + H/W 维度归一化，保留通道间差异，适合 CV（不同通道是不同特征）；
- LN：在 H/W/C 维度归一化，会破坏通道差异，但**不依赖 batch size**；
- 小 batch（< 8）用 GN（Group Norm）或 LN；正常 batch 用 BN。

### Q39：模型压缩有哪些方法？

1. **轻量化网络设计**（MobileNet 系列）；
2. **剪枝**（结构化 / 非结构化）；
3. **量化**（INT8、FP16、混合精度）；
4. **蒸馏**（KD）；
5. **低秩分解**（SVD、Tucker）；
6. **NAS**（搜索高效结构）；
7. **重参数化**（RepVGG）。

### Q40：如何选择 backbone？

| 场景 | 推荐 |
| --- | --- |
| 高精度优先 | ResNet-101/152、EfficientNet-B5+、ConvNeXt-L+ |
| 平衡 | ResNet-50、ConvNeXt-T、Swin-T |
| 移动端 | MobileNetV3、ShuffleNetV2、EfficientNet-Lite |
| 极致速度 | RepVGG-A0、SqueezeNet |
| 检测分割 | ResNet-50 + FPN、Swin + FPN |
| 视频理解 | ResNet-50 3D、SlowFast、TimeSformer |

### Q41：CNN 和 ViT 的本质区别？

| 维度 | CNN | ViT |
| --- | --- | --- |
| 归纳偏置 | 平移等变 + 局部性 | 无（数据驱动） |
| 数据效率 | 小数据表现好 | 需大数据 |
| 长距离依赖 | 弱 | 强 |
| 计算复杂度 | 与分辨率线性 | $O(N^2)$，二次 |
| 部署友好 | 是 | 一般 |

### Q42：如何理解"深度学习是表示学习"？

- 模型自动从原始数据中学习多层次、由低到高的特征表示；
- 浅层学边缘纹理、中层学部件、深层学语义；
- 不需手工设计特征（如 SIFT/HOG），端到端学习；
- 表示质量决定最终性能。

### Q43：如何理解 CNN 的"层次化特征"？

- 浅层：边缘、角点、颜色（Gabor 类）；
- 中层：纹理、简单部件（眼睛、轮子）；
- 深层：物体类别、场景语义；
- 每层都是上一层的组合，形成层次化抽象。

### Q44：CNN 在 ViT 时代还有价值吗？

有：

1. **小数据场景**：CNN 归纳偏置强，小数据表现更稳；
2. **部署友好**：推理快、量化稳、内存占用少；
3. **特定任务**：医学影像、工业质检等数据量小的场景；
4. **混合架构**：CNN + Transformer 结合，如 CoAtNet、MobileViT；
5. **ConvNeXt** 证明精心设计的 CNN 仍能与 ViT 抗衡。

### Q45：CNN 训练技巧有哪些？

1. **数据增强**：Mixup、CutMix、RandAugment、AutoAugment；
2. **学习率**：Warmup + Cosine Annealing；
3. **正则化**：Dropout、DropPath、Label Smoothing；
4. **优化器**：SGD + Momentum 或 AdamW；
5. **归一化**：BN（或 GN/LN）；
6. **训练时长**：300+ epochs（ImageNet）；
7. **EMA**：指数滑动平均权重；
8. **混合精度**：FP16 加速。
