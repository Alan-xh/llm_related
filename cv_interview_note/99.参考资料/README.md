# 99.参考资料

CV 面试与学习相关的经典论文、课程、工具与网站汇总（按仓库章节分组，仅列公认著名的工作）。

## 1.经典论文（按章节）

### 01.计算机视觉基础

- Distinctive Image Features from Scale-Invariant Keypoints（SIFT, 2004）
- SURF: Speeded Up Robust Features（2006）
- Fast Approximate Energy Minimization via Graph Cuts（1999）

### 02.CNN与经典架构

- ImageNet Classification with Deep CNNs（AlexNet, 2012）
- Very Deep Convolutional Networks（VGG, 2014）
- Going Deeper with Convolutions（GoogLeNet, 2014）
- Deep Residual Learning for Image Recognition（ResNet, 2015）
- Aggregated Residual Transformations（ResNeXt, 2017）
- MobileNets: Efficient CNNs for Mobile Vision（2017）
- ShuffleNet: An Extremely Efficient CNN（2017）
- Squeeze-and-Excitation Networks（SENet, 2018）
- Random Depthwise Separable Convolution 系列：GhostNet（2020）
- EfficientNet: Rethinking Model Scaling（2019）
- RepVGG: Making VGG-style ConvNets Great Again（2021）

### 03.视觉Transformer

- An Image is Worth 16x16 Words（ViT, 2020）
- Swin Transformer: Hierarchical Vision Transformer（2021）
- Training data-efficient image transformers（DeiT, 2021）
- BEiT: BERT Pre-training of Image Transformers（2021）
- Masked Autoencoders Are Scalable Vision Learners（MAE, 2021）

### 04.训练数据与增强

- ImageNet: A Large-Scale Hierarchical Image Database（2009）
- AutoAugment: Learning Augmentation Policies（2019）
- mixup: Beyond Empirical Risk Minimization（2018）
- RandAugment（2020）

### 05.训练技术

- Batch Normalization: Accelerating Deep Network Training（2015）
- Adam: A Method for Stochastic Optimization（2015）
- Batch Normalization 之外：Group Normalization（2018）
- Deep Mutual Learning / 相关蒸馏见第 10 章

### 06.目标检测

- Rich feature hierarchies for object detection（R-CNN, 2014）
- Fast R-CNN（2015）、Faster R-CNN（2015）
- Feature Pyramid Networks for Object Detection（FPN, 2017）
- Mask R-CNN（2017）
- You Only Look Once: Unified, Real-Time Object Detection（YOLO, 2016）
- YOLO9000 / YOLOv3（2017/2018）
- SSD: Single Shot MultiBox Detector（2016）
- Focal Loss for Dense Object Detection（RetinaNet, 2017）
- Deformable Convolutional Networks（DCN, 2017）
- CornerNet（2018）、FCOS（2019）
- End-to-End Object Detection with Transformers（DETR, 2020）
- Deformable DETR（2021）、DINO（2022）
- COCO: Common Objects in Context（数据集, 2014）

### 07.图像分割

- Fully Convolutional Networks for Semantic Segmentation（FCN, 2015）
- U-Net: Convolutional Networks for Biomedical Image Segmentation（2015）
- DeepLab 系列（v1~v3+, 2014-2018）
- PSPNet: Scene Parsing through Pyramid Pooling（2017）
- SegFormer: Simple and Efficient Design for Semantic Segmentation（2021）
- Mask R-CNN（实例分割, 2017）
- YOLACT: Real-time Instance Segmentation（2019）
- Segment Anything（SAM, 2023）、SAM 2: Segment Anything in Images and Videos（2024）
- MaskFormer（2021）、Mask2Former（2022）
- Panoptic Segmentation（2018）

### 08.生成模型

- Auto-Encoding Variational Bayes（VAE, 2013）
- Generative Adversarial Nets（GAN, 2014）
- Unsupervised Representation Learning with GANs（DCGAN, 2015）
- Image-to-Image Translation with Conditional GANs（pix2pix, 2017）
- A Style-Based Generator Architecture for GANs（StyleGAN, 2019）
- Progressive Growing of GANs（PGGAN, 2017）
- Denoising Diffusion Probabilistic Models（DDPM, 2020）
- High-Resolution Image Synthesis with Latent Diffusion Models（Stable Diffusion / LDM, 2022）
- Denoising Diffusion Implicit Models（DDIM, 2020）

### 09.多模态与大视觉模型

- Learning Transferable Visual Models From Natural Language Supervision（CLIP, 2021）
- BLIP: Bootstrapping Language-Image Pre-training（2022）、BLIP-2（2023）
- Visual Instruction Tuning（LLaVA, 2023）
- DINOv2: Learning Robust Visual Features without Supervision（2023）

### 10.模型评估与部署

- Distilling the Knowledge in a Neural Network（Hinton KD, 2015）
- Learning both Weights and Connections for Efficient Neural Networks（2015）
- Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference（2018）
- Straight-Through Estimator 相关：Bengio et al. 2013
- Deep Learning with Low Precision by Revisiting BatchNorm（相关量化系列）

## 2.开源课程

- [斯坦福 CS231n: Deep Learning for Computer Vision](https://cs231n.stanford.edu/ "斯坦福CS231n官网")：CV 入门首选，见 [98.相关课程/斯坦福CS231n](/98.相关课程/斯坦福CS231n/斯坦福CS231n.md)
- [CS231n 课程讲义](https://cs231n.github.io/ "CS231n Notes")：文字版 notes，复习效率高
- [Deep Learning（Goodfellow 等）](https://www.deeplearningbook.org/ "Deep Learning Book")：经典教材，免费在线阅读
- [Dive into Deep Learning（d2l）](https://d2l.ai/ "动手学深度学习")：代码驱动，中英文双版
- [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/unit0/1 "HF 扩散模型课程")：扩散模型实战
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome "HF 视觉课程")：覆盖传统 CV 到 ViT/CLIP
- 姊妹仓库 [llm_interview_note](../../../llm_interview_note/README.md)：LLM 方向面试笔记（含清华大模型公开课链接）

## 3.常用工具与网站

### 论文与代码

- [Papers with Code](https://paperswithcode.com/ "Papers with Code")：论文 + 代码 + SOTA 榜单（现已归档为只读，仍可查历史榜单）
- [arXiv](https://arxiv.org/ "arXiv")：预印本主站（cs.CV 分类）
- [arXiv-sanity](https://arxiv-sanity-lite.com/ "arXiv-sanity lite")：arXiv 浏览与推荐
- [Semantic Scholar](https://www.semanticscholar.org/ "Semantic Scholar")：文献检索与引用关系
- [Connected Papers](https://www.connectedpapers.com/ "Connected Papers")：论文关系图

### 模型与数据集

- [Hugging Face](https://huggingface.co/ "Hugging Face")：模型/数据集/Spaces 生态
- [PyTorch](https://pytorch.org/ "PyTorch") / [TorchVision](https://pytorch.org/vision/stable/index.html "TorchVision")：训练框架与视觉工具
- [OpenMMLab](https://openmmlab.com/ "OpenMMLab")：MMDetection / MMSegmentation 等 CV 工具箱
- [Kaggle](https://www.kaggle.com/ "Kaggle")：数据集与竞赛
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/ "PASCAL VOC") / [COCO](https://cocodataset.org/ "COCO")：检测分割标准数据集

### 部署与推理

- [ONNX](https://onnx.ai/ "ONNX")：开放模型交换格式
- [TensorRT](https://developer.nvidia.com/tensorrt "TensorRT")：NVIDIA 推理引擎
- [onnxruntime](https://onnxruntime.ai/ "onnxruntime")：跨平台 ONNX 推理
- [OpenVINO](https://docs.openvino.ai/ "OpenVINO")：Intel 推理引擎
- [NCNN](https://github.com/Tencent/ncnn "NCNN")：腾讯移动端推理
- [MNN](https://github.com/alibaba/MNN "MNN")：阿里移动端推理
- [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/ "Triton")：服务化框架

### 可视化与调试

- [Netron](https://netron.app/ "Netron")：模型结构可视化（ONNX/TRT/tflite 等）
- [Weights & Biases](https://wandb.ai/ "wandb") / TensorBoard：训练监控

## 4.面试题仓库

- [LLMs_interview_notes](https://github.com/km1994/LLMs_interview_notes "大模型算法工程师面试题")：LLM 面试题
- [DA-southampton/NLP_ability](https://github.com/DA-southampton/NLP_ability "深度学习自然语言处理")：NLP 面试整理
- [amusi/CV-interview](https://github.com/amusi/CV-interview "CV面试")：CV 面试题合集
- [inzva/Awesome-Deep-Learning-Interview](https://github.com/inzva/Awesome-Deep-Learning-Interview "深度学习面试")：深度学习面试题

## 5.说明

- 以上仅收录确实存在且广为引用的论文与资源；
- 论文年份以 arXiv/正式发表为准，可能有 1 年内浮动；
- 若链接失效，建议按论文标题搜索最新出处。
