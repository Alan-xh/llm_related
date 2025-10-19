# VoxCPM

* origin page 
https://github.com/OpenBMB/VoxCPM

## architecture
VoxCPM 是一种新型的无分词器文本到语音（TTS）系统，重新定义了语音合成的真实感。通过在连续空间中建模语音，它克服了离散分词的局限性，并实现了两大核心功能：上下文感知的语音生成和逼真的零样本语音克隆。\
与将语音转换为离散分词的主流方法不同，VoxCPM 采用端到端的扩散自回归架构，直接从文本生成连续的语音表示。基于 MiniCPM-4 骨干网络，通过层次语言建模和 FSQ 约束实现隐式语义-声学解耦，大幅提升了表达力和生成稳定性。

- DiTAR 提供语音生成的扩散自回归骨干。
- MiniCPM-4 作为语言模型基础。
- CosyVoice 提供基于流匹配的 LocDiT 实现。
- DAC 提供音频变分自编码器骨干。

## 