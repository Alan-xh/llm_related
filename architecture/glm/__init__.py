"""GLM/ChatGLM 教学模型包。

各版本目录提供模型配置、训练入口和推理入口；公共的 Transformer、RoPE、
GQA/MQA、MoE、KV cache、byte tokenizer 与 prompt 工具位于 ``common.py``。
所有实现均以 [B, T] token ids 为输入，以 [B, T, V] logits 为输出。
"""
