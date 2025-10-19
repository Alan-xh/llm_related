from voxcpm.core import VoxCPM
import soundfile as sf
import numpy as np
import time
import torchaudio

model = VoxCPM.from_pretrained(
    hf_model_id="./models/openbmb/VoxCPM-0.5B",
    zipenhancer_model_id="./models/iic/speech_zipenhancer_ans_multiloss_16k_base",
)

# 显存占用 2690MiB

prompt_wav_path = "./examples/guqinhuan.wav"
prompt_text = "今天天气很好。"

audio, sr = torchaudio.load(prompt_wav_path)
print(f"sr:{sr}")

# 非流式生成
start_time = time.time()
wav = model.generate(
    # text="VoxCPM 是 ModelBest 开发的创新端到端 TTS 模型，旨在生成高表达力的语音。",
    text="最近你有没有觉得，只要别人稍微对你不好，或者让你觉得受了委屈，你就会特别生气，甚至想狠狠地回击？比如在监舍里，有人不小心碰了你一下，或者说了句不中听的话，你就特别想争个高低，甚至想动手？",
    prompt_wav_path=prompt_wav_path,  # 可选：用于语音克隆的提示音频路径
    prompt_text=prompt_text,  # 可选：参考文本
    cfg_value=2.0,  # LocDiT 的语言模型引导值，值越高越贴近提示，但可能影响质量
    inference_timesteps=20,  # LocDiT 推理步数，步数越高结果越好，步数越低速度越快
    normalize=True,  # 启用外部文本规范化工具
    denoise=True,  # 启用外部降噪工具
    retry_badcase=True,  # 启用重试模式以处理某些异常情况（无法停止）
    retry_badcase_max_times=3,  # 最大重试次数
    retry_badcase_ratio_threshold=6.0,  # 异常检测的最大长度限制（简单但有效），可为慢节奏语音调整
)

print(
    "推理结果", wav
)  # [ 4.6799338e-05  5.5606775e-05  7.7712044e-05 ... -2.7966246e-04 -1.1092706e-03 -1.1030500e-03]
print("输出维度", wav.shape)  # (176640,)
print(f"推理时长：{time.time() - start_time:.2f}")
sf.write("output03.wav", wav, 16000)
print("保存文件：output03.wav")

# # 流式生成
# chunks = []
# for chunk in model.generate_streaming(
#     text="使用 VoxCPM 进行流式文本到语音转换非常简单！",
#     # 支持与上述相同的参数
# ):
#     chunks.append(chunk)
# wav = np.concatenate(chunks)

# sf.write("output_streaming.wav", wav, 16000)
# print("保存文件：output_streaming.wav")
