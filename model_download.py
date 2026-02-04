# SDK模型下载
from modelscope import snapshot_download

# 语音识别模型
# snapshot_download(
#     "FunAudioLLM/Fun-ASR-Nano-2512", local_dir="pretrained_models/Fun-ASR-Nano-2512"
# )
# snapshot_download("iic/SenseVoiceSmall", local_dir="pretrained_models/SenseVoiceSmall")
# snapshot_download(
#     "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
#     local_dir="pretrained_models/paraformer-zh-streaming",
# )
# snapshot_download(
#     "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
#     local_dir="pretrained_models/paraformer-zh",
# )


# 声音检测模型
# snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',local_dir='pretrained_models/speech_fsmn_vad_zh-cn-16k-common-pytorch')
# snapshot_download(
#     "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
#     local_dir="pretrained_models/speech_fsmn_vad",
# )

# 标点符号恢复模型
# model_dir = snapshot_download(
#     "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
#     local_dir="pretrained_models/punc_ct",
# )


# 说话人验证分离模型
# snapshot_download(
#     "iic/speech_campplus_sv_zh-cn_16k-common",
#     local_dir="pretrained_models/cam++",
# )
