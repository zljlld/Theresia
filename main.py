from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 模型目录：可以是模型名称（如官方提供的iic/SenseVoiceSmall）或本地磁盘中的模型路径
model_dir = "iic/SenseVoiceSmall"

# 加载SenseVoiceSmall语音识别模型
model = AutoModel(
    model=model_dir,  # 模型名称/路径
    # trust_remote_code：是否信任远程代码
    # True：从remote_code指定的路径加载模型代码（支持绝对/相对路径、网络url）
    # False：使用FunASR内部集成的模型代码（修改本地model.py不生效）
    trust_remote_code=True,
    remote_code="./model.py",  # 模型代码的具体位置
    vad_model="fsmn-vad",  # 开启VAD（语音活动检测），用于将长音频切割成短音频
    # VAD模型配置参数
    # max_single_segment_time：VAD最大切割音频时长，单位为毫秒(ms)
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",  # 运行设备，cuda:0表示使用第0块GPU，也可设为"cpu"
)

# 对英文音频进行语音识别
res = model.generate(
    input=f"C:\\Users\\zljlld\\Desktop\\Theresia\\ASR_test\\录音.mp3",  # 输入音频文件路径
    cache={},  # 缓存字典（暂未使用）
    # 识别语言：可选auto(自动识别)、zh(中文)、en(英文)、yue(粤语)、ja(日语)、ko(韩语)、nospeech(无语音)
    language="auto",
    use_itn=True,  # 是否开启逆文本正则化（输出结果包含标点）
    batch_size_s=60,  # 动态batch的总音频时长，单位为秒(s)
    merge_vad=True,  # 是否将VAD切割的短音频碎片合并
    merge_length_s=15,  # 合并后音频片段的长度，单位为秒(s)
    # ban_emo_unk=False,  # 可选：是否禁用emo_unk标签，默认False（禁用后所有句子都会赋予情感标签）
)

# 对识别结果进行后处理，优化输出格式
text = rich_transcription_postprocess(res[0]["text"])
# 打印最终的识别文本
print(text)