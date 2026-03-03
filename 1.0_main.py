import pyaudio
import numpy as np
import soundfile as sf
from funasr import AutoModel
import threading
import queue
import time

# 配置参数（SenseVoice 要求的核心参数）
FORMAT = pyaudio.paInt16  # 16bit 位深
CHANNELS = 1  # 单声道
RATE = 16000  # 16kHz 采样率（必须和模型要求一致）
CHUNK = 1024  # 每次采集的音频块大小
RECORD_SECONDS = 3  # 每3秒识别一次（可调整）
AUDIO_QUEUE = queue.Queue()  # 音频数据队列

# 初始化 SenseVoice 模型（强制CPU运行，避免CUDA报错）
model = AutoModel(
    model="iic/SenseVoiceSmall",
    device="cpu",
    disable_update=True,  # 关闭版本检查
    vad_model="fsmn-vad",  # 开启语音活动检测（过滤静音）
    punc_model="ct-punc",  # 开启标点符号预测
)

def audio_capture():
    """麦克风音频采集线程"""
    p = pyaudio.PyAudio()
    # 打开麦克风流
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    print("✅ 麦克风已开启，开始实时语音识别（按 Ctrl+C 退出）...")
    
    try:
        while True:
            # 读取音频数据
            data = stream.read(CHUNK)
            # 转换为numpy数组（方便后续处理）
            audio_np = np.frombuffer(data, dtype=np.int16)
            AUDIO_QUEUE.put(audio_np)
    except KeyboardInterrupt:
        print("\n🛑 停止采集音频...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def audio_recognize():
    """音频识别线程"""
    audio_buffer = []
    while True:
        try:
            # 从队列获取音频数据
            audio_data = AUDIO_QUEUE.get(timeout=1)
            audio_buffer.append(audio_data)
            
            # 当缓存的音频达到指定时长时，进行识别
            if len(audio_buffer) * CHUNK >= RATE * RECORD_SECONDS:
                # 拼接音频数据
                audio_concat = np.concatenate(audio_buffer)
                # 转换为 SenseVoice 支持的格式（float32，归一化）
                audio_float = audio_concat.astype(np.float32) / 32768.0
                
                # 调用模型识别
                result = model.generate(
                    input=audio_float,
                    cache={},
                    language="auto",  # 自动识别语言（中/英/日/韩/粤/川）
                    use_itn=True,  # 开启文本规范化（比如把"123"转为"一百二十三"）
                )
                
                # 输出识别结果
                if result and len(result) > 0:
                    text = result[0]["text"]
                    print(f"\n🎤 识别结果：{text}")
                
                # 清空缓存（准备下一段音频）
                audio_buffer = []
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    # 启动音频采集线程
    capture_thread = threading.Thread(target=audio_capture)
    capture_thread.daemon = True
    capture_thread.start()
    
    # 启动音频识别线程
    recognize_thread = threading.Thread(target=audio_recognize)
    recognize_thread.daemon = True
    recognize_thread.start()
    
    # 主线程等待退出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 实时语音识别已退出")