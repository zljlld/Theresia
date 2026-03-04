import pyaudio
import numpy as np
import soundfile as sf
from funasr import AutoModel
import threading
import queue
import time
from collections import deque

# 音频采集参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
AUDIO_QUEUE = queue.Queue()

# 端点检测参数（可按麦克风和环境调整）
ENERGY_THRESHOLD = 600
START_VOICE_CHUNKS = 2
END_SILENCE_SECONDS = 0.8
MIN_RECORD_SECONDS = 0.3
MAX_RECORD_SECONDS = 15
PRE_ROLL_CHUNKS = 3

END_SILENCE_CHUNKS = max(1, int(END_SILENCE_SECONDS * RATE / CHUNK))
MIN_RECORD_CHUNKS = max(1, int(MIN_RECORD_SECONDS * RATE / CHUNK))
MAX_RECORD_CHUNKS = max(1, int(MAX_RECORD_SECONDS * RATE / CHUNK))

# 初始化 SenseVoice 模型（CPU）
model = AutoModel(
    model="iic/SenseVoiceSmall",
    device="cpu",
    disable_update=True,
    vad_model="fsmn-vad",
    punc_model="ct-punc",
)


def chunk_rms(chunk: np.ndarray) -> float:
    """计算单个音频块的 RMS 能量。"""
    chunk_f = chunk.astype(np.float32)
    return float(np.sqrt(np.mean(chunk_f * chunk_f)))


def audio_capture():
    """采集音频，并把完整语音段放入 AUDIO_QUEUE。"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("麦克风已启动，等待说话...（按 Ctrl+C 退出）")

    pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)
    in_speech = False
    voice_chunks = 0
    silence_chunks = 0
    speech_frames = []

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.int16)
            rms = chunk_rms(audio_np)
            is_speech = rms >= ENERGY_THRESHOLD

            pre_roll.append(audio_np)

            if not in_speech:
                if is_speech:
                    voice_chunks += 1
                else:
                    voice_chunks = 0

                if voice_chunks >= START_VOICE_CHUNKS:
                    in_speech = True
                    silence_chunks = 0
                    speech_frames = list(pre_roll)
                    print("[采集] 检测到开始说话")
            else:
                speech_frames.append(audio_np)

                if is_speech:
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                hit_silence_end = silence_chunks >= END_SILENCE_CHUNKS
                hit_max_len = len(speech_frames) >= MAX_RECORD_CHUNKS

                if hit_silence_end or hit_max_len:
                    if len(speech_frames) >= MIN_RECORD_CHUNKS:
                        utterance = np.concatenate(speech_frames)
                        AUDIO_QUEUE.put(utterance)
                        duration = len(utterance) / RATE
                        print(f"[采集] 检测到说话结束，时长={duration:.2f}秒")

                    in_speech = False
                    voice_chunks = 0
                    silence_chunks = 0
                    speech_frames = []
                    pre_roll.clear()

    except KeyboardInterrupt:
        print("\n正在停止音频采集...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def audio_recognize():
    """每次从 AUDIO_QUEUE 取一段语音进行识别。"""
    while True:
        try:
            audio_data = AUDIO_QUEUE.get(timeout=1)
            audio_float = audio_data.astype(np.float32) / 32768.0

            result = model.generate(
                input=audio_float,
                cache={},
                language="auto",
                use_itn=True,
            )

            if result and len(result) > 0 and "text" in result[0]:
                text = result[0]["text"]
                print(f"\n[识别] {text}")
            else:
                print("\n[识别] 未识别到文本")

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    capture_thread = threading.Thread(target=audio_capture, daemon=True)
    capture_thread.start()

    recognize_thread = threading.Thread(target=audio_recognize, daemon=True)
    recognize_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n实时语音识别已退出")
