import json
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyaudio
from funasr import AutoModel

# 音频采集参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# 端点检测默认参数（可在 Web/GUI 中动态调整）
DEFAULT_ENERGY_THRESHOLD = 1330
MIN_ENERGY_THRESHOLD = 100
MAX_ENERGY_THRESHOLD = 5000
DEFAULT_RECOGNITION_LANGUAGE = "zh"
DEFAULT_UNKNOWN_REJECT_THRESHOLD = 0.98
DEFAULT_ENABLE_SPEAKER_COMPARE = True
START_VOICE_CHUNKS = 2
END_SILENCE_SECONDS = 0.4
MIN_RECORD_SECONDS = 0.3
MAX_RECORD_SECONDS = 15
PRE_ROLL_CHUNKS = 3

# 声纹参数
VOICEPRINT_DB_PATH = Path(__file__).with_name("voiceprints.json")
SETTINGS_PATH = Path(__file__).with_name("asr_settings.json")
SPEAKER_SIM_THRESHOLD = 0.80

END_SILENCE_CHUNKS = max(1, int(END_SILENCE_SECONDS * RATE / CHUNK))
MIN_RECORD_CHUNKS = max(1, int(MIN_RECORD_SECONDS * RATE / CHUNK))
MAX_RECORD_CHUNKS = max(1, int(MAX_RECORD_SECONDS * RATE / CHUNK))

TOKEN_RE = re.compile(r"<\s*\|([^<>]*?)\|\s*>")
TEXT_TOKEN_RE = re.compile(r"<\s*\|.*?\|\s*>")
EMOTION_TOKEN_MAP = {
    "HAPPY": "开心",
    "SAD": "悲伤",
    "ANGRY": "愤怒",
    "NEUTRAL": "平静",
    "FEAR": "恐惧",
    "SURPRISE": "惊讶",
    "DISGUST": "厌恶",
}

LANGUAGE_ALIAS_MAP = {
    "中文": "zh",
    "汉语": "zh",
    "汉语普通话": "zh",
    "普通话": "zh",
    "chinese": "zh",
    "zh-cn": "zh",
    "zh_cn": "zh",
    "cn": "zh",
    "zh": "zh",
    "日文": "ja",
    "日语": "ja",
    "日本語": "ja",
    "japanese": "ja",
    "jp": "ja",
    "ja": "ja",
    "自动": "auto",
    "auto": "auto",
}


def chunk_rms(chunk: np.ndarray) -> float:
    """计算单个音频块的 RMS 能量。"""
    chunk_f = chunk.astype(np.float32)
    return float(np.sqrt(np.mean(chunk_f * chunk_f)))


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """向量 L2 归一化。"""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def preprocess_audio_for_recognition(audio_float: np.ndarray) -> np.ndarray:
    """识别前处理：去直流 + 峰值归一化。"""
    x = np.asarray(audio_float, dtype=np.float32)
    if x.size == 0:
        return x

    # 去直流分量
    x = x - float(np.mean(x))

    # 峰值归一化（保留一定余量，避免削顶）
    peak = float(np.max(np.abs(x)))
    if peak > 1e-6:
        x = 0.95 * x / peak

    return np.clip(x, -1.0, 1.0)


def extract_voice_embedding(audio_float: np.ndarray) -> Optional[np.ndarray]:
    """提取简化声纹向量（log 频谱均值）。"""
    if audio_float is None or len(audio_float) < int(0.3 * RATE):
        return None

    x = audio_float.astype(np.float32)
    x = x - np.mean(x)

    frame_len = int(0.025 * RATE)
    hop_len = int(0.010 * RATE)
    n_fft = 512
    keep_bins = 128

    if len(x) < frame_len:
        return None

    window = np.hamming(frame_len).astype(np.float32)
    feats = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        frame = x[start : start + frame_len] * window
        spec = np.abs(np.fft.rfft(frame, n=n_fft))
        spec = np.log1p(spec[1 : keep_bins + 1])
        feats.append(spec)

    if not feats:
        return None

    emb = np.mean(np.stack(feats, axis=0), axis=0)
    return l2_normalize(emb.astype(np.float32))


def parse_text_and_emotion(raw_text: str) -> tuple[str, str]:
    """从识别结果中提取纯文本与情绪。"""
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return "", "未知"

    tokens_raw = TOKEN_RE.findall(raw_text)
    tokens = [re.sub(r"[^A-Za-z0-9_]+", "", t).upper() for t in tokens_raw]
    emotion = "未知"
    for t in tokens:
        if t in EMOTION_TOKEN_MAP:
            emotion = EMOTION_TOKEN_MAP[t]
            break

    clean_text = TEXT_TOKEN_RE.sub("", raw_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text, emotion


def normalize_language_value(value: str) -> str:
    """规范化语种输入，避免把“中文”等标签误传给模型。"""
    raw = str(value or "").strip()
    if not raw:
        return DEFAULT_RECOGNITION_LANGUAGE
    lowered = raw.lower()
    if raw in LANGUAGE_ALIAS_MAP:
        return LANGUAGE_ALIAS_MAP[raw]
    if lowered in LANGUAGE_ALIAS_MAP:
        return LANGUAGE_ALIAS_MAP[lowered]
    return lowered


def load_voiceprint_db() -> dict[str, list[list[float]]]:
    """加载声纹库。"""
    if not VOICEPRINT_DB_PATH.exists():
        return {}
    try:
        with VOICEPRINT_DB_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_voiceprint_db(db: dict[str, list[list[float]]]) -> None:
    """保存声纹库。"""
    with VOICEPRINT_DB_PATH.open("w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def load_saved_threshold(default_value: int) -> int:
    """从本地配置读取上次保存的阈值。"""
    if not SETTINGS_PATH.exists():
        return int(default_value)
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        value = int(data.get("energy_threshold", default_value))
        return max(MIN_ENERGY_THRESHOLD, min(MAX_ENERGY_THRESHOLD, value))
    except Exception:
        return int(default_value)


def load_saved_language(default_value: str = DEFAULT_RECOGNITION_LANGUAGE) -> str:
    """从本地配置读取上次保存的识别语种。"""
    if not SETTINGS_PATH.exists():
        return default_value
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        value = normalize_language_value(str(data.get("recognition_language", default_value)))
        if value:
            return value
    except Exception:
        pass
    return default_value


def load_saved_unknown_reject_threshold(default_value: float = DEFAULT_UNKNOWN_REJECT_THRESHOLD) -> float:
    """从本地配置读取陌生人拒识阈值（0-1）。"""
    if not SETTINGS_PATH.exists():
        return float(default_value)
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        value = float(data.get("unknown_reject_threshold", default_value))
        return max(0.0, min(1.0, value))
    except Exception:
        return float(default_value)


def load_saved_enable_speaker_compare(default_value: bool = DEFAULT_ENABLE_SPEAKER_COMPARE) -> bool:
    """Load whether speaker comparison is enabled from local settings."""
    if not SETTINGS_PATH.exists():
        return bool(default_value)
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        value = data.get("enable_speaker_compare", default_value)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    except Exception:
        pass
    return bool(default_value)


def save_threshold(value: int) -> None:
    """Persist energy threshold and keep other settings."""
    safe_value = max(MIN_ENERGY_THRESHOLD, min(MAX_ENERGY_THRESHOLD, int(value)))
    payload: dict[str, object] = {"energy_threshold": safe_value}

    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                old_data = json.load(f)
            if isinstance(old_data, dict) and "recognition_language" in old_data:
                payload["recognition_language"] = str(old_data["recognition_language"])
            if isinstance(old_data, dict) and "unknown_reject_threshold" in old_data:
                payload["unknown_reject_threshold"] = float(old_data["unknown_reject_threshold"])
            if isinstance(old_data, dict) and "enable_speaker_compare" in old_data:
                payload["enable_speaker_compare"] = bool(old_data["enable_speaker_compare"])
        except Exception:
            pass

    if "recognition_language" not in payload:
        payload["recognition_language"] = DEFAULT_RECOGNITION_LANGUAGE
    if "unknown_reject_threshold" not in payload:
        payload["unknown_reject_threshold"] = DEFAULT_UNKNOWN_REJECT_THRESHOLD
    if "enable_speaker_compare" not in payload:
        payload["enable_speaker_compare"] = DEFAULT_ENABLE_SPEAKER_COMPARE

    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_language(language: str) -> None:
    """Persist recognition language and keep other settings."""
    lang = normalize_language_value(language)
    payload: dict[str, object] = {"recognition_language": lang}

    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                old_data = json.load(f)
            if isinstance(old_data, dict) and "energy_threshold" in old_data:
                payload["energy_threshold"] = int(old_data["energy_threshold"])
            if isinstance(old_data, dict) and "unknown_reject_threshold" in old_data:
                payload["unknown_reject_threshold"] = float(old_data["unknown_reject_threshold"])
            if isinstance(old_data, dict) and "enable_speaker_compare" in old_data:
                payload["enable_speaker_compare"] = bool(old_data["enable_speaker_compare"])
        except Exception:
            pass

    if "energy_threshold" not in payload:
        payload["energy_threshold"] = DEFAULT_ENERGY_THRESHOLD
    if "unknown_reject_threshold" not in payload:
        payload["unknown_reject_threshold"] = DEFAULT_UNKNOWN_REJECT_THRESHOLD
    if "enable_speaker_compare" not in payload:
        payload["enable_speaker_compare"] = DEFAULT_ENABLE_SPEAKER_COMPARE

    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_unknown_reject_threshold(value: float) -> None:
    """Persist unknown-speaker reject threshold (0-1) and keep other settings."""
    threshold = max(0.0, min(1.0, float(value)))
    payload: dict[str, object] = {"unknown_reject_threshold": threshold}

    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                old_data = json.load(f)
            if isinstance(old_data, dict) and "energy_threshold" in old_data:
                payload["energy_threshold"] = int(old_data["energy_threshold"])
            if isinstance(old_data, dict) and "recognition_language" in old_data:
                payload["recognition_language"] = str(old_data["recognition_language"])
            if isinstance(old_data, dict) and "enable_speaker_compare" in old_data:
                payload["enable_speaker_compare"] = bool(old_data["enable_speaker_compare"])
        except Exception:
            pass

    if "energy_threshold" not in payload:
        payload["energy_threshold"] = DEFAULT_ENERGY_THRESHOLD
    if "recognition_language" not in payload:
        payload["recognition_language"] = DEFAULT_RECOGNITION_LANGUAGE
    if "enable_speaker_compare" not in payload:
        payload["enable_speaker_compare"] = DEFAULT_ENABLE_SPEAKER_COMPARE

    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_enable_speaker_compare(enabled: bool) -> None:
    """Persist whether speaker comparison is enabled and keep other settings."""
    flag = bool(enabled)
    payload: dict[str, object] = {"enable_speaker_compare": flag}

    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                old_data = json.load(f)
            if isinstance(old_data, dict) and "energy_threshold" in old_data:
                payload["energy_threshold"] = int(old_data["energy_threshold"])
            if isinstance(old_data, dict) and "recognition_language" in old_data:
                payload["recognition_language"] = str(old_data["recognition_language"])
            if isinstance(old_data, dict) and "unknown_reject_threshold" in old_data:
                payload["unknown_reject_threshold"] = float(old_data["unknown_reject_threshold"])
        except Exception:
            pass

    if "energy_threshold" not in payload:
        payload["energy_threshold"] = DEFAULT_ENERGY_THRESHOLD
    if "recognition_language" not in payload:
        payload["recognition_language"] = DEFAULT_RECOGNITION_LANGUAGE
    if "unknown_reject_threshold" not in payload:
        payload["unknown_reject_threshold"] = DEFAULT_UNKNOWN_REJECT_THRESHOLD

    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def upsert_voiceprint(db: dict[str, list[list[float]]], speaker_name: str, embedding: np.ndarray) -> None:
    """写入一条注册样本。"""
    if speaker_name not in db:
        db[speaker_name] = []
    db[speaker_name].append(embedding.tolist())


def speaker_centroid(samples: list[list[float]]) -> Optional[np.ndarray]:
    """计算说话人中心向量。"""
    if not samples:
        return None
    arr = np.asarray(samples, dtype=np.float32)
    center = np.mean(arr, axis=0)
    return l2_normalize(center)


def identify_speaker(
    db: dict[str, list[list[float]]], embedding: Optional[np.ndarray], threshold: float = SPEAKER_SIM_THRESHOLD
) -> tuple[str, float]:
    """返回识别到的说话人和相似度。"""
    if embedding is None or not db:
        return "陌生人", 0.0

    best_name = "陌生人"
    best_score = -1.0

    for name, samples in db.items():
        center = speaker_centroid(samples)
        if center is None:
            continue
        score = float(np.dot(embedding, center))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return "陌生人", best_score
    return best_name, best_score


def next_default_name(db: dict[str, list[list[float]]]) -> str:
    """生成默认注册名。"""
    i = 1
    while True:
        candidate = f"用户{i}"
        if candidate not in db:
            return candidate
        i += 1


class RealtimeASRService:
    def __init__(
        self,
        on_log: Optional[Callable[[str], None]] = None,
        on_result: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.on_log = on_log
        self.on_result = on_result

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stop_event = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.recognize_thread: Optional[threading.Thread] = None

        self.model: Optional[AutoModel] = None
        self.db = load_voiceprint_db()

        self.energy_threshold = load_saved_threshold(DEFAULT_ENERGY_THRESHOLD)
        self.recognition_language = load_saved_language(DEFAULT_RECOGNITION_LANGUAGE)
        self.unknown_reject_threshold = load_saved_unknown_reject_threshold(DEFAULT_UNKNOWN_REJECT_THRESHOLD)
        self.enable_speaker_compare = load_saved_enable_speaker_compare(DEFAULT_ENABLE_SPEAKER_COMPARE)
        self.config_lock = threading.Lock()

        # 手动注册录音状态
        self.manual_reg_lock = threading.Lock()
        self.manual_reg_active = False
        self.manual_reg_name = ""
        self.manual_reg_frames: list[np.ndarray] = []

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = AutoModel(
                model="iic/SenseVoiceSmall",
                device="cpu",
                disable_update=True,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
            )

    def _log(self, message: str) -> None:
        if self.on_log:
            self.on_log(message)

    def _emit_result(self, speaker: str, score: float, text: str, emotion: str) -> None:
        if self.on_result:
            self.on_result({"speaker": speaker, "score": score, "text": text, "emotion": emotion})

    def is_running(self) -> bool:
        return self.capture_thread is not None and self.capture_thread.is_alive()

    def set_energy_threshold(self, value: int) -> int:
        value = int(value)
        value = max(MIN_ENERGY_THRESHOLD, min(MAX_ENERGY_THRESHOLD, value))
        with self.config_lock:
            self.energy_threshold = value
        # 每次阈值调整后立即持久化，确保下次启动沿用最新值。
        save_threshold(value)
        self._log(
            f"[配置] 开始说话阈值已设置为 {value} "
            f"(允许范围: {MIN_ENERGY_THRESHOLD}-{MAX_ENERGY_THRESHOLD})"
        )
        return value

    def get_energy_threshold(self) -> int:
        with self.config_lock:
            return int(self.energy_threshold)

    def set_unknown_reject_threshold_percent(self, threshold_percent: float) -> float:
        """设置陌生人拒识阈值（百分比，0-100）。"""
        percent = max(0.0, min(100.0, float(threshold_percent)))
        ratio = percent / 100.0
        with self.config_lock:
            self.unknown_reject_threshold = ratio
        save_unknown_reject_threshold(ratio)
        self._log(f"[配置] 陌生人拒识阈值已设置为 {percent:.2f}%")
        return percent

    def get_unknown_reject_threshold(self) -> float:
        with self.config_lock:
            return float(self.unknown_reject_threshold)

    def get_unknown_reject_threshold_percent(self) -> float:
        return self.get_unknown_reject_threshold() * 100.0

    def set_enable_speaker_compare(self, enabled: bool) -> bool:
        """Set whether speaker comparison is enabled."""
        flag = bool(enabled)
        with self.config_lock:
            self.enable_speaker_compare = flag
        save_enable_speaker_compare(flag)
        self._log(f"[config] speaker comparison {'enabled' if flag else 'disabled'}")
        return flag

    def get_enable_speaker_compare(self) -> bool:
        with self.config_lock:
            return bool(self.enable_speaker_compare)

    def set_recognition_language(self, language: str) -> str:
        """设置识别语种。"""
        lang = normalize_language_value(language)
        with self.config_lock:
            self.recognition_language = lang
        save_language(lang)
        self._log(f"[配置] 识别语种已设置为 {lang}")
        return lang

    def get_recognition_language(self) -> str:
        with self.config_lock:
            return str(self.recognition_language)

    def get_voiceprint_summary(self) -> list[dict[str, int | str]]:
        return [{"name": n, "samples": len(s)} for n, s in sorted(self.db.items(), key=lambda x: x[0])]

    def delete_voiceprint(self, speaker_name: str) -> bool:
        speaker_name = speaker_name.strip()
        if not speaker_name or speaker_name not in self.db:
            return False
        del self.db[speaker_name]
        save_voiceprint_db(self.db)
        self._log(f"[声纹] 已删除：{speaker_name}")
        return True

    def start_manual_registration(self, speaker_name: str = "") -> tuple[bool, str]:
        """开始手动注册录音。"""
        if not self.is_running():
            return False, "服务未运行，无法开始注册录音。"

        with self.manual_reg_lock:
            if self.manual_reg_active:
                return False, f"已在录音中：{self.manual_reg_name}"
            if not speaker_name.strip():
                speaker_name = next_default_name(self.db)
            self.manual_reg_name = speaker_name.strip()
            self.manual_reg_frames = []
            self.manual_reg_active = True

        self._log(f"[声纹] 开始录音注册：{self.manual_reg_name}（点击“结束录音”完成）")
        return True, f"录音中：{self.manual_reg_name}"

    def stop_manual_registration(self) -> tuple[bool, str]:
        """结束手动注册录音并写入声纹库。"""
        with self.manual_reg_lock:
            if not self.manual_reg_active:
                return False, "当前没有进行中的注册录音。"
            speaker_name = self.manual_reg_name
            frames = self.manual_reg_frames[:]
            self.manual_reg_active = False
            self.manual_reg_name = ""
            self.manual_reg_frames = []

        if not frames:
            msg = "注册失败：未录到有效音频。"
            self._log(f"[声纹] {msg}")
            return False, msg

        audio_data = np.concatenate(frames).astype(np.int16)
        duration = len(audio_data) / RATE
        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_float = preprocess_audio_for_recognition(audio_float)
        embedding = extract_voice_embedding(audio_float)
        if embedding is None:
            msg = "注册失败：音频过短或音质不足。"
            self._log(f"[声纹] {msg}（目标：{speaker_name}）")
            return False, msg

        upsert_voiceprint(self.db, speaker_name, embedding)
        save_voiceprint_db(self.db)
        sample_count = len(self.db[speaker_name])
        msg = f"注册完成：{speaker_name}（样本数={sample_count}，时长={duration:.2f}秒）"
        self._log(f"[声纹] {msg}")
        return True, msg

    def start(self) -> None:
        if self.is_running():
            self._log("[系统] 识别服务已经在运行")
            return

        self._ensure_model()
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.recognize_thread = threading.Thread(target=self._audio_recognize_loop, daemon=True)
        self.capture_thread.start()
        self.recognize_thread.start()

        if self.db:
            self._log(f"[声纹] 已加载 {len(self.db)} 位注册说话人：{', '.join(self.db.keys())}")
        else:
            self._log("[声纹] 当前没有已注册说话人")
        self._log("[系统] 服务已启动")

    def stop(self) -> None:
        if not self.is_running():
            save_threshold(self.get_energy_threshold())
            self._log("[系统] 服务未在运行")
            return

        with self.manual_reg_lock:
            self.manual_reg_active = False
            self.manual_reg_name = ""
            self.manual_reg_frames = []

        self.stop_event.set()
        for _ in range(5):
            try:
                self.audio_queue.put_nowait(np.zeros(CHUNK, dtype=np.int16))
            except queue.Full:
                pass

        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.recognize_thread:
            self.recognize_thread.join(timeout=2)

        self.capture_thread = None
        self.recognize_thread = None
        save_threshold(self.get_energy_threshold())
        self._log("[系统] 服务已停止")

    def reload_voiceprints(self) -> list[str]:
        self.db = load_voiceprint_db()
        return list(self.db.keys())

    def _audio_capture_loop(self) -> None:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        self._log("[采集] 麦克风已启动，等待说话...")

        pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)
        in_speech = False
        voice_chunks = 0
        silence_chunks = 0
        speech_frames: list[np.ndarray] = []

        try:
            while not self.stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16)

                # 手动注册录音期间，直接缓存整段音频，不走自动分段逻辑。
                with self.manual_reg_lock:
                    if self.manual_reg_active:
                        self.manual_reg_frames.append(audio_np.copy())
                        continue

                with self.config_lock:
                    current_threshold = self.energy_threshold
                is_speech = chunk_rms(audio_np) >= current_threshold
                pre_roll.append(audio_np)

                if not in_speech:
                    voice_chunks = voice_chunks + 1 if is_speech else 0
                    if voice_chunks >= START_VOICE_CHUNKS:
                        in_speech = True
                        silence_chunks = 0
                        speech_frames = list(pre_roll)
                        self._log("[采集] 检测到开始说话")
                else:
                    speech_frames.append(audio_np)
                    silence_chunks = 0 if is_speech else silence_chunks + 1

                    hit_silence_end = silence_chunks >= END_SILENCE_CHUNKS
                    hit_max_len = len(speech_frames) >= MAX_RECORD_CHUNKS
                    if hit_silence_end or hit_max_len:
                        if len(speech_frames) >= MIN_RECORD_CHUNKS:
                            utterance = np.concatenate(speech_frames)
                            self.audio_queue.put(utterance)
                            duration = len(utterance) / RATE
                            self._log(f"[采集] 检测到说话结束，时长={duration:.2f}秒")

                        in_speech = False
                        voice_chunks = 0
                        silence_chunks = 0
                        speech_frames = []
                        pre_roll.clear()

        except Exception as exc:
            self._log(f"[采集] 发生错误：{exc}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self._log("[采集] 音频采集线程已退出")

    def _audio_recognize_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.stop_event.is_set():
                break

            try:
                audio_float = audio_data.astype(np.float32) / 32768.0
                audio_float = preprocess_audio_for_recognition(audio_float)
                current_language = normalize_language_value(self.get_recognition_language())
                result = self.model.generate(
                    input=audio_float,
                    cache={},
                    language=current_language,
                    use_itn=True,
                )

                raw_text = ""
                if result and len(result) > 0 and "text" in result[0]:
                    raw_text = str(result[0]["text"]).strip()

                clean_text, emotion = parse_text_and_emotion(raw_text)
                if self.get_enable_speaker_compare():
                    embedding = extract_voice_embedding(audio_float)
                    speaker_name, score = identify_speaker(
                        self.db,
                        embedding,
                        threshold=self.get_unknown_reject_threshold(),
                    )
                else:
                    speaker_name, score = "\u672a\u542f\u7528", 0.0

                # 仅在识别出有效文本时才更新前端展示区域。
                if clean_text:
                    self._emit_result(speaker_name, score, clean_text, emotion)

            except Exception as exc:
                self._log(f"[识别] 发生错误：{exc}")

        self._log("[识别] 识别线程已退出")


def run_cli() -> None:
    def on_log(msg: str) -> None:
        print(msg)

    def on_result(item: dict) -> None:
        print(f"[识别][{item['speaker']}|{item['score']:.2f}|{item.get('emotion', '未知')}] {item['text']}")

    service = RealtimeASRService(on_log=on_log, on_result=on_result)
    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        print("实时语音识别已退出")


if __name__ == "__main__":
    run_cli()
