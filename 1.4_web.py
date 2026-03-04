import importlib.util
import queue
import threading
import time
from pathlib import Path
from typing import Any

import gradio as gr


BACKEND_PATH = Path(__file__).with_name("1.4_main.py")
MAX_LOG_LINES = 300
MAX_TEXT_HISTORY_LINES = 120
THRESHOLD_MIN = 100
THRESHOLD_MAX = 5000
REJECT_THRESHOLD_MIN = 0.0
REJECT_THRESHOLD_MAX = 100.0


def load_backend_class():
    spec = importlib.util.spec_from_file_location("asr_backend_13", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载后端文件: {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RealtimeASRService


RealtimeASRService = load_backend_class()


class WebAppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "未启动"
        self.current_speaker = "无"
        self.last_text = ""
        self.text_history: list[str] = []
        self.current_emotion = "未知"
        self.register_state = "未录音"
        self.register_started_at: float | None = None
        self.log_lines: list[str] = []
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.service = RealtimeASRService(
            on_log=self._on_backend_log,
            on_result=self._on_backend_result,
        )

    def _on_backend_log(self, message: str) -> None:
        self.events.put(("log", message))

    def _on_backend_result(self, payload: dict) -> None:
        self.events.put(("result", payload))

    def append_log(self, text: str) -> None:
        with self.lock:
            self.log_lines.append(text)
            if len(self.log_lines) > MAX_LOG_LINES:
                self.log_lines = self.log_lines[-MAX_LOG_LINES:]

    def set_register_state(self, text: str) -> None:
        with self.lock:
            self.register_state = text

    def start_register_timer(self) -> None:
        with self.lock:
            self.register_started_at = time.time()

    def stop_register_timer(self) -> None:
        with self.lock:
            self.register_started_at = None

    def register_elapsed_text(self) -> str:
        with self.lock:
            if self.register_started_at is None:
                return "00:00"
            elapsed = max(0, int(time.time() - self.register_started_at))
        mm = elapsed // 60
        ss = elapsed % 60
        return f"{mm:02d}:{ss:02d}"

    def drain_events(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self.append_log(str(payload))
            elif kind == "result":
                speaker = str(payload.get("speaker", "陌生人"))
                score = float(payload.get("score", 0.0))
                text = str(payload.get("text", ""))
                emotion = str(payload.get("emotion", "未知"))
                with self.lock:
                    self.current_speaker = f"{speaker}（相似度 {score:.2f}）"
                    self.last_text = text
                    self.current_emotion = emotion
                    if text:
                        self.text_history.append(f"{speaker}：[{emotion}]{text}")
                        if len(self.text_history) > MAX_TEXT_HISTORY_LINES:
                            self.text_history = self.text_history[-MAX_TEXT_HISTORY_LINES:]
                self.append_log(f"[识别][{speaker}|{score:.2f}|{emotion}] {text}")

    def ui_snapshot(self):
        self.drain_events()
        with self.lock:
            return (
                f"状态：{self.status}",
                self.current_speaker,
                "\n".join(self.text_history),
                self.current_emotion,
                self.register_state,
                "\n".join(self.log_lines),
            )


APP = WebAppState()


def voiceprint_rows():
    rows = APP.service.get_voiceprint_summary()
    if not rows:
        return [], []
    table = [[r["name"], r["samples"]] for r in rows]
    choices = [str(r["name"]) for r in rows]
    return table, choices


def normalize_threshold(value: Any) -> tuple[int, str | None]:
    try:
        ivalue = int(float(value))
    except Exception:
        return APP.service.get_energy_threshold(), "阈值输入无效，已恢复到当前值。"

    if ivalue < THRESHOLD_MIN:
        return THRESHOLD_MIN, f"阈值过小，已自动调整为 {THRESHOLD_MIN}。"
    if ivalue > THRESHOLD_MAX:
        return THRESHOLD_MAX, f"阈值过大，已自动调整为 {THRESHOLD_MAX}。"
    return ivalue, None


def normalize_reject_threshold(value: Any) -> tuple[float, str | None]:
    try:
        fvalue = float(value)
    except Exception:
        return APP.service.get_unknown_reject_threshold_percent(), "拒识阈值输入无效，已恢复到当前值。"

    if fvalue < REJECT_THRESHOLD_MIN:
        return REJECT_THRESHOLD_MIN, f"拒识阈值过小，已自动调整为 {REJECT_THRESHOLD_MIN:.0f}。"
    if fvalue > REJECT_THRESHOLD_MAX:
        return REJECT_THRESHOLD_MAX, f"拒识阈值过大，已自动调整为 {REJECT_THRESHOLD_MAX:.0f}。"
    return fvalue, None


def start_service():
    APP.service.start()
    with APP.lock:
        APP.status = "运行中"
    APP.append_log("[网页] 已点击启动识别")
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    table, choices = voiceprint_rows()
    return (
        status,
        speaker,
        text,
        emotion,
        reg_state,
        APP.register_elapsed_text(),
        logs,
        table,
        gr.Dropdown(choices=choices, value=None),
    )


def stop_service():
    APP.service.stop()
    with APP.lock:
        APP.status = "已停止"
    APP.set_register_state("未录音")
    APP.stop_register_timer()
    APP.append_log("[网页] 已点击停止识别")
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    table, choices = voiceprint_rows()
    return (
        status,
        speaker,
        text,
        emotion,
        reg_state,
        APP.register_elapsed_text(),
        logs,
        table,
        gr.Dropdown(choices=choices, value=None),
    )


def apply_threshold(value: int):
    safe_value, msg = normalize_threshold(value)
    try:
        final_value = APP.service.set_energy_threshold(safe_value)
    except Exception as exc:
        final_value = APP.service.get_energy_threshold()
        APP.append_log(f"[网页] 阈值更新失败：{exc}")
        status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
        return final_value, status, speaker, text, emotion, reg_state, logs

    if msg:
        APP.append_log(f"[网页] {msg}")
    APP.append_log(f"[网页] 已应用开始说话阈值：{final_value}")
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    return final_value, status, speaker, text, emotion, reg_state, logs


def apply_reject_threshold(value: float):
    safe_value, msg = normalize_reject_threshold(value)
    try:
        final_value = APP.service.set_unknown_reject_threshold_percent(safe_value)
    except Exception as exc:
        final_value = APP.service.get_unknown_reject_threshold_percent()
        APP.append_log(f"[网页] 拒识阈值更新失败：{exc}")
        status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
        return final_value, status, speaker, text, emotion, reg_state, logs

    if msg:
        APP.append_log(f"[网页] {msg}")
    APP.append_log(f"[网页] 已应用拒识阈值：{final_value:.2f}%")
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    return final_value, status, speaker, text, emotion, reg_state, logs


def apply_language(language: str):
    final_lang = APP.service.set_recognition_language(language)
    APP.append_log(f"[网页] 已切换识别语种：{final_lang}")
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    return final_lang, status, speaker, text, emotion, reg_state, logs


def start_register_recording(name: str):
    ok, msg = APP.service.start_manual_registration((name or "").strip())
    APP.append_log(f"[网页] {msg}")
    if ok:
        APP.set_register_state(msg)
        APP.start_register_timer()
    else:
        APP.stop_register_timer()
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    return "", reg_state, APP.register_elapsed_text(), status, speaker, text, emotion, logs


def stop_register_recording():
    ok, msg = APP.service.stop_manual_registration()
    APP.append_log(f"[网页] {msg}")
    APP.set_register_state("未录音")
    APP.stop_register_timer()
    table, choices = voiceprint_rows()
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    return (
        reg_state,
        APP.register_elapsed_text(),
        status,
        speaker,
        text,
        emotion,
        logs,
        table,
        gr.Dropdown(choices=choices, value=None),
    )


def refresh_voiceprints():
    APP.service.reload_voiceprints()
    table, choices = voiceprint_rows()
    APP.append_log("[网页] 已刷新声纹列表")
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    return table, gr.Dropdown(choices=choices, value=None), status, speaker, text, emotion, reg_state, logs


def delete_voiceprint(name: str):
    if not name:
        APP.append_log("[网页] 删除失败：未选择说话人")
    else:
        ok = APP.service.delete_voiceprint(name)
        if ok:
            APP.append_log(f"[网页] 已删除声纹：{name}")
        else:
            APP.append_log(f"[网页] 删除失败：未找到 {name}")

    table, choices = voiceprint_rows()
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    return table, gr.Dropdown(choices=choices, value=None), status, speaker, text, emotion, reg_state, logs


def poll_updates():
    status, speaker, text, emotion, reg_state, logs = APP.ui_snapshot()
    table, choices = voiceprint_rows()
    return (
        status,
        speaker,
        text,
        emotion,
        reg_state,
        APP.register_elapsed_text(),
        logs,
        table,
        gr.Dropdown(choices=choices),
    )


custom_css = """
:root {
  --bg: #f7f9f9;
  --card: #ffffff;
  --text: #0f1419;
  --muted: #536471;
  --border: #e6ecf0;
  --brand: #1d9bf0;
  --brand-hover: #1a8cd8;
}
body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}
#app-shell { max-width: 1100px; margin: 0 auto; }
#topbar {
  background: rgba(255,255,255,0.85);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px 10px 16px;
  margin-bottom: 14px;
}
.x-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; }
.gr-button-primary { background: var(--brand) !important; border-color: var(--brand) !important; }
.gr-button-primary:hover { background: var(--brand-hover) !important; border-color: var(--brand-hover) !important; }
"""


with gr.Blocks(title="ASR 声纹管理网页", css=custom_css) as demo:
    with gr.Column(elem_id="app-shell"):
        with gr.Column(elem_id="topbar"):
            gr.Markdown("### ASR 声纹管理中心")
            gr.Markdown("实时识别、阈值调参与声纹管理")
            with gr.Row():
                start_btn = gr.Button("启动识别", variant="primary")
                stop_btn = gr.Button("停止识别")

        with gr.Row():
            status_box = gr.Textbox(label="运行状态", interactive=False, value="状态：未启动")
            speaker_box = gr.Textbox(label="当前说话人", interactive=False, value="无")
            emotion_box = gr.Textbox(label="当前情绪", interactive=False, value="未知")

        text_box = gr.Textbox(label="最近识别文本（含历史）", interactive=False, lines=8, value="")

        with gr.Tabs():
            with gr.Tab("阈值设置"):
                with gr.Group(elem_classes=["x-card"]):
                    gr.Markdown("#### 开始说话检测阈值")
                    gr.Markdown("阈值越小越敏感，越大越不容易触发。建议在 `300-1500` 区间调试。")
                    language_selector = gr.Dropdown(
                        choices=[
                            ("中文", "zh"),
                            ("日文", "ja"),
                            ("英文", "en"),
                            ("自动", "auto"),
                            ("粤语", "yue"),
                        ],
                        value=APP.service.get_recognition_language(),
                        label="识别语种",
                        allow_custom_value=True,
                    )
                    threshold = gr.Slider(
                        minimum=THRESHOLD_MIN,
                        maximum=THRESHOLD_MAX,
                        step=1,
                        value=APP.service.get_energy_threshold(),
                        label="阈值",
                    )
                    reject_threshold = gr.Slider(
                        minimum=REJECT_THRESHOLD_MIN,
                        maximum=REJECT_THRESHOLD_MAX,
                        step=0.1,
                        value=APP.service.get_unknown_reject_threshold_percent(),
                        label="陌生人拒识阈值（%）",
                    )

            with gr.Tab("声纹管理"):
                with gr.Group(elem_classes=["x-card"]):
                    register_name = gr.Textbox(label="注册名称（留空自动命名）", placeholder="例如：张三")
                    reg_state = gr.Textbox(label="注册录音状态", interactive=False, value="未录音")
                    reg_timer = gr.Textbox(label="录音时长", interactive=False, value="00:00")
                    with gr.Row():
                        start_reg_btn = gr.Button("开始录音", variant="primary")
                        stop_reg_btn = gr.Button("结束录音")
                    with gr.Row():
                        refresh_btn = gr.Button("刷新列表")
                        delete_target = gr.Dropdown(label="删除目标", choices=[], value=None)
                        delete_btn = gr.Button("删除选中")
                    voiceprint_table = gr.Dataframe(
                        headers=["说话人", "样本数"],
                        datatype=["str", "number"],
                        value=[],
                        wrap=True,
                    )

            with gr.Tab("实时日志"):
                with gr.Group(elem_classes=["x-card"]):
                    log_box = gr.Textbox(label="日志", lines=20, interactive=False)

    start_btn.click(
        fn=start_service,
        outputs=[
            status_box,
            speaker_box,
            text_box,
            emotion_box,
            reg_state,
            reg_timer,
            log_box,
            voiceprint_table,
            delete_target,
        ],
    )
    stop_btn.click(
        fn=stop_service,
        outputs=[
            status_box,
            speaker_box,
            text_box,
            emotion_box,
            reg_state,
            reg_timer,
            log_box,
            voiceprint_table,
            delete_target,
        ],
    )
    threshold.release(
        fn=apply_threshold,
        inputs=[threshold],
        outputs=[threshold, status_box, speaker_box, text_box, emotion_box, reg_state, log_box],
    )
    reject_threshold.release(
        fn=apply_reject_threshold,
        inputs=[reject_threshold],
        outputs=[reject_threshold, status_box, speaker_box, text_box, emotion_box, reg_state, log_box],
    )
    language_selector.change(
        fn=apply_language,
        inputs=[language_selector],
        outputs=[language_selector, status_box, speaker_box, text_box, emotion_box, reg_state, log_box],
    )
    start_reg_btn.click(
        fn=start_register_recording,
        inputs=[register_name],
        outputs=[register_name, reg_state, reg_timer, status_box, speaker_box, text_box, emotion_box, log_box],
    )
    stop_reg_btn.click(
        fn=stop_register_recording,
        outputs=[
            reg_state,
            reg_timer,
            status_box,
            speaker_box,
            text_box,
            emotion_box,
            log_box,
            voiceprint_table,
            delete_target,
        ],
    )
    refresh_btn.click(
        fn=refresh_voiceprints,
        outputs=[voiceprint_table, delete_target, status_box, speaker_box, text_box, emotion_box, reg_state, log_box],
    )
    delete_btn.click(
        fn=delete_voiceprint,
        inputs=[delete_target],
        outputs=[voiceprint_table, delete_target, status_box, speaker_box, text_box, emotion_box, reg_state, log_box],
    )

    timer = gr.Timer(0.8)
    timer.tick(
        fn=poll_updates,
        outputs=[
            status_box,
            speaker_box,
            text_box,
            emotion_box,
            reg_state,
            reg_timer,
            log_box,
            voiceprint_table,
            delete_target,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
