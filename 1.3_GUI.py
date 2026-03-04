import importlib.util
import queue
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText


BACKEND_PATH = Path(__file__).with_name("1.3_main.py")


def load_backend_class():
    spec = importlib.util.spec_from_file_location("asr_backend_13", BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载后端文件: {BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RealtimeASRService


class ASRGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ASR 声纹管理界面")
        self.root.geometry("980x700")
        self.root.minsize(800, 560)

        self.status_var = tk.StringVar(value="状态：未启动")
        self.current_speaker_var = tk.StringVar(value="当前说话人：无")
        self.last_text_var = tk.StringVar(value="最近识别：")
        self.threshold_var = tk.IntVar(value=600)

        self.ui_queue: queue.Queue[tuple[str, str | dict]] = queue.Queue()

        backend_cls = load_backend_class()
        self.service = backend_cls(on_log=self._on_backend_log, on_result=self._on_backend_result)
        self.threshold_var.set(self.service.get_energy_threshold())

        self._build_ui()
        self._poll_ui_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x")

        ttk.Label(
            header,
            text="实时语音识别与声纹管理",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(side="left")

        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True, pady=(10, 0))

        self.page_realtime = ttk.Frame(self.notebook, padding=10)
        self.page_threshold = ttk.Frame(self.notebook, padding=10)
        self.page_voiceprints = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.page_realtime, text="实时识别")
        self.notebook.add(self.page_threshold, text="阈值设置")
        self.notebook.add(self.page_voiceprints, text="已注册声纹")

        self._build_realtime_page()
        self._build_threshold_page()
        self._build_voiceprints_page()

    def _build_realtime_page(self) -> None:
        action_frame = ttk.LabelFrame(self.page_realtime, text="控制", padding=10)
        action_frame.pack(fill="x")

        self.start_btn = ttk.Button(action_frame, text="启动识别", command=self.on_start)
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = ttk.Button(action_frame, text="停止识别", command=self.on_stop)
        self.stop_btn.pack(side="left", padx=(0, 8))

        ttk.Button(action_frame, text="注册声纹", command=self.on_register).pack(side="left", padx=(0, 8))
        ttk.Button(action_frame, text="刷新声纹库", command=self.on_refresh).pack(side="left", padx=(0, 8))

        info_frame = ttk.Frame(self.page_realtime)
        info_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(info_frame, textvariable=self.current_speaker_var).pack(side="left", padx=(0, 16))
        ttk.Label(info_frame, textvariable=self.last_text_var).pack(side="left")

        log_frame = ttk.LabelFrame(self.page_realtime, text="识别日志（实时刷新）", padding=8)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))

        self.log_text = ScrolledText(log_frame, wrap="word", font=("Consolas", 11))
        self.log_text.pack(fill="both", expand=True)

        self.append_log("界面已就绪。点击“启动识别”开始。")

    def _build_threshold_page(self) -> None:
        frame = ttk.LabelFrame(self.page_threshold, text="开始说话检测阈值", padding=12)
        frame.pack(fill="x")

        ttk.Label(
            frame,
            text="阈值越小越敏感，越大越不容易触发。建议在 300-1500 范围内调试。",
        ).pack(anchor="w")

        slider_row = ttk.Frame(frame)
        slider_row.pack(fill="x", pady=(10, 0))

        self.threshold_scale = ttk.Scale(
            slider_row,
            from_=100,
            to=5000,
            orient="horizontal",
            command=self._on_threshold_scale,
        )
        self.threshold_scale.set(self.threshold_var.get())
        self.threshold_scale.pack(side="left", fill="x", expand=True)

        self.threshold_value_label = ttk.Label(slider_row, text=f"{self.threshold_var.get()}")
        self.threshold_value_label.pack(side="left", padx=(10, 0))

        action_row = ttk.Frame(frame)
        action_row.pack(fill="x", pady=(10, 0))

        ttk.Button(action_row, text="应用阈值", command=self.apply_threshold).pack(side="left", padx=(0, 8))
        ttk.Button(action_row, text="读取当前阈值", command=self.sync_threshold_from_service).pack(side="left")

    def _build_voiceprints_page(self) -> None:
        top = ttk.Frame(self.page_voiceprints)
        top.pack(fill="x")

        ttk.Button(top, text="刷新列表", command=self.refresh_voiceprint_table).pack(side="left")
        ttk.Button(top, text="删除选中", command=self.delete_selected_voiceprint).pack(side="left", padx=(8, 0))

        table_frame = ttk.LabelFrame(self.page_voiceprints, text="已注册声纹", padding=8)
        table_frame.pack(fill="both", expand=True, pady=(10, 0))

        columns = ("name", "samples")
        self.voiceprint_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        self.voiceprint_table.heading("name", text="说话人")
        self.voiceprint_table.heading("samples", text="样本数")
        self.voiceprint_table.column("name", width=220, anchor="w")
        self.voiceprint_table.column("samples", width=100, anchor="center")
        self.voiceprint_table.pack(fill="both", expand=True)

        self.refresh_voiceprint_table()

    def _on_threshold_scale(self, value: str) -> None:
        v = int(float(value))
        self.threshold_var.set(v)
        self.threshold_value_label.config(text=str(v))

    def apply_threshold(self) -> None:
        value = self.threshold_var.get()
        final_value = self.service.set_energy_threshold(value)
        self.threshold_var.set(final_value)
        self.threshold_scale.set(final_value)
        self.threshold_value_label.config(text=str(final_value))

    def sync_threshold_from_service(self) -> None:
        value = self.service.get_energy_threshold()
        self.threshold_var.set(value)
        self.threshold_scale.set(value)
        self.threshold_value_label.config(text=str(value))
        self.append_log(f"[配置] 当前开始说话阈值：{value}")

    def refresh_voiceprint_table(self) -> None:
        for item in self.voiceprint_table.get_children():
            self.voiceprint_table.delete(item)

        rows = self.service.get_voiceprint_summary()
        if not rows:
            self.append_log("[声纹] 列表为空，当前没有已注册声纹")
            return

        for row in rows:
            self.voiceprint_table.insert("", "end", values=(row["name"], row["samples"]))
        self.append_log(f"[声纹] 已加载 {len(rows)} 条注册记录")

    def delete_selected_voiceprint(self) -> None:
        selected = self.voiceprint_table.selection()
        if not selected:
            messagebox.showinfo("提示", "请先在列表中选择要删除的声纹")
            return

        item_id = selected[0]
        values = self.voiceprint_table.item(item_id, "values")
        if not values:
            messagebox.showerror("错误", "未读取到选中项数据")
            return

        speaker_name = str(values[0]).strip()
        if not speaker_name:
            messagebox.showerror("错误", "说话人名称无效")
            return

        ok = messagebox.askyesno("确认删除", f"确定删除声纹“{speaker_name}”？")
        if not ok:
            return

        deleted = self.service.delete_voiceprint(speaker_name)
        if deleted:
            self.append_log(f"[声纹] 已删除：{speaker_name}")
            self.refresh_voiceprint_table()
        else:
            messagebox.showwarning("提示", f"删除失败，未找到：{speaker_name}")

    def _on_backend_log(self, message: str) -> None:
        self.ui_queue.put(("log", message))

    def _on_backend_result(self, payload: dict) -> None:
        self.ui_queue.put(("result", payload))

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self.append_log(str(payload))
                if "注册完成" in str(payload):
                    self.refresh_voiceprint_table()
            elif kind == "result":
                speaker = str(payload.get("speaker", "陌生人"))
                score = float(payload.get("score", 0.0))
                text = str(payload.get("text", ""))
                self.current_speaker_var.set(f"当前说话人：{speaker}（相似度 {score:.2f}）")
                self.last_text_var.set(f"最近识别：{text}")
                self.append_log(f"[识别][{speaker}|{score:.2f}] {text}")

        self.root.after(100, self._poll_ui_queue)

    def append_log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def on_start(self) -> None:
        self.service.start()
        self.status_var.set("状态：运行中")

    def on_stop(self) -> None:
        self.service.stop()
        self.status_var.set("状态：已停止")

    def on_register(self) -> None:
        speaker_name = simpledialog.askstring("注册声纹", "输入说话人名称（可留空自动命名）：", parent=self.root)
        self.service.request_register(speaker_name or "")

    def on_refresh(self) -> None:
        names = self.service.reload_voiceprints()
        self.refresh_voiceprint_table()
        if names:
            self.append_log(f"[声纹] 已刷新：{', '.join(names)}")
        else:
            self.append_log("[声纹] 已刷新：当前没有已注册说话人")

    def on_close(self) -> None:
        self.service.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ASRGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
