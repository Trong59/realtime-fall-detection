"""
Phase 5: Giao diện GUI (CustomTkinter) cho hệ thống phát hiện té ngã.

Kiến trúc:
    Main thread (GUI)  <---queue.Queue--->  VideoProcessorThread (nền)
    Chỉ vẽ lại giao diện, không xử lý AI trực tiếp -> không bị đơ.

Cách chạy (từ thư mục gốc fall_detection_system/):
    python -m src.gui.app
"""
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

# Cho phép chạy trực tiếp file này (python src/gui/app.py) lẫn qua -m
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.gui.video_worker import VideoProcessorThread

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLOR_ALERT_LSTM = "#e74c3c"     # đỏ
COLOR_ALERT_STATIC = "#e67e22"   # cam
COLOR_NORMAL = "#2ecc71"         # xanh lá


class FallDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hệ Thống Phát Hiện Té Ngã - Phase 5")
        self.geometry("1200x720")
        self.minsize(1000, 620)

        self.worker = None
        self.video_source_path = None  # None -> dùng webcam (index 0)

        self._build_layout()
        self._poll_queue()  # bắt đầu vòng lặp polling GUI

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Cột trái: khung hiển thị video ---
        left_frame = ctk.CTkFrame(self, corner_radius=8)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(left_frame, text="Chưa chạy - Nhấn 'Bắt đầu' để khởi động",
                                         fg_color="#111111", corner_radius=8)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.status_banner = ctk.CTkLabel(left_frame, text="TRẠNG THÁI: BÌNH THƯỜNG",
                                           font=ctk.CTkFont(size=18, weight="bold"),
                                           text_color=COLOR_NORMAL)
        self.status_banner.grid(row=1, column=0, pady=(0, 8))

        # --- Cột phải: bảng điều khiển ---
        right_frame = ctk.CTkScrollableFrame(self, corner_radius=8, width=320)
        right_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")

        ctk.CTkLabel(right_frame, text="NGUỒN VIDEO", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(4, 2))
        self.source_mode = tk.StringVar(value="webcam")
        ctk.CTkRadioButton(right_frame, text="Webcam mặc định", variable=self.source_mode,
                            value="webcam", command=self._update_source_widgets).pack(anchor="w", pady=2)
        ctk.CTkRadioButton(right_frame, text="File video", variable=self.source_mode,
                            value="file", command=self._update_source_widgets).pack(anchor="w", pady=2)

        self.file_path_label = ctk.CTkLabel(right_frame, text="(chưa chọn file)",
                                             text_color="gray70", wraplength=260, justify="left")
        self.file_path_label.pack(anchor="w", pady=(0, 2))
        self.browse_btn = ctk.CTkButton(right_frame, text="Chọn file video...",
                                         command=self._browse_file, state="disabled")
        self.browse_btn.pack(anchor="w", pady=(0, 10), fill="x")

        ctk.CTkLabel(right_frame, text="THAM SỐ MÔ HÌNH", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(10, 2))

        self.threshold_var = ctk.DoubleVar(value=0.60)
        self._build_slider_row(right_frame, "Ngưỡng LSTM (threshold)", self.threshold_var,
                                0.05, 0.95, decimals=2)

        self.consecutive_var = ctk.IntVar(value=5)
        self._build_slider_row(right_frame, "Số frame liên tiếp xác nhận", self.consecutive_var,
                                1, 15, decimals=0)

        self.cooldown_var = ctk.DoubleVar(value=3.0)
        self._build_slider_row(right_frame, "Cooldown LSTM (giây)", self.cooldown_var,
                                0.5, 10.0, decimals=1)

        self.static_duration_var = ctk.DoubleVar(value=4.0)
        self._build_slider_row(right_frame, "Ngưỡng bất động (giây)", self.static_duration_var,
                                1.0, 15.0, decimals=1)

        self.safety_net_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(right_frame, text="Bật Safety Net (nằm bất động)",
                         variable=self.safety_net_var).pack(anchor="w", pady=(6, 2))

        self.sound_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(right_frame, text="Bật âm thanh cảnh báo",
                         variable=self.sound_var).pack(anchor="w", pady=2)

        ctk.CTkLabel(right_frame, text="ĐIỀU KHIỂN", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(14, 2))
        self.start_btn = ctk.CTkButton(right_frame, text="▶ Bắt đầu", fg_color=COLOR_NORMAL,
                                        hover_color="#27ae60", command=self._start)
        self.start_btn.pack(fill="x", pady=4)
        self.stop_btn = ctk.CTkButton(right_frame, text="■ Dừng", fg_color=COLOR_ALERT_LSTM,
                                       hover_color="#c0392b", command=self._stop, state="disabled")
        self.stop_btn.pack(fill="x", pady=4)

        ctk.CTkLabel(right_frame, text="THÔNG SỐ THỜI GIAN THỰC", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(14, 2))
        self.fps_label = ctk.CTkLabel(right_frame, text="FPS: -")
        self.fps_label.pack(anchor="w", pady=1)
        self.prob_label = ctk.CTkLabel(right_frame, text="P(fall): -")
        self.prob_label.pack(anchor="w", pady=1)
        self.static_label = ctk.CTkLabel(right_frame, text="Bất động: 0.0s")
        self.static_label.pack(anchor="w", pady=1)

        ctk.CTkLabel(right_frame, text="NHẬT KÝ CẢNH BÁO", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", pady=(14, 2))
        self.log_box = ctk.CTkTextbox(right_frame, height=160, state="disabled")
        self.log_box.pack(fill="both", pady=(0, 10))

    def _build_slider_row(self, parent, label, var, from_, to, decimals=2):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=3)
        initial_text = f"{label}: {var.get():.{decimals}f}" if decimals else f"{label}: {int(var.get())}"
        value_label = ctk.CTkLabel(row, text=initial_text, anchor="w")
        value_label.pack(anchor="w")

        def _on_change(v):
            text = f"{label}: {float(v):.{decimals}f}" if decimals else f"{label}: {int(float(v))}"
            value_label.configure(text=text)

        slider = ctk.CTkSlider(row, from_=from_, to=to, variable=var, command=_on_change)
        slider.pack(fill="x")

    # -------------------------------------------------------------- Events
    def _update_source_widgets(self):
        is_file = self.source_mode.get() == "file"
        self.browse_btn.configure(state="normal" if is_file else "disabled")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Chọn video test",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("Tất cả file", "*.*")],
        )
        if path:
            self.video_source_path = path
            self.file_path_label.configure(text=os.path.basename(path))

    def _resolve_source(self):
        if self.source_mode.get() == "webcam":
            return 0
        if not self.video_source_path:
            messagebox.showwarning("Thiếu file", "Vui lòng chọn file video trước khi bắt đầu.")
            return None
        return self.video_source_path

    def _start(self):
        source = self._resolve_source()
        if source is None:
            return

        self.worker = VideoProcessorThread(
            source=source,
            threshold=self.threshold_var.get(),
            consecutive_frames=int(self.consecutive_var.get()),
            cooldown=self.cooldown_var.get(),
            static_duration=self.static_duration_var.get(),
            enable_safety_net=self.safety_net_var.get(),
            enable_sound=self.sound_var.get(),
            on_alert=self._queue_log_message,
        )
        self.worker.start()

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._append_log("Đã khởi động pipeline...")

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_banner.configure(text="TRẠNG THÁI: ĐÃ DỪNG", text_color="gray70")

    def _on_close(self):
        self._stop()
        self.destroy()

    # ---------------------------------------------------------- Log queue
    def _queue_log_message(self, message):
        # on_alert được gọi TỪ worker thread -> không được đụng trực tiếp vào
        # widget Tkinter ở đây (không thread-safe). Chỉ lưu tạm, _poll_queue
        # (chạy ở main thread) sẽ đọc và cập nhật UI.
        if not hasattr(self, "_pending_log_messages"):
            self._pending_log_messages = []
        self._pending_log_messages.append(message)

    def _append_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"[{timestamp}] {message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    # ------------------------------------------------------------- Polling
    def _poll_queue(self):
        # Xử lý log cảnh báo tích lũy từ worker thread (an toàn vì chạy ở main thread)
        pending = getattr(self, "_pending_log_messages", [])
        if pending:
            for msg in pending:
                self._append_log(msg)
            self._pending_log_messages = []

        if self.worker is not None:
            try:
                item = self.worker.frame_queue.get_nowait()
            except Exception:
                item = None

            if item is not None:
                if item.get("type") == "end_of_stream":
                    self._append_log("Hết video / mất kết nối nguồn.")
                    self._stop()
                elif item.get("type") == "frame":
                    self._render_frame(item)

            if self.worker is not None and self.worker.error:
                messagebox.showerror("Lỗi", self.worker.error)
                self._append_log(f"Lỗi: {self.worker.error}")
                self._stop()

        self.after(15, self._poll_queue)

    def _render_frame(self, item):
        frame_rgb = item["frame_rgb"]
        img = Image.fromarray(frame_rgb)

        # Co giãn ảnh vừa khung hiển thị, giữ tỷ lệ
        target_w = max(self.video_label.winfo_width(), 320)
        target_h = max(self.video_label.winfo_height(), 240)
        img.thumbnail((target_w, target_h))

        photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # giữ tham chiếu, tránh bị garbage collect

        fps = item.get("fps", 0.0)
        prob = item.get("probability")
        static_dur = item.get("static_duration", 0.0)

        self.fps_label.configure(text=f"FPS: {fps:.1f}")
        self.prob_label.configure(text=f"P(fall): {prob:.2f}" if prob is not None else "P(fall): N/A")
        self.static_label.configure(text=f"Bất động: {static_dur:.1f}s")

        if item.get("lstm_alert_active"):
            self.status_banner.configure(text="⚠ CẢNH BÁO TÉ NGÃ!", text_color=COLOR_ALERT_LSTM)
        elif item.get("static_alert_active"):
            self.status_banner.configure(text="⚠ NGHI NGỜ BẤT ĐỘNG - KIỂM TRA NGAY",
                                          text_color=COLOR_ALERT_STATIC)
        else:
            self.status_banner.configure(text="TRẠNG THÁI: BÌNH THƯỜNG", text_color=COLOR_NORMAL)


def main():
    app = FallDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
