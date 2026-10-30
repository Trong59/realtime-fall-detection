"""
Phase 5 (bản Web): Flask server cho hệ thống phát hiện té ngã.

Kiến trúc:
    VideoProcessorThread (đã có từ Phase 5 desktop, không sửa gì)
              |  frame_queue (maxsize=1, dict: frame_rgb + stats)
              v
    PipelineBridge (thread cầu nối, CHỈ nơi duy nhất đọc frame_queue)
        - Giải mã dict -> mã hóa JPEG -> lưu vào latest_jpeg (có lock)
        - Lưu latest_stats (có lock) để /api/status đọc
        - Nhận callback on_alert -> ghi vào alerts_log (có lock)
              |
              v
    Flask routes (nhiều request cùng lúc, mỗi request chỉ ĐỌC, không tranh chấp)
        /video_feed   -> MJPEG stream, đọc latest_jpeg liên tục
        /api/status   -> JSON, đọc latest_stats + alerts_log
        /api/start    -> khởi tạo VideoProcessorThread + PipelineBridge
        /api/stop     -> dừng cả 2
        /api/videos   -> quét thư mục tìm sẵn file video test

LÝ DO cần bridge riêng thay vì để MJPEG generator tự đọc frame_queue:
    frame_queue chỉ chứa 1 phần tử (maxsize=1). Nếu để nhiều nơi (MJPEG +
    status polling) cùng gọi get_nowait(), chúng sẽ TRANH NHAU lấy - request
    nào tới trước sẽ "nuốt" mất frame của request kia, gây giật/mất số liệu.
    Bridge là nơi DUY NHẤT tiêu thụ queue, các route khác chỉ đọc bản sao mới
    nhất đã được bridge lưu sẵn.

Cách chạy (từ thư mục gốc fall_detection_system/):
    python web_server.py
    Rồi mở trình duyệt: http://127.0.0.1:5000
"""
import os
import threading
import time
from collections import deque

import cv2
from flask import Flask, Response, jsonify, render_template, request

from src.gui.video_worker import VideoProcessorThread

app = Flask(__name__)

# Các thư mục sẽ được quét tìm file video test (khớp với cấu trúc dự án của bạn)
VIDEO_SEARCH_DIRS = ["test video", "test_videos", "."]
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


class PipelineBridge:
    """Cầu nối giữa VideoProcessorThread (nền) và các Flask route (đọc)."""

    def __init__(self):
        self.worker = None
        self._lock = threading.Lock()
        self._bridge_thread = None
        self._stop_bridge = threading.Event()

        self.latest_jpeg = None
        self.latest_stats = {
            "running": False,
            "fps": 0.0,
            "probability": None,
            "static_duration": 0.0,
            "lstm_alert_active": False,
            "static_alert_active": False,
        }
        self.alerts_log = deque(maxlen=100)
        self.error = None

    def is_running(self):
        with self._lock:
            return self.worker is not None

    def start(self, config):
        with self._lock:
            if self.worker is not None:
                return False, "Pipeline đang chạy rồi."

            self.error = None
            self.latest_jpeg = None
            self.alerts_log.clear()
            self._stop_bridge.clear()

            self.worker = VideoProcessorThread(
                source=config["source"],
                threshold=config["threshold"],
                consecutive_frames=config["consecutive_frames"],
                cooldown=config["cooldown"],
                static_duration=config["static_duration"],
                enable_safety_net=config["enable_safety_net"],
                enable_sound=config["enable_sound"],
                on_alert=self._on_alert,
            )
            self.worker.start()

            self._bridge_thread = threading.Thread(target=self._bridge_loop, daemon=True)
            self._bridge_thread.start()

        return True, "OK"

    def stop(self):
        with self._lock:
            if self.worker is not None:
                self.worker.stop()
            self._stop_bridge.set()
            self.worker = None
        self.latest_stats["running"] = False

    def _on_alert(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.alerts_log.appendleft(f"[{timestamp}] {message}")

    def _bridge_loop(self):
        """Vòng lặp DUY NHẤT tiêu thụ frame_queue của worker."""
        worker = self.worker
        worker.ready_event.wait(timeout=30)
        if worker.error:
            self.error = worker.error
            self.latest_stats["running"] = False
            return

        self.latest_stats["running"] = True

        while not self._stop_bridge.is_set():
            try:
                item = worker.frame_queue.get(timeout=1.0)
            except Exception:
                if worker.error:
                    self.error = worker.error
                    break
                continue

            if item.get("type") == "end_of_stream":
                self.latest_stats["running"] = False
                self._on_alert("Hết video / mất kết nối nguồn.")
                break

            if item.get("type") == "frame":
                # frame_rgb (dùng cho PIL/Tkinter ở bản desktop) -> đổi lại BGR
                # trước khi encode JPEG, vì cv2.imencode mặc định hiểu input
                # là BGR - nếu encode thẳng RGB, ảnh hiển thị sẽ bị lệch màu
                # (đỏ/xanh dương đảo ngược).
                frame_bgr = cv2.cvtColor(item["frame_rgb"], cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.latest_jpeg = buf.tobytes()

                self.latest_stats.update({
                    "running": True,
                    "fps": item.get("fps", 0.0),
                    "probability": item.get("probability"),
                    "static_duration": item.get("static_duration", 0.0),
                    "lstm_alert_active": item.get("lstm_alert_active", False),
                    "static_alert_active": item.get("static_alert_active", False),
                })

        self.latest_stats["running"] = False

    def get_status(self):
        return {
            **self.latest_stats,
            "alerts": list(self.alerts_log),
            "error": self.error,
        }


bridge = PipelineBridge()


def generate_mjpeg():
    placeholder = _make_placeholder_jpeg("Chưa chạy - Nhấn Bắt đầu")
    while True:
        if bridge.latest_jpeg is not None and bridge.is_running():
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bridge.latest_jpeg + b"\r\n")
        else:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n")
        time.sleep(0.033)  # ~30 fps giới hạn tốc độ gửi, không phụ thuộc tốc độ xử lý AI


def _make_placeholder_jpeg(text):
    import numpy as np
    img = np.zeros((480, 720, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.putText(img, text, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(bridge.get_status())


@app.route("/api/videos")
def api_videos():
    found = []
    for d in VIDEO_SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith(VIDEO_EXTENSIONS):
                rel_path = os.path.join(d, fname) if d != "." else fname
                found.append(rel_path)
    # Loại trùng, giữ thứ tự
    seen = set()
    unique = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return jsonify({"videos": unique})


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(force=True)

    source_mode = data.get("source_mode", "webcam")
    if source_mode == "webcam":
        source = 0
    else:
        source = data.get("file_path")
        if not source:
            return jsonify({"status": "error", "message": "Chưa chọn file video."}), 400

    config = {
        "source": source,
        "threshold": float(data.get("threshold", 0.60)),
        "consecutive_frames": int(data.get("consecutive_frames", 5)),
        "cooldown": float(data.get("cooldown", 3.0)),
        "static_duration": float(data.get("static_duration", 4.0)),
        "enable_safety_net": bool(data.get("enable_safety_net", True)),
        "enable_sound": bool(data.get("enable_sound", True)),
    }

    ok, message = bridge.start(config)
    status_code = 200 if ok else 409
    return jsonify({"status": "ok" if ok else "error", "message": message}), status_code


@app.route("/api/stop", methods=["POST"])
def api_stop():
    bridge.stop()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # QUAN TRỌNG: use_reloader=False - nếu bật (mặc định khi debug=True), Flask
    # sẽ khởi động 2 tiến trình song song (1 để theo dõi thay đổi code), khiến
    # global `bridge` bị nhân đôi và pipeline chạy 2 lần cùng lúc.
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
