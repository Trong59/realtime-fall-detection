"""
Worker thread đóng gói pipeline xử lý (YOLO -> LSTM -> Safety Net) chạy nền,
tách khỏi thread chính của GUI để không làm đơ giao diện.

Thiết kế giao tiếp: dùng queue.Queue(maxsize=1) cho frame hiển thị - nếu GUI
vẽ chậm hơn tốc độ xử lý, frame cũ bị bỏ qua thay vì xếp hàng dồn cục (tránh
video bị trễ tích lũy theo thời gian). Queue riêng (không giới hạn) cho log
cảnh báo để không bỏ sót sự kiện quan trọng.
"""
import queue
import threading
import time

import cv2
import numpy as np

from src.pose.yolo_extractor import YOLOPoseExtractor
from src.utils.normalization import normalize_keypoints
from src.utils.fps import FPSCalculator
from src.utils.alert_sound import try_beep
from src.fall_detection.lstm_classifier import FallLSTMPredictor
from src.fall_detection.safety_net import StaticImmobilitySafetyNet


class VideoProcessorThread(threading.Thread):
    def __init__(
        self,
        source,
        yolo_weights="yolov8n-pose.pt",
        lstm_weights="best_fall_lstm.pth",
        threshold=0.60,
        consecutive_frames=5,
        cooldown=3.0,
        yolo_conf=0.5,
        device="cpu",
        static_duration=4.0,
        enable_safety_net=True,
        enable_sound=True,
        on_alert=None,
    ):
        super().__init__(daemon=True)
        self.source = source
        self.yolo_weights = yolo_weights
        self.lstm_weights = lstm_weights
        self.threshold = threshold
        self.consecutive_frames = consecutive_frames
        self.cooldown = cooldown
        self.yolo_conf = yolo_conf
        self.device = device
        self.static_duration = static_duration
        self.enable_safety_net = enable_safety_net
        self.enable_sound = enable_sound
        self.on_alert = on_alert  # callback(str) - gọi khi có cảnh báo mới, dùng cho log

        self.frame_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self.error = None
        self.ready_event = threading.Event()  # báo hiệu đã load xong model

    def stop(self):
        self._stop_event.set()

    def _put_latest_frame(self, item):
        """Đẩy frame mới nhất vào queue, bỏ frame cũ nếu GUI chưa kịp lấy."""
        try:
            self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(item)
        except queue.Full:
            pass

    def run(self):
        try:
            extractor = YOLOPoseExtractor(
                model_path=self.yolo_weights, conf_threshold=self.yolo_conf, device=self.device
            )
            predictor = FallLSTMPredictor(
                weight_path=self.lstm_weights,
                sequence_length=30,
                input_size=34,
                threshold=self.threshold,
                consecutive_frames_required=self.consecutive_frames,
                cooldown_seconds=self.cooldown,
            )
            safety_net = (
                StaticImmobilitySafetyNet(min_duration_seconds=self.static_duration)
                if self.enable_safety_net
                else None
            )
        except Exception as e:
            self.error = f"Lỗi khi tải model: {e}"
            self.ready_event.set()
            return

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"Không mở được nguồn video: {self.source}"
            self.ready_event.set()
            return

        self.ready_event.set()
        fps_calc = FPSCalculator(buffer_size=30)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                self._put_latest_frame({"type": "end_of_stream"})
                break

            kps, bbox, _ = extractor.extract_pose(frame)

            if kps is not None:
                norm_kps = normalize_keypoints(kps, bbox)
                extractor.draw_skeleton(frame, kps, bbox, label="Person")
            else:
                norm_kps = np.zeros(34, dtype=np.float32)

            lstm_result = predictor.update(norm_kps)
            if lstm_result["ready"] and lstm_result["alert"]:
                msg = f"[LSTM] CẢNH BÁO TÉ NGÃ! probability={lstm_result['probability']:.3f}"
                if self.enable_sound:
                    try_beep()
                if self.on_alert:
                    self.on_alert(msg)

            static_result = {"duration": 0.0, "alert": False}
            if safety_net is not None:
                static_result = safety_net.update(kps, bbox)
                if static_result["alert"]:
                    msg = f"[SAFETY NET] Nằm bất động {static_result['duration']:.1f}s"
                    if self.enable_sound:
                        try_beep()
                    if self.on_alert:
                        self.on_alert(msg)

            fps = fps_calc.update()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._put_latest_frame({
                "type": "frame",
                "frame_rgb": frame_rgb,
                "fps": fps,
                "probability": lstm_result["probability"] if lstm_result["ready"] else None,
                "lstm_alert_active": lstm_result.get("alert", False),
                "static_duration": static_result["duration"],
                "static_alert_active": static_result.get("alert", False),
            })

        cap.release()
