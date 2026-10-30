"""
Lớp bảo vệ thứ 2 (safety net) cho hệ thống phát hiện té ngã.

BỐI CẢNH: LSTM (lstm_classifier.py) được train để nhận diện HÀNH ĐỘNG ngã
đang xảy ra (chuỗi chuyển động đứng -> mất thăng bằng -> va chạm). Nó KHÔNG
được thiết kế để nhận diện TRẠNG THÁI đã nằm sẵn, bất động từ trước (ví dụ
hệ thống mới khởi động, hoặc nạn nhân ngã ở góc khuất rồi mới vào khung hình).
Đây là giới hạn đã được xác minh thực nghiệm (xem fall_example_1.mp4 vs
fall_example_2.mp4).

GIẢI PHÁP: Dùng lại chỉ số hình học tĩnh (body_angle, aspect_ratio) đã có
sẵn trong FallHeuristics (Phase 2), nhưng xác nhận theo THỜI GIAN THỰC
(giây) thay vì đếm frame - để không phụ thuộc vào FPS dao động của máy.
"""
import time

from src.fall_detection.heuristics import FallHeuristics


class StaticImmobilitySafetyNet:
    def __init__(
        self,
        angle_threshold=60.0,
        aspect_ratio_threshold=0.85,
        min_duration_seconds=4.0,
        flicker_tolerance_seconds=1.0,
        cooldown_seconds=5.0,
    ):
        """
        Args:
            angle_threshold: Ngưỡng góc nghiêng thân người (độ) - dùng lại
                đúng công thức từ FallHeuristics.calculate_metrics().
            aspect_ratio_threshold: Ngưỡng tỷ lệ width/height bbox.
            min_duration_seconds: Số giây liên tục ở tư thế nằm ngang mới
                xác nhận cảnh báo (tránh nhầm với cúi/ngồi xổm nhanh).
            flicker_tolerance_seconds: Cho phép mất detect thoáng qua tối đa
                bao nhiêu giây trước khi coi là "người đã đứng dậy/rời đi"
                và reset bộ đếm.
            cooldown_seconds: Khóa cảnh báo trong bao lâu sau khi đã báo,
                tránh spam liên tục khi nạn nhân vẫn nằm im.
        """
        self._metrics_helper = FallHeuristics(
            angle_threshold=angle_threshold,
            aspect_ratio_threshold=aspect_ratio_threshold,
        )
        self.min_duration_seconds = min_duration_seconds
        self.flicker_tolerance_seconds = flicker_tolerance_seconds
        self.cooldown_seconds = cooldown_seconds

        self._lying_start_time = None
        self._last_seen_lying_time = None
        self._last_alert_time = 0.0

    def reset(self):
        self._lying_start_time = None
        self._last_seen_lying_time = None
        self._last_alert_time = 0.0
        self._metrics_helper.reset_history()

    def update(self, keypoints, bbox):
        """
        Args:
            keypoints: np.ndarray (17, 3) hoặc None.
            bbox: [x1, y1, x2, y2] hoặc None.

        Returns:
            dict:
                is_lying_now (bool): frame hiện tại có ở tư thế nằm ngang không
                duration (float): số giây đã nằm liên tục (0 nếu không nằm)
                alert (bool): có nên kích hoạt cảnh báo an toàn dự phòng không
        """
        now = time.time()
        metrics = self._metrics_helper.calculate_metrics(keypoints, bbox)

        is_lying_now = (
            metrics["is_valid"]
            and metrics["body_angle"] > self._metrics_helper.angle_threshold
            and metrics["aspect_ratio"] > self._metrics_helper.aspect_ratio_threshold
        )

        if is_lying_now:
            if self._lying_start_time is None:
                self._lying_start_time = now
            self._last_seen_lying_time = now
        else:
            # Dung sai nhiễu: chỉ reset nếu mất tín hiệu "đang nằm" quá lâu
            if (
                self._last_seen_lying_time is not None
                and (now - self._last_seen_lying_time) > self.flicker_tolerance_seconds
            ):
                self._lying_start_time = None
                self._last_seen_lying_time = None

        duration = (now - self._lying_start_time) if self._lying_start_time else 0.0

        alert = False
        if (
            duration >= self.min_duration_seconds
            and (now - self._last_alert_time) >= self.cooldown_seconds
        ):
            alert = True
            self._last_alert_time = now

        return {"is_lying_now": is_lying_now, "duration": duration, "alert": alert}
