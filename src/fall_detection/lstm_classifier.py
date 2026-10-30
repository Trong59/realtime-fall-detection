"""
Bộ phân loại té ngã thời gian thực dựa trên LSTM đã train.

Quản lý:
  1. Sliding window buffer (deque maxlen=30) chứa các vector keypoints
     đã chuẩn hóa, cập nhật mỗi frame (stride=1, khác với lúc train
     dùng stride=5 để tăng lượng data).
  2. Dự đoán xác suất té ngã mỗi khi buffer đầy.
  3. Cơ chế hysteresis: chỉ báo "FALL" khi có N frame liên tiếp vượt
     ngưỡng, tránh báo động giả do dao động 1 frame.
  4. Cooldown: sau khi báo động, khóa cảnh báo trong T giây để tránh
     spam liên tục khi nạn nhân vẫn nằm im.
"""
import time
from collections import deque

import numpy as np
import torch

from src.models.lstm import FallDetectionLSTM


class FallLSTMPredictor:
    def __init__(
        self,
        weight_path,
        sequence_length=30,
        input_size=34,
        hidden_size=64,
        num_layers=2,
        threshold=0.60,
        consecutive_frames_required=5,
        cooldown_seconds=3.0,
        device=None,
    ):
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.consecutive_frames_required = consecutive_frames_required
        self.cooldown_seconds = cooldown_seconds

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = FallDetectionLSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=1
        ).to(self.device)
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.buffer = deque(maxlen=sequence_length)

        # Trạng thái cho cơ chế hysteresis + cooldown
        self._consecutive_hits = 0
        self._last_alert_time = 0.0

    def reset(self):
        """Xóa buffer - gọi khi chuyển video/nguồn mới để tránh rò rỉ context cũ."""
        self.buffer.clear()
        self._consecutive_hits = 0
        self._last_alert_time = 0.0

    def update(self, normalized_keypoints):
        """
        Thêm 1 frame mới vào buffer và trả về kết quả dự đoán.

        Args:
            normalized_keypoints: np.ndarray shape (34,) - đã qua
                normalize_keypoints(). Nếu không detect được người,
                truyền vào vector zeros(34) để giữ tính liên tục thời
                gian (giống hệt logic lúc chuẩn bị dataset training).

        Returns:
            dict:
                ready (bool): buffer đã đủ 30 frame chưa
                probability (float | None): xác suất té ngã [0,1]
                is_fall_frame (bool): frame này có vượt ngưỡng không
                alert (bool): có nên kích hoạt CẢNH BÁO ngay bây giờ không
                              (đã qua hysteresis + cooldown)
        """
        self.buffer.append(normalized_keypoints)

        result = {"ready": False, "probability": None, "is_fall_frame": False, "alert": False}

        if len(self.buffer) < self.sequence_length:
            return result

        result["ready"] = True

        seq = np.array(self.buffer, dtype=np.float32)  # (30, 34)
        seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, 30, 34)

        with torch.no_grad():
            logit = self.model(seq_tensor)
            prob = torch.sigmoid(logit).item()

        result["probability"] = prob
        is_fall_frame = prob > self.threshold
        result["is_fall_frame"] = is_fall_frame

        # --- Hysteresis: đếm số frame liên tiếp vượt ngưỡng ---
        if is_fall_frame:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        now = time.time()
        should_trigger = (
            self._consecutive_hits >= self.consecutive_frames_required
            and (now - self._last_alert_time) >= self.cooldown_seconds
        )

        if should_trigger:
            result["alert"] = True
            self._last_alert_time = now
            # Reset đếm để cần đủ N frame liên tiếp mới cho lần báo động tiếp theo
            self._consecutive_hits = 0

        return result
