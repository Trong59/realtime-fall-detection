import time
from collections import deque

class FPSCalculator:
    def __init__(self, buffer_size=30):
        """
        Khởi tạo bộ tính toán FPS với trung bình trượt (sliding window).
        
        Args:
            buffer_size (int): Số khung hình dùng để tính trung bình trượt.
        """
        self.timestamps = deque(maxlen=buffer_size)
        self.fps = 0.0

    def update(self):
        """
        Gọi hàm này ở cuối mỗi vòng lặp xử lý khung hình để cập nhật FPS.
        """
        current_time = time.time()
        self.timestamps.append(current_time)
        
        if len(self.timestamps) > 1:
            total_time = self.timestamps[-1] - self.timestamps[0]
            if total_time > 0:
                self.fps = (len(self.timestamps) - 1) / total_time
            else:
                self.fps = 0.0
        return self.fps

    def get_fps(self):
        """
        Trả về giá trị FPS hiện tại.
        """
        return self.fps
