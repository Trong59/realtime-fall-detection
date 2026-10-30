import numpy as np
import math

class FallHeuristics:
    def __init__(self, aspect_ratio_threshold=0.85, angle_threshold=60.0, velocity_threshold=0.15, history_len=10):
        """
        Khởi tạo bộ phát hiện té ngã bằng quy luật vật lý.
        
        Args:
            aspect_ratio_threshold (float): Ngưỡng tỷ lệ Width / Height của Bounding Box.
            angle_threshold (float): Ngưỡng góc nghiêng cơ thể so với trục đứng (độ).
            velocity_threshold (float): Ngưỡng vận tốc rơi của hông (đã chuẩn hóa).
            history_len (int): Số khung hình lưu trữ lịch sử để tính vận tốc và làm mịn kết quả.
        """
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.angle_threshold = angle_threshold
        self.velocity_threshold = velocity_threshold
        
        # Lưu lịch sử tọa độ Y của hông để tính vận tốc rơi
        self.hip_y_history = []
        # Lưu lịch sử chiều dài thân người để chuẩn hóa khoảng cách
        self.body_length_history = []
        self.history_len = history_len
        
        # Đếm số khung hình liên tiếp thỏa mãn điều kiện ngã để tránh nhiễu (False Positives)
        self.fall_counter = 0
        self.fall_confirmation_frames = 4  # Cần ít nhất 4 frames liên tiếp thỏa mãn để xác nhận ngã

    def reset_history(self):
        self.hip_y_history.clear()
        self.body_length_history.clear()
        self.fall_counter = 0

    def calculate_metrics(self, keypoints, bbox):
        """
        Tính toán các chỉ số hình học và động học từ keypoints và bbox.
        
        Args:
            keypoints (np.ndarray): Mảng shape (17, 3) chứa tọa độ keypoints của YOLOv8.
            bbox (list): [x1, y1, x2, y2]
            
        Returns:
            metrics (dict): Chứa aspect_ratio, body_angle, hip_velocity.
        """
        metrics = {
            "aspect_ratio": 0.0,
            "body_angle": 0.0,
            "hip_velocity": 0.0,
            "is_valid": False
        }
        
        if keypoints is None or bbox is None:
            self.reset_history()
            return metrics

        # 1. Tính Tỷ lệ khung hình (Aspect Ratio = Width / Height)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = max(y2 - y1, 1.0) # Tránh chia cho 0
        aspect_ratio = width / height
        metrics["aspect_ratio"] = aspect_ratio

        # 2. Tính Góc nghiêng thân người (Body Angle)
        # Điểm 5: Vai trái, 6: Vai phải
        # Điểm 11: Hông trái, 12: Hông phải
        kp_l_shoulder = keypoints[5]
        kp_r_shoulder = keypoints[6]
        kp_l_hip = keypoints[11]
        kp_r_hip = keypoints[12]

        # Kiểm tra độ tin cậy của các khớp vai và hông
        if (kp_l_shoulder[2] > 0.5 and kp_r_shoulder[2] > 0.5 and 
            kp_l_hip[2] > 0.5 and kp_r_hip[2] > 0.5):
            
            # Trung điểm của vai và hông
            mid_shoulder = np.array([(kp_l_shoulder[0] + kp_r_shoulder[0]) / 2.0, 
                                     (kp_l_shoulder[1] + kp_r_shoulder[1]) / 2.0])
            mid_hip = np.array([(kp_l_hip[0] + kp_r_hip[0]) / 2.0, 
                                 (kp_l_hip[1] + kp_r_hip[1]) / 2.0])
            
            # Chiều dài thân người (khoảng cách vai - hông)
            body_vector = mid_hip - mid_shoulder
            body_length = np.linalg.norm(body_vector)
            
            # Tính góc giữa body_vector và trục dọc thẳng đứng [0, 1]
            # cos(theta) = dy / body_length
            dy = body_vector[1] # Tọa độ Y trong OpenCV hướng xuống dưới
            cos_theta = dy / max(body_length, 1.0)
            # Clip giá trị cos trong khoảng [-1, 1] để tránh lỗi lượng giác
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            angle_rad = math.acos(cos_theta)
            angle_deg = math.degrees(angle_rad)
            metrics["body_angle"] = angle_deg

            # 3. Tính Vận tốc rơi của hông (Hip Velocity)
            # Trọng tâm hông (Y coordinate)
            hip_y = mid_hip[1]
            self.hip_y_history.append(hip_y)
            self.body_length_history.append(body_length)
            
            if len(self.hip_y_history) > self.history_len:
                self.hip_y_history.pop(0)
                self.body_length_history.pop(0)
                
            # Tính vận tốc rơi bằng hiệu tọa độ Y hiện tại và trước đó, chuẩn hóa theo chiều dài thân người
            if len(self.hip_y_history) >= 2:
                # dy_hip = hip_y_t - hip_y_{t-1}
                dy_hip = self.hip_y_history[-1] - self.hip_y_history[-2]
                # Chuẩn hóa vận tốc theo chiều dài thân trung bình để loại bỏ yếu tố khoảng cách xa/gần
                avg_body_length = np.mean(self.body_length_history)
                normalized_velocity = dy_hip / max(avg_body_length, 1.0)
                metrics["hip_velocity"] = normalized_velocity
            
            metrics["is_valid"] = True
        else:
            # Nếu keypoints quan trọng bị che khuất, thử dùng Bounding Box thay thế để ước lượng góc
            # Mặc dù không chính xác bằng keypoints nhưng hữu ích khi bị che khuất
            metrics["is_valid"] = False

        return metrics

    def detect(self, keypoints, bbox):
        """
        Kiểm tra xem đối tượng có đang bị té ngã hay không.
        
        Args:
            keypoints (np.ndarray): Mảng keypoints (17, 3).
            bbox (list): [x1, y1, x2, y2].
            
        Returns:
            is_fall (bool): True nếu phát hiện té ngã, False nếu bình thường.
            metrics (dict): Các chỉ số tính toán được để hiển thị lên màn hình.
        """
        metrics = self.calculate_metrics(keypoints, bbox)
        
        if not metrics["is_valid"]:
            # Nếu không trích xuất đủ keypoints, dựa hoàn toàn vào Bounding Box Aspect Ratio làm dự phòng
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                aspect_ratio = (x2 - x1) / max(y2 - y1, 1.0)
                if aspect_ratio > 1.2:  # BBox nằm ngang rõ rệt
                    self.fall_counter += 1
                else:
                    self.fall_counter = max(0, self.fall_counter - 1)
            else:
                self.fall_counter = max(0, self.fall_counter - 1)
        else:
            # Điều kiện kết hợp các chỉ số:
            # 1. Góc nghiêng thân người lớn (> angle_threshold)
            # 2. Bounding Box nghiêng hoặc nằm ngang (aspect_ratio > aspect_ratio_threshold)
            # 3. Hoặc có một cú rơi nhanh đột ngột (hip_velocity > velocity_threshold)
            
            is_lying = (metrics["body_angle"] > self.angle_threshold) and (metrics["aspect_ratio"] > self.aspect_ratio_threshold)
            is_falling_fast = metrics["hip_velocity"] > self.velocity_threshold
            
            if is_lying or is_falling_fast:
                self.fall_counter += 1
            else:
                # Giảm dần counter thay vì đặt về 0 ngay để giảm hiện tượng chớp tắt trạng thái
                self.fall_counter = max(0, self.fall_counter - 1)

        # Xác nhận té ngã nếu trạng thái bất thường duy trì đủ số khung hình
        is_fall = self.fall_counter >= self.fall_confirmation_frames
        return is_fall, metrics
