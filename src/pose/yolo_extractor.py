import cv2
import numpy as np
from ultralytics import YOLO

class YOLOPoseExtractor:
    def __init__(self, model_path="yolov8n-pose.pt", conf_threshold=0.5, device="cpu"):
        """
        Khởi tạo bộ trích xuất khung xương bằng YOLOv8-Pose.
        
        Args:
            model_path (str): Đường dẫn tới mô hình YOLOv8-Pose (mặc định tự tải yolov8n-pose.pt nếu chưa có).
            conf_threshold (float): Ngưỡng tin cậy của bounding box để phát hiện người.
            device (str): Thiết bị chạy mô hình ('cpu', 'cuda', hoặc số thứ tự GPU '0').
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        
        # Định nghĩa các đường nối khung xương của YOLOv8-pose (17 keypoints)
        # Các đường nối nối giữa các chỉ số keypoint
        self.SKELETON_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),             # Mặt
            (5, 6),                                     # Vai
            (5, 7), (7, 9),                             # Tay trái
            (6, 8), (8, 10),                            # Tay phải
            (5, 11), (6, 12), (11, 12),                 # Thân người
            (11, 13), (13, 15),                         # Chân trái
            (12, 14), (14, 16)                          # Chân phải
        ]
        
        # Màu sắc khi vẽ (BGR)
        self.KEYPOINT_COLOR = (0, 255, 0)   # Xanh lá cho các điểm
        self.LINE_COLOR = (255, 255, 0)     # Xanh lam nhạt cho đường nối
        self.BOX_COLOR = (0, 255, 255)      # Vàng cho Bounding Box

    def extract_pose(self, frame):
        """
        Dự đoán và trích xuất keypoints từ một khung hình.
        Nếu phát hiện nhiều người, trả về keypoints của người có bounding box lớn nhất (đối tượng chính).
        
        Args:
            frame (np.ndarray): Ảnh đầu vào (BGR OpenCV).
            
        Returns:
            keypoints (np.ndarray): Mảng shape (17, 3) gồm [x, y, confidence] của đối tượng chính.
                                   Nếu không phát hiện thấy ai, trả về None.
            bbox (list): Tọa độ [x1, y1, x2, y2] của đối tượng chính.
            all_results: Kết quả gốc từ mô hình YOLO để sử dụng cho việc vẽ hoặc phân tích sâu.
        """
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None, results
        
        # Lấy kết quả đầu tiên (thường chỉ xử lý 1 ảnh tại một thời điểm)
        result = results[0]
        boxes = result.boxes.cpu().numpy()
        keypoints_data = result.keypoints.data.cpu().numpy() # shape (N, 17, 3)
        
        if len(keypoints_data) == 0:
            return None, None, results
            
        # Tìm người có diện tích bounding box lớn nhất (đối tượng chính để theo dõi té ngã)
        max_area = 0
        best_idx = 0
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_idx = idx
                
        best_keypoints = keypoints_data[best_idx] # shape (17, 3)
        best_bbox = boxes[best_idx].xyxy[0].tolist() # [x1, y1, x2, y2]
        
        return best_keypoints, best_bbox, result

    def draw_skeleton(self, frame, keypoints, bbox=None, label="Person"):
        """
        Vẽ bounding box, keypoints và skeleton lên khung hình.
        
        Args:
            frame (np.ndarray): Khung hình ảnh gốc (sẽ bị vẽ đè trực tiếp).
            keypoints (np.ndarray): Mảng keypoints (17, 3).
            bbox (list): Tọa độ bounding box [x1, y1, x2, y2].
            label (str): Nhãn hiển thị trên bounding box.
        """
        h, w, _ = frame.shape
        
        # 1. Vẽ Bounding Box nếu có
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.BOX_COLOR, 2)
            
        if keypoints is None:
            return
            
        # 2. Vẽ các đường nối skeleton
        for start_idx, end_idx in self.SKELETON_CONNECTIONS:
            start_kp = keypoints[start_idx]
            end_kp = keypoints[end_idx]
            
            # Chỉ vẽ khi cả hai điểm đều có độ tin cậy lớn hơn 0.5
            if start_kp[2] > 0.5 and end_kp[2] > 0.5:
                pt1 = (int(start_kp[0]), int(start_kp[1]))
                pt2 = (int(end_kp[0]), int(end_kp[1]))
                cv2.line(frame, pt1, pt2, self.LINE_COLOR, 2)
                
        # 3. Vẽ các điểm keypoint tròn
        for idx, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, self.KEYPOINT_COLOR, -1)
                # Thỉnh thoảng ta có thể đánh số điểm để dễ debug
                # cv2.putText(frame, str(idx), (int(x)+5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
