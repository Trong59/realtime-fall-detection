import numpy as np

def normalize_keypoints(keypoints, bbox=None):
    """
    Chuẩn hóa 17 điểm keypoints của YOLOv8-pose.
    1. Dịch chuyển gốc tọa độ về trung điểm của hông (mid_hip).
    2. Chia tỷ lệ cho chiều dài thân người (khoảng cách mid_shoulder đến mid_hip) 
       hoặc chiều cao bounding box để loại bỏ ảnh hưởng của khoảng cách xa/gần camera.
    
    Args:
        keypoints (np.ndarray): Mảng shape (17, 3) chứa [x, y, conf] từ YOLO.
        bbox (list, optional): [x1, y1, x2, y2] dùng làm phương án dự phòng chuẩn hóa.
        
    Returns:
        flat_kps (np.ndarray): Mảng phẳng shape (34,) chứa [x0, y0, x1, y1, ...] đã chuẩn hóa.
                              Nếu không đủ keypoints chính xác, trả về mảng toàn 0.
    """
    # Khởi tạo mảng kết quả mặc định
    default_output = np.zeros(34, dtype=np.float32)
    
    if keypoints is None:
        return default_output

    kp_l_shoulder = keypoints[5]
    kp_r_shoulder = keypoints[6]
    kp_l_hip = keypoints[11]
    kp_r_hip = keypoints[12]

    # Kiểm tra độ tin cậy của các điểm vai và hông
    if (kp_l_shoulder[2] > 0.5 and kp_r_shoulder[2] > 0.5 and 
        kp_l_hip[2] > 0.5 and kp_r_hip[2] > 0.5):
        
        # 1. Tính trung điểm vai và hông
        mid_shoulder = np.array([(kp_l_shoulder[0] + kp_r_shoulder[0]) / 2.0, 
                                 (kp_l_shoulder[1] + kp_r_shoulder[1]) / 2.0])
        mid_hip = np.array([(kp_l_hip[0] + kp_r_hip[0]) / 2.0, 
                             (kp_l_hip[1] + kp_r_hip[1]) / 2.0])
        
        # 2. Chiều dài thân người
        body_length = np.linalg.norm(mid_hip - mid_shoulder)
        if body_length < 1.0:
            body_length = 1.0

        # 3. Tiến hành dịch chuyển gốc tọa độ về mid_hip và chia cho body_length
        normalized = []
        for kp in keypoints:
            x, y, _ = kp
            # Dịch tâm về mid_hip
            norm_x = (x - mid_hip[0]) / body_length
            norm_y = (y - mid_hip[1]) / body_length
            normalized.extend([norm_x, norm_y])
            
        return np.array(normalized, dtype=np.float32)
        
    elif bbox is not None:
        # Phương án dự phòng: Chuẩn hóa theo bounding box nếu mất các khớp vai/hông chính
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = max(y2 - y1, 1.0)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        
        # Chuẩn hóa theo kích thước bounding box
        normalized = []
        for kp in keypoints:
            x, y, _ = kp
            norm_x = (x - center_x) / max(width, 1.0)
            norm_y = (y - center_y) / height
            normalized.extend([norm_x, norm_y])
            
        return np.array(normalized, dtype=np.float32)
        
    else:
        # Không thể chuẩn hóa, trả về mảng 0
        return default_output
