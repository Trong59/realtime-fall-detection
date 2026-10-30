import cv2
import argparse
import sys
import os

# Thêm thư mục gốc vào PYTHONPATH để nhận diện src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pose.yolo_extractor import YOLOPoseExtractor
from src.fall_detection.heuristics import FallHeuristics
from src.utils.fps import FPSCalculator

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fall Detection using Physical Heuristics (Rule-based)")
    parser.add_argument("--source", type=str, default="0", 
                        help="Nguồn video: '0' cho webcam, hoặc đường dẫn tới file video")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Thiết bị chạy model: 'cpu' hoặc 'cuda'")
    parser.add_argument("--conf", type=float, default=0.5, 
                        help="Ngưỡng tin cậy phát hiện người")
    args = parser.parse_args()

    # 1. Khởi tạo nguồn video
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)
        print(f"[*] Đang mở Webcam ID: {source_val}...")
    else:
        if not os.path.exists(source_val):
            print(f"[!] Lỗi: File video '{source_val}' không tồn tại.")
            sys.exit(1)
        print(f"[*] Đang mở file video: {source_val}...")

    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"[!] Lỗi: Không thể kết nối tới nguồn video: {args.source}")
        sys.exit(1)

    # 2. Khởi tạo YOLOv8-Pose
    print(f"[*] Đang tải mô hình YOLOv8-Pose trên thiết bị: {args.device}...")
    try:
        extractor = YOLOPoseExtractor(model_path="yolov8n-pose.pt", conf_threshold=args.conf, device=args.device)
        print("[+] Tải mô hình thành công!")
    except Exception as e:
        print(f"[!] Lỗi tải mô hình: {e}")
        sys.exit(1)

    # 3. Khởi tạo bộ phát hiện ngã bằng Heuristics
    # Điều chỉnh nhẹ ngưỡng góc thành 60 độ, tỷ lệ khung hình thành 0.85, ngưỡng vận tốc rơi 0.15
    detector = FallHeuristics(aspect_ratio_threshold=0.85, angle_threshold=60.0, velocity_threshold=0.15)
    
    # 4. Khởi tạo bộ đo FPS
    fps_calc = FPSCalculator()

    print("[*] Bắt đầu luồng xử lý. Nhấn 'q' hoặc 'Esc' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] Kết thúc luồng video hoặc không đọc được frame.")
            break

        # Lật ảnh nếu dùng webcam để hiển thị thuận mắt
        if isinstance(source_val, int):
            frame = cv2.flip(frame, 1)

        # Trích xuất keypoints
        keypoints, bbox, _ = extractor.extract_pose(frame)

        # Chạy thuật toán phát hiện ngã
        is_fall = False
        metrics = {}
        if keypoints is not None:
            is_fall, metrics = detector.detect(keypoints, bbox)

        # 5. Vẽ kết quả lên màn hình
        if keypoints is not None and bbox is not None:
            # Chọn màu vẽ dựa trên trạng thái ngã
            if is_fall:
                color = (0, 0, 255)  # Đỏ khi ngã
                label = "FALL DETECTED!"
                
                # Thêm hiệu ứng viền cảnh báo màu đỏ xung quanh toàn bộ màn hình
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[2] if len(frame.shape)>2 else frame.shape[0]), (0, 0, 255), 10)
            else:
                color = (0, 255, 0)  # Xanh lá khi bình thường
                label = "Normal"

            # Vẽ bounding box và skeleton tùy chỉnh màu sắc trạng thái
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Vẽ các đường nối xương
            for start_idx, end_idx in extractor.SKELETON_CONNECTIONS:
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]
                if start_kp[2] > 0.5 and end_kp[2] > 0.5:
                    pt1 = (int(start_kp[0]), int(start_kp[1]))
                    pt2 = (int(end_kp[0]), int(end_kp[1]))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
            
            # Vẽ các khớp tròn
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, color, -1)

            # Hiển thị các chỉ số debug thời gian thực lên góc màn hình
            if metrics.get("is_valid", False):
                cv2.putText(frame, f"Aspect Ratio: {metrics['aspect_ratio']:.2f}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Body Angle: {metrics['body_angle']:.1f} deg", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Hip Velocity: {metrics['hip_velocity']:.3f}", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Fall Counter: {detector.fall_counter}/{detector.fall_confirmation_frames}", (20, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No person detected", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Cập nhật và hiển thị FPS
        fps = fps_calc.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hiển thị kết quả ra cửa sổ OpenCV
        cv2.imshow("Phase 2: Fall Detection (Heuristics)", frame)

        # Nhấn q hoặc Esc để thoát
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print("[*] Đã đóng hệ thống.")

if __name__ == "__main__":
    main()
