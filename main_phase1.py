import cv2
import argparse
import sys
import os

# Thêm thư mục gốc vào PYTHONPATH để nhận diện src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pose.yolo_extractor import YOLOPoseExtractor
from src.utils.fps import FPSCalculator

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Real-time Pose Estimation using YOLOv8-Pose")
    parser.add_argument("--source", type=str, default="0", 
                        help="Nguồn video: '0' cho webcam, hoặc đường dẫn tới file video (.mp4, .avi, ...)")
    parser.add_argument("--device", type=str, default="cpu", 
                        help="Thiết bị chạy model: 'cpu' hoặc 'cuda'")
    parser.add_argument("--conf", type=float, default=0.5, 
                        help="Ngưỡng tin cậy phát hiện người (confidence threshold)")
    args = parser.parse_args()

    # 1. Khởi tạo nguồn video
    # Kiểm tra nếu là số thì chuyển về kiểu int để đọc webcam
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

    # 2. Khởi tạo YOLOv8-Pose Extractor
    print(f"[*] Đang tải mô hình YOLOv8-Pose trên thiết bị: {args.device}...")
    try:
        extractor = YOLOPoseExtractor(model_path="yolov8n-pose.pt", conf_threshold=args.conf, device=args.device)
        print("[+] Tải mô hình thành công!")
    except Exception as e:
        print(f"[!] Lỗi tải mô hình: {e}")
        sys.exit(1)

    # 3. Khởi tạo bộ đo FPS
    fps_calc = FPSCalculator()

    print("[*] Bắt đầu luồng xử lý. Nhấn 'q' hoặc 'Esc' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] Kết thúc luồng video hoặc không đọc được frame.")
            break

        # Để hiển thị mượt mà hơn, lật ảnh nếu dùng webcam
        if isinstance(source_val, int):
            frame = cv2.flip(frame, 1)

        # Trích xuất keypoints
        keypoints, bbox, _ = extractor.extract_pose(frame)

        # Vẽ khung xương và bounding box lên ảnh
        if keypoints is not None:
            extractor.draw_skeleton(frame, keypoints, bbox, label="Tracking...")
        else:
            # Hiển thị trạng thái không có người
            cv2.putText(frame, "No person detected", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Cập nhật và hiển thị FPS
        fps = fps_calc.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hiển thị kết quả ra cửa sổ OpenCV
        cv2.imshow("Phase 1: Real-time Pose Estimation", frame)

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
