"""
Phase 4: Tích hợp mô hình LSTM vào hệ thống phát hiện té ngã thời gian thực.
 
Kiến trúc HYBRID 2 TẦNG:
    Tầng 1 (LSTM)       - phát hiện HÀNH ĐỘNG ngã đang diễn ra (chuyển động
                          đứng -> mất thăng bằng -> va chạm). Phản ứng nhanh,
                          độ tin cậy cao khi bắt được đúng khoảnh khắc.
    Tầng 2 (Safety Net) - phát hiện TRẠNG THÁI nằm bất động kéo dài, bù cho
                          giới hạn của LSTM khi hệ thống bắt đầu giám sát
                          SAU KHI người đã ngã (không có transition để học).
 
Pipeline: Webcam/Video -> YOLOPoseExtractor -> [normalize_keypoints -> LSTM]
                                             -> [StaticImmobilitySafetyNet]
                                             -> Cảnh báo (2 nguồn độc lập)
 
Dùng lại các module sẵn có của dự án:
    - src/pose/yolo_extractor.py       (YOLOPoseExtractor)
    - src/utils/normalization.py       (normalize_keypoints)
    - src/utils/fps.py                 (FPSCalculator)
    - src/models/lstm.py               (FallDetectionLSTM)
    - src/fall_detection/heuristics.py (FallHeuristics - dùng qua safety_net)
 
Cách chạy (chạy từ thư mục gốc fall_detection_system/):
    python main_phase4.py --source 0                      # webcam mặc định
    python main_phase4.py --source path/to/video.mp4       # file video
"""
import argparse
import sys
import time
 
import cv2
import numpy as np
 
from src.pose.yolo_extractor import YOLOPoseExtractor
from src.utils.normalization import normalize_keypoints
from src.utils.fps import FPSCalculator
from src.utils.alert_sound import try_beep
from src.fall_detection.lstm_classifier import FallLSTMPredictor
from src.fall_detection.safety_net import StaticImmobilitySafetyNet
 
COLOR_ALERT = (0, 0, 255)        # đỏ (BGR) - LSTM: hành động ngã
COLOR_ALERT_STATIC = (0, 140, 255)  # cam-đỏ (BGR) - Safety net: nằm bất động
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 - Real-time Fall Detection")
    parser.add_argument("--source", type=str, default="0",
                         help="'0' cho webcam mặc định, hoặc đường dẫn file video")
    parser.add_argument("--yolo-weights", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--lstm-weights", type=str, default="best_fall_lstm.pth")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--consecutive-frames", type=int, default=5,
                         help="Số frame liên tiếp vượt ngưỡng để kích hoạt cảnh báo")
    parser.add_argument("--cooldown", type=float, default=3.0,
                         help="Thời gian (giây) khóa cảnh báo sau khi đã báo động")
    parser.add_argument("--yolo-conf", type=float, default=0.5,
                         help="Ngưỡng confidence detect người (khớp lúc training)")
    parser.add_argument("--device", type=str, default="cpu",
                         help="'cpu', 'cuda', hoặc '0' cho GPU index (dùng cho YOLO)")
    parser.add_argument("--no-sound", action="store_true", help="Tắt âm thanh cảnh báo")
    parser.add_argument("--static-duration", type=float, default=4.0,
                         help="Số giây nằm bất động liên tục để Safety Net xác nhận cảnh báo")
    parser.add_argument("--disable-safety-net", action="store_true",
                         help="Tắt tầng bảo vệ thứ 2 (chỉ dùng LSTM)")
    return parser.parse_args()
 
 
def main():
    args = parse_args()
 
    # Nguồn video: '0' -> webcam index 0, ngược lại coi là đường dẫn file
    source = int(args.source) if args.source.isdigit() else args.source
 
    print(f"[*] Đang tải YOLOv8-Pose ({args.yolo_weights})...")
    extractor = YOLOPoseExtractor(model_path=args.yolo_weights,
                                   conf_threshold=args.yolo_conf,
                                   device=args.device)
 
    print(f"[*] Đang tải LSTM classifier ({args.lstm_weights})...")
    predictor = FallLSTMPredictor(
        weight_path=args.lstm_weights,
        sequence_length=30,
        input_size=34,
        threshold=args.threshold,
        consecutive_frames_required=args.consecutive_frames,
        cooldown_seconds=args.cooldown,
    )
    safety_net = None
    if not args.disable_safety_net:
        safety_net = StaticImmobilitySafetyNet(min_duration_seconds=args.static_duration)
        print(f"[+] Safety Net bật: cảnh báo nếu nằm bất động >= {args.static_duration}s")
 
    print(f"[+] Sẵn sàng. Device: {predictor.device}, threshold={args.threshold}")
 
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Lỗi: Không mở được nguồn video '{args.source}'")
        sys.exit(1)
 
    fps_calc = FPSCalculator(buffer_size=30)
    alert_display_until = 0.0    # banner đỏ (LSTM) hiển thị đến khi nào
    static_display_until = 0.0   # banner cam (Safety Net) hiển thị đến khi nào
 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[*] Hết video hoặc mất kết nối camera.")
            break
 
        kps, bbox, _ = extractor.extract_pose(frame)
        prob_text = "N/A"
        static_duration_text = "0.0s"
 
        if kps is not None:
            norm_kps = normalize_keypoints(kps, bbox)
            extractor.draw_skeleton(frame, kps, bbox, label="Person")
        else:
            # Không detect được người -> vector 0, giữ tính liên tục thời gian
            # (khớp với logic chuẩn bị dataset lúc training)
            norm_kps = np.zeros(34, dtype=np.float32)
 
        # --- Tầng 1: LSTM (phát hiện hành động ngã) ---
        result = predictor.update(norm_kps)
        if result["ready"]:
            prob_text = f"{result['probability']:.2f}"
            if result["alert"]:
                alert_display_until = time.time() + 2.0
                if not args.no_sound:
                    try_beep()
                print(f"[!] [LSTM] CẢNH BÁO TÉ NGÃ! probability={result['probability']:.3f}")
 
        # --- Tầng 2: Safety Net (phát hiện nằm bất động kéo dài) ---
        if safety_net is not None:
            static_result = safety_net.update(kps, bbox)
            static_duration_text = f"{static_result['duration']:.1f}s"
            if static_result["alert"]:
                static_display_until = time.time() + 2.0
                if not args.no_sound:
                    try_beep()
                print(f"[!] [SAFETY NET] NẰM BẤT ĐỘNG {static_result['duration']:.1f}s - Cần kiểm tra!")
 
        # --- Hiển thị banner: ưu tiên LSTM (khẩn cấp hơn) nếu cả 2 cùng active ---
        now = time.time()
        if now < alert_display_until:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), COLOR_ALERT, 8)
            cv2.putText(frame, "CANH BAO TE NGA!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_ALERT, 3)
        elif now < static_display_until:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), COLOR_ALERT_STATIC, 8)
            cv2.putText(frame, "NGHI NGO BAT DONG - KIEM TRA NGAY", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ALERT_STATIC, 3)
 
        fps = fps_calc.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"P(fall): {prob_text}", (10, frame.shape[0] - 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Static: {static_duration_text}", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
        cv2.imshow("Fall Detection - Phase 4", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()