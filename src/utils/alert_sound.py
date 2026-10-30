"""Hàm phát âm thanh cảnh báo dùng chung cho cả CLI (main_phase4.py) và GUI (app.py)."""
import platform


def try_beep():
    """Phát âm thanh cảnh báo. Chỉ hỗ trợ đầy đủ trên Windows (winsound)."""
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1500, 400)
        else:
            print("\a", end="", flush=True)  # bell character - fallback Linux/Mac
    except Exception:
        pass
