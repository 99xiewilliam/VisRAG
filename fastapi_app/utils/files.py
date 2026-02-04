import os
import re
from typing import Optional
from fastapi import UploadFile


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def slugify(name: str, fallback: str = "item") -> str:
    text = (name or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def save_upload_file(upload: UploadFile, target_dir: str, filename: Optional[str] = None) -> str:
    ensure_dir(target_dir)
    safe_name = filename or upload.filename or "file"
    file_path = os.path.join(target_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(upload.file.read())
    upload.file.close()
    return file_path


def extract_frame_from_video(video_path: str, output_dir: str, frame_index: Optional[int] = None) -> str:
    """
    从视频中抽取一帧保存为图像，返回图像路径。
    frame_index: 第几帧，None 表示取中间帧。
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError(
            "Video frame extraction requires opencv-python. Install with: pip install opencv-python-headless"
        )
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 1
    if frame_index is None:
        frame_index = total // 2  # 中间帧
    frame_index = min(max(0, frame_index), total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise ValueError(f"Failed to read frame {frame_index} from video: {video_path}")
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{base}_frame{frame_index}.png")
    cv2.imwrite(out_path, frame)
    return out_path
