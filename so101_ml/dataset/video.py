"""Encode and decode RGB episode videos with OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError("Cannot write an empty video")
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                raise ValueError(f"Frame size {frame.shape[:2]} != {(height, width)}")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def read_mp4(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames in {path}")
    return np.stack(frames, axis=0)
