"""
pose_detector.py — Thin wrapper around MediaPipe Pose.

Why a wrapper?
  • Isolates the MediaPipe dependency so swapping to a different pose
    estimator (e.g., MoveNet, robot-side ONNX model) only requires
    changing this one file.
  • Converts raw MediaPipe landmarks into a clean dict of
    {landmark_name: (x, y, z, visibility)} for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

import config

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


@dataclass
class LandmarkPoint:
    """A single detected body landmark in normalised coordinates."""
    x: float           # 0..1 (fraction of frame width)
    y: float           # 0..1 (fraction of frame height)
    z: float           # Depth relative to hip midpoint
    visibility: float  # 0..1 confidence

    def to_pixel(self, w: int, h: int) -> Tuple[int, int]:
        """Convert normalised coords to pixel coords."""
        return int(self.x * w), int(self.y * h)

    def to_array(self) -> np.ndarray:
        """Return [x, y, z] as numpy array (useful for angle math)."""
        return np.array([self.x, self.y, self.z])


# Type alias for the full set of landmarks keyed by name.
Landmarks = Dict[str, LandmarkPoint]


class PoseDetector:
    """
    Real-time pose detector backed by MediaPipe Pose.

    Usage:
        detector = PoseDetector()
        landmarks = detector.process(bgr_frame)
        if landmarks:
            print(landmarks["LEFT_SHOULDER"].x)
        detector.close()
    """

    # Build a name→index lookup from the MediaPipe enum once.
    _NAME_TO_IDX: Dict[str, int] = {
        lm.name: lm.value for lm in mp_pose.PoseLandmark
    }

    def __init__(self) -> None:
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.POSE_MODEL_COMPLEXITY,
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            enable_segmentation=False,
        )

    # ── Public API ───────────────────────────────────────────────

    def process(self, bgr_frame: np.ndarray) -> Optional[Landmarks]:
        """
        Run pose estimation on a BGR frame.

        Returns:
            Dict mapping landmark name → LandmarkPoint, or None if no
            pose detected.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False          # Small perf gain
        results = self._pose.process(rgb)

        if results.pose_landmarks is None:
            return None

        landmarks: Landmarks = {}
        for name, idx in self._NAME_TO_IDX.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = LandmarkPoint(
                x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility,
            )
        return landmarks

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    # ── Context manager support ──────────────────────────────────

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()
