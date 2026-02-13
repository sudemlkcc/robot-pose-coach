"""
Pose Detector: Wrapper around MediaPipe Pose for real-time human pose detection.

Handles initialization, inference, and landmark extraction with confidence scores.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class PoseDetector:
    """
    Real-time pose detection using MediaPipe.

    Provides a clean interface for pose detection with frame-by-frame inference.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the pose detector.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose # type: ignore
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils # type: ignore
        self.mp_drawing_styles = mp.solutions.drawing_styles # type: ignore

        logger.info("PoseDetector initialized with MediaPipe Pose")

    def detect(self, frame: np.ndarray) -> Tuple[bool, List, float]:
        """
        Detect pose in a single frame.

        Args:
            frame: Input BGR image from OpenCV

        Returns:
            Tuple of (success: bool, landmarks: list, confidence: float)
            - success: True if pose was detected
            - landmarks: List of 33 landmarks, each [x, y, z, visibility]
            - confidence: Average visibility score of detected landmarks
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is None:
            return False, [], 0.0

        landmarks = []
        confidences = []

        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
            confidences.append(landmark.visibility)

        avg_confidence = float(np.mean(confidences))

        return True, landmarks, avg_confidence

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: List,
        confidence_threshold: float = 0.5,
        skeleton_color: Tuple = (0, 255, 0),
        joint_color: Tuple = (0, 0, 255),
        skeleton_thickness: int = 2,
        joint_radius: int = 5,
    ) -> np.ndarray:
        """
        Draw pose skeleton on frame.

        Args:
            frame: Input image (modified in-place)
            landmarks: List of pose landmarks
            confidence_threshold: Only draw landmarks with visibility above this
            skeleton_color: BGR color for skeleton lines
            joint_color: BGR color for joint circles
            skeleton_thickness: Line thickness
            joint_radius: Joint circle radius

        Returns:
            Frame with drawn skeleton
        """
        if not landmarks:
            return frame

        h, w = frame.shape[:2]

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
            (11, 24), (12, 23),
        ]

        for connection in connections:
            idx_a, idx_b = connection
            if idx_a >= len(landmarks) or idx_b >= len(landmarks):
                continue

            point_a = landmarks[idx_a]
            point_b = landmarks[idx_b]

            if len(point_a) < 3 or len(point_b) < 3:
                continue

            visibility_a = point_a[3] if len(point_a) > 3 else 1.0
            visibility_b = point_b[3] if len(point_b) > 3 else 1.0

            if visibility_a < confidence_threshold or visibility_b < confidence_threshold:
                continue

            x_a, y_a = int(point_a[0] * w), int(point_a[1] * h)
            x_b, y_b = int(point_b[0] * w), int(point_b[1] * h)

            cv2.line(frame, (x_a, y_a), (x_b, y_b), skeleton_color, skeleton_thickness)

        for idx, landmark in enumerate(landmarks):
            if len(landmark) < 2:
                continue

            visibility = landmark[3] if len(landmark) > 3 else 1.0
            if visibility < confidence_threshold:
                continue

            x = int(landmark[0] * w)
            y = int(landmark[1] * h)

            cv2.circle(frame, (x, y), joint_radius, joint_color, -1)

        return frame

    def get_landmark_coordinates(
        self, landmarks: List, index: int, frame_shape: Tuple
    ) -> Tuple[int, int]:
        """
        Convert normalized landmark coordinates to pixel coordinates.

        Args:
            landmarks: List of landmarks
            index: Landmark index
            frame_shape: Frame shape (height, width, channels)

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        if index >= len(landmarks):
            return 0, 0

        h, w = frame_shape[:2]
        landmark = landmarks[index]
        x = int(landmark[0] * w)
        y = int(landmark[1] * h)
        return x, y

    def release(self):
        """Clean up resources."""
        if self.pose:
            self.pose.close()
        logger.info("PoseDetector released")