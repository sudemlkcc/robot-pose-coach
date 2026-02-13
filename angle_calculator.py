"""
Angle Calculator: Computes joint angles and posture metrics from pose landmarks.

This module provides utilities to calculate angles between joints and derive
posture-related metrics from MediaPipe pose landmarks. All angles are in degrees.

Landmark Reference (MediaPipe Pose):
  0: nose, 11: left shoulder, 12: right shoulder, 13: left elbow,
  14: right elbow, 15: left wrist, 16: right wrist, 23: left hip,
  24: right hip, 25: left knee, 26: right knee, 27: left ankle, 28: right ankle
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AngleCalculator:
    """Utility class for calculating angles and posture metrics."""

    @staticmethod
    def angle_between_points(
        point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray
    ) -> float:
        """
        Calculate the angle at point B formed by points A-B-C.

        Args:
            point_a: First point [x, y, z]
            point_b: Vertex point [x, y, z]
            point_c: Third point [x, y, z]

        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        a = np.array(point_a)
        b = np.array(point_b)
        c = np.array(point_c)

        # Vectors from B to A and B to C
        ba = a - b
        bc = c - b

        # Calculate angle using dot product
        cos_angle = np.dot(ba, bc) / (
            np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10
        )
        # Clamp to [-1, 1] to avoid numerical errors in arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    @staticmethod
    def left_elbow_angle(landmarks: list) -> float:
        """Calculate left elbow angle (shoulder-elbow-wrist)."""
        if len(landmarks) < 16:
            return 0.0
        # Landmark indices: 11=shoulder, 13=elbow, 15=wrist
        return AngleCalculator.angle_between_points(
            landmarks[11], landmarks[13], landmarks[15]
        )

    @staticmethod
    def right_elbow_angle(landmarks: list) -> float:
        """Calculate right elbow angle (shoulder-elbow-wrist)."""
        if len(landmarks) < 16:
            return 0.0
        # Landmark indices: 12=shoulder, 14=elbow, 16=wrist
        return AngleCalculator.angle_between_points(
            landmarks[12], landmarks[14], landmarks[16]
        )

    @staticmethod
    def left_shoulder_angle(landmarks: list) -> float:
        """Calculate left shoulder angle (hip-shoulder-elbow)."""
        if len(landmarks) < 14:
            return 0.0
        # Landmark indices: 23=hip, 11=shoulder, 13=elbow
        return AngleCalculator.angle_between_points(
            landmarks[23], landmarks[11], landmarks[13]
        )

    @staticmethod
    def right_shoulder_angle(landmarks: list) -> float:
        """Calculate right shoulder angle (hip-shoulder-elbow)."""
        if len(landmarks) < 15:
            return 0.0
        # Landmark indices: 24=hip, 12=shoulder, 14=elbow
        return AngleCalculator.angle_between_points(
            landmarks[24], landmarks[12], landmarks[14]
        )

    @staticmethod
    def left_knee_angle(landmarks: list) -> float:
        """Calculate left knee angle (hip-knee-ankle)."""
        if len(landmarks) < 27:
            return 0.0
        # Landmark indices: 23=hip, 25=knee, 27=ankle
        return AngleCalculator.angle_between_points(
            landmarks[23], landmarks[25], landmarks[27]
        )

    @staticmethod
    def right_knee_angle(landmarks: list) -> float:
        """Calculate right knee angle (hip-knee-ankle)."""
        if len(landmarks) < 28:
            return 0.0
        # Landmark indices: 24=hip, 26=knee, 28=ankle
        return AngleCalculator.angle_between_points(
            landmarks[24], landmarks[26], landmarks[28]
        )

    @staticmethod
    def left_hip_angle(landmarks: list) -> float:
        """Calculate left hip angle (shoulder-hip-knee)."""
        if len(landmarks) < 25:
            return 0.0
        # Landmark indices: 11=shoulder, 23=hip, 25=knee
        return AngleCalculator.angle_between_points(
            landmarks[11], landmarks[23], landmarks[25]
        )

    @staticmethod
    def right_hip_angle(landmarks: list) -> float:
        """Calculate right hip angle (shoulder-hip-knee)."""
        if len(landmarks) < 26:
            return 0.0
        # Landmark indices: 12=shoulder, 24=hip, 26=knee
        return AngleCalculator.angle_between_points(
            landmarks[12], landmarks[24], landmarks[26]
        )

    @staticmethod
    def spine_inclination(landmarks: list) -> float:
        """
        Calculate spine inclination (how much the torso leans forward).

        Uses the vector from middle hip to middle shoulder to estimate
        spine angle. 0 degrees = vertical, 90 degrees = horizontal.

        Returns:
            Inclination angle in degrees (0-90, higher = more leaning forward)
        """
        if len(landmarks) < 24:
            return 0.0

        # Get middle points
        left_shoulder = np.array(landmarks[11][:2])  # x, y only
        right_shoulder = np.array(landmarks[12][:2])
        left_hip = np.array(landmarks[23][:2])
        right_hip = np.array(landmarks[24][:2])

        # Calculate midpoints
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2

        # Vector from hip to shoulder
        spine_vector = shoulder_mid - hip_mid

        # Angle from vertical (upward is 0 degrees)
        angle_from_vertical = np.degrees(np.arctan2(spine_vector[0], spine_vector[1]))

        # Absolute inclination
        inclination = abs(angle_from_vertical)

        return float(min(inclination, 90.0))

    @staticmethod
    def symmetry_difference(left_angle: float, right_angle: float) -> float:
        """
        Calculate the absolute difference between left and right angles.

        Useful for detecting asymmetrical movements (e.g., one arm higher than the other).

        Args:
            left_angle: Left side joint angle
            right_angle: Right side joint angle

        Returns:
            Absolute difference in degrees
        """
        return abs(left_angle - right_angle)

    @staticmethod
    def extract_all_angles(landmarks: list) -> dict:
        """
        Extract all relevant joint angles from landmarks.

        Returns a dictionary of all calculated angles for easier feature extraction.

        Args:
            landmarks: List of MediaPipe landmarks

        Returns:
            Dictionary with angle names and values
        """
        return {
            "left_elbow": AngleCalculator.left_elbow_angle(landmarks),
            "right_elbow": AngleCalculator.right_elbow_angle(landmarks),
            "left_shoulder": AngleCalculator.left_shoulder_angle(landmarks),
            "right_shoulder": AngleCalculator.right_shoulder_angle(landmarks),
            "left_knee": AngleCalculator.left_knee_angle(landmarks),
            "right_knee": AngleCalculator.right_knee_angle(landmarks),
            "left_hip": AngleCalculator.left_hip_angle(landmarks),
            "right_hip": AngleCalculator.right_hip_angle(landmarks),
            "spine_inclination": AngleCalculator.spine_inclination(landmarks),
        }

    @staticmethod
    def smooth_angle(
        current_angle: float, previous_angle: Optional[float], alpha: float = 0.3
    ) -> float:
        """
        Apply exponential moving average smoothing to reduce jitter.

        Args:
            current_angle: Current angle measurement
            previous_angle: Previous angle measurement (None for first frame)
            alpha: Smoothing factor (0-1, lower = more smoothing)

        Returns:
            Smoothed angle
        """
        if previous_angle is None:
            return current_angle
        return alpha * current_angle + (1 - alpha) * previous_angle
