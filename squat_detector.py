"""
Squat Detector: Specialized detector for squat exercises.

Tracks squat depth, form, and provides squat-specific feedback.
Extensible for later ML-based classification.
"""

from typing import Tuple, Dict, List
import logging
from angle_calculator import AngleCalculator
from feedback_generator import FeedbackGenerator, Feedback, FeedbackLevel
import config

logger = logging.getLogger(__name__)


class SquatDetector:
    """
    Detects and analyzes squat exercises.

    Provides:
    - Squat depth detection
    - Form evaluation
    - Real-time feedback
    """

    def __init__(self):
        """Initialize the squat detector."""
        self.feedback_generator = FeedbackGenerator()
        self.squat_in_progress = False
        self.max_depth_reached = 180  # Initialize to full extension
        logger.info("SquatDetector initialized")

    def analyze_squat_form(self, landmarks: List) -> Tuple[Dict, List[Feedback]]:
        """
        Analyze squat form from landmarks.

        Args:
            landmarks: List of pose landmarks

        Returns:
            Tuple of (metrics_dict, feedback_list)
            - metrics_dict: Contains all calculated angles
            - feedback_list: List of Feedback objects
        """
        # Extract all angles
        angles = AngleCalculator.extract_all_angles(landmarks)

        # Get specific angles for squat analysis
        left_knee = angles["left_knee"]
        right_knee = angles["right_knee"]
        left_hip = angles["left_hip"]
        right_hip = angles["right_hip"]
        spine_inclination = angles["spine_inclination"]

        # Update max depth (minimum knee angle during squat)
        avg_knee = (left_knee + right_knee) / 2
        if avg_knee < self.max_depth_reached:
            self.max_depth_reached = avg_knee

        # Determine if person is in a squat position
        in_squat = avg_knee < 120  # Roughly 120 degrees indicates active squat

        # Build metrics dictionary
        metrics = {
            "left_knee": left_knee,
            "right_knee": right_knee,
            "avg_knee": avg_knee,
            "left_hip": left_hip,
            "right_hip": right_hip,
            "spine_inclination": spine_inclination,
            "in_squat": in_squat,
            "max_depth_reached": self.max_depth_reached,
        }

        # Generate feedback using thresholds from config
        thresholds = {
            "knee_min": config.SQUAT_KNEE_ANGLE_MIN,
            "knee_max": config.SQUAT_KNEE_ANGLE_MAX,
            "back_max": config.SQUAT_BACK_INCLINATION_MAX,
        }

        feedback_list = self.feedback_generator.evaluate_squat(
            left_knee, right_knee, left_hip, right_hip, spine_inclination, thresholds
        )

        return metrics, feedback_list

    def is_squat_complete(self) -> bool:
        """
        Detect if a squat rep has been completed.

        A complete squat involves:
        1. Starting from standing (knee ~180°)
        2. Descending to depth (knee < threshold)
        3. Returning to standing

        Returns:
            True if squat appears complete
        """
        # This is a simplified check. In production, you'd track state machine:
        # standing -> descending -> bottom -> ascending -> standing
        return self.max_depth_reached < config.SQUAT_KNEE_ANGLE_MIN

    def reset_squat(self):
        """Reset tracking for next squat rep."""
        self.max_depth_reached = 180
        logger.debug("Squat tracking reset")

    def get_squat_depth_percentage(self) -> float:
        """
        Calculate squat depth as a percentage.

        100% = full depth achieved (knee angle at min threshold)
        0% = no depth (standing position)

        Returns:
            Percentage (0-100)
        """
        # Full extension to max depth range
        full_range = 180 - config.SQUAT_KNEE_ANGLE_MIN
        current_depth = 180 - self.max_depth_reached

        if full_range <= 0:
            return 0.0

        depth_percentage = (current_depth / full_range) * 100
        return max(0, min(100, depth_percentage))  # Clamp to 0-100