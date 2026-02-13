"""
Arm Raise Detector: Specialized detector for arm raise exercises.

Tracks arm height, elbow straightness, and symmetry.
Extensible for later ML-based classification.
"""

from typing import Tuple, Dict, List
import logging
from angle_calculator import AngleCalculator
from feedback_generator import FeedbackGenerator, Feedback
import config

logger = logging.getLogger(__name__)


class ArmRaiseDetector:
    """
    Detects and analyzes arm raise exercises.

    Provides:
    - Arm height tracking
    - Elbow straightness checking
    - Symmetry detection
    - Real-time feedback
    """

    def __init__(self):
        """Initialize the arm raise detector."""
        self.feedback_generator = FeedbackGenerator()
        self.max_height_reached = {
            "left": 0,
            "right": 0,
        }
        logger.info("ArmRaiseDetector initialized")

    def analyze_arm_raise_form(self, landmarks: List) -> Tuple[Dict, List[Feedback]]:
        """
        Analyze arm raise form from landmarks.

        Args:
            landmarks: List of pose landmarks

        Returns:
            Tuple of (metrics_dict, feedback_list)
            - metrics_dict: Contains all calculated angles
            - feedback_list: List of Feedback objects
        """
        # Extract all angles
        angles = AngleCalculator.extract_all_angles(landmarks)

        # Get specific angles for arm raise
        left_shoulder = angles["left_shoulder"]
        right_shoulder = angles["right_shoulder"]
        left_elbow = angles["left_elbow"]
        right_elbow = angles["right_elbow"]

        # Track max height reached
        self.max_height_reached["left"] = max(self.max_height_reached["left"], left_shoulder)
        self.max_height_reached["right"] = max(self.max_height_reached["right"], right_shoulder)

        # Detect if arms are currently raised
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        arms_raised = avg_shoulder > 160

        # Build metrics dictionary
        metrics = {
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "avg_shoulder": avg_shoulder,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "arms_raised": arms_raised,
            "max_left_height": self.max_height_reached["left"],
            "max_right_height": self.max_height_reached["right"],
        }

        # Generate feedback using thresholds from config
        thresholds = {
            "shoulder_min": config.ARM_RAISE_SHOULDER_MIN,
            "elbow_variance_max": config.ARM_RAISE_ELBOW_STRAIGHTNESS,
            "symmetry_tolerance": config.ARM_RAISE_SYMMETRY_TOLERANCE,
        }

        feedback_list = self.feedback_generator.evaluate_arm_raise(
            left_shoulder, right_shoulder, left_elbow, right_elbow, thresholds
        )

        return metrics, feedback_list

    def is_arm_raise_complete(self) -> bool:
        """
        Detect if an arm raise rep has been completed.

        Returns:
            True if arms were raised to sufficient height and lowered back
        """
        # A rep is complete if max height was reached above threshold
        avg_max_height = (
            self.max_height_reached["left"] + self.max_height_reached["right"]
        ) / 2
        return avg_max_height > config.ARM_RAISE_SHOULDER_MIN

    def reset_arm_raise(self):
        """Reset tracking for next arm raise rep."""
        self.max_height_reached = {"left": 0, "right": 0}
        logger.debug("Arm raise tracking reset")

    def get_arm_raise_height_percentage(self) -> Dict[str, float]:
        """
        Calculate arm raise height as percentage.

        100% = full height achieved (shoulder angle at max)
        0% = arms at sides (shoulder angle low)

        Returns:
            Dict with 'left', 'right', and 'avg' percentages
        """
        # Range from arms at sides (~90°) to full overhead (~170°)
        min_angle = 90
        max_angle = 170
        full_range = max_angle - min_angle

        left_pct = (
            max(0, min(100, (self.max_height_reached["left"] - min_angle) / full_range * 100))
        )
        right_pct = (
            max(0, min(100, (self.max_height_reached["right"] - min_angle) / full_range * 100))
        )
        avg_pct = (left_pct + right_pct) / 2

        return {
            "left": left_pct,
            "right": right_pct,
            "avg": avg_pct,
        }

    def get_symmetry_difference(self) -> float:
        """
        Get the difference in max height between left and right arms.

        Returns:
            Absolute angle difference in degrees
        """
        return abs(
            self.max_height_reached["left"] - self.max_height_reached["right"]
        )
