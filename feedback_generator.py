"""
Feedback Generator: Rule-based feedback system for coaching.

Analyzes poses and angles to generate corrective feedback messages.
This module is designed to be easily extended with ML-based corrections later.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackLevel(Enum):
    """Severity levels for feedback."""
    GOOD = "good"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Feedback:
    """Represents a single feedback message."""
    message: str
    level: FeedbackLevel
    joint: str
    priority: int = 0  # Higher = more important


class FeedbackGenerator:
    """Generates corrective feedback based on exercise performance."""

    def __init__(self):
        """Initialize the feedback generator."""
        logger.info("FeedbackGenerator initialized")

    def evaluate_squat(
        self,
        left_knee: float,
        right_knee: float,
        left_hip: float,
        right_hip: float,
        spine_inclination: float,
        thresholds: Dict,
    ) -> List[Feedback]:
        """
        Evaluate squat form and generate feedback.

        Args:
            left_knee: Left knee angle (degrees)
            right_knee: Right knee angle (degrees)
            left_hip: Left hip angle (degrees)
            right_hip: Right hip angle (degrees)
            spine_inclination: Spine forward lean (degrees)
            thresholds: Dictionary with 'knee_min', 'knee_max', 'back_max', etc.

        Returns:
            List of Feedback objects
        """
        feedback_list = []

        # Check knee depth (should bend below threshold)
        avg_knee = (left_knee + right_knee) / 2
        if avg_knee > thresholds.get("knee_min", 60):
            feedback_list.append(
                Feedback(
                    message="Go deeper! Lower your hips more.",
                    level=FeedbackLevel.WARNING,
                    joint="knee",
                    priority=2,
                )
            )

        # Check knee doesn't overextend
        if avg_knee < thresholds.get("knee_max", 10):
            feedback_list.append(
                Feedback(
                    message="Don't lock your knees at the bottom.",
                    level=FeedbackLevel.WARNING,
                    joint="knee",
                    priority=1,
                )
            )

        # Check back posture
        if spine_inclination > thresholds.get("back_max", 50):
            feedback_list.append(
                Feedback(
                    message="Keep your back straighter. Don't lean forward.",
                    level=FeedbackLevel.ERROR,
                    joint="spine",
                    priority=3,
                )
            )

        # Check for symmetry (left/right imbalance)
        knee_difference = abs(left_knee - right_knee)
        if knee_difference > 15:
            feedback_list.append(
                Feedback(
                    message="Keep your knees even. One side is bending more.",
                    level=FeedbackLevel.WARNING,
                    joint="knee",
                    priority=2,
                )
            )

        # Check hip symmetry
        hip_difference = abs(left_hip - right_hip)
        if hip_difference > 15:
            feedback_list.append(
                Feedback(
                    message="Balance your weight evenly between both legs.",
                    level=FeedbackLevel.WARNING,
                    joint="hip",
                    priority=1,
                )
            )

        # If no errors, give positive feedback
        if not feedback_list:
            feedback_list.append(
                Feedback(
                    message="Great form!",
                    level=FeedbackLevel.GOOD,
                    joint="overall",
                    priority=0,
                )
            )

        return feedback_list

    def evaluate_arm_raise(
        self,
        left_shoulder: float,
        right_shoulder: float,
        left_elbow: float,
        right_elbow: float,
        thresholds: Dict,
    ) -> List[Feedback]:
        """
        Evaluate arm raise form and generate feedback.

        Args:
            left_shoulder: Left shoulder angle (degrees)
            right_shoulder: Right shoulder angle (degrees)
            left_elbow: Left elbow angle (degrees)
            right_elbow: Right elbow angle (degrees)
            thresholds: Dictionary with 'shoulder_min', 'elbow_variance_max', etc.

        Returns:
            List of Feedback objects
        """
        feedback_list = []

        # Check arm height
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        if avg_shoulder < thresholds.get("shoulder_min", 160):
            feedback_list.append(
                Feedback(
                    message="Raise your arms higher.",
                    level=FeedbackLevel.WARNING,
                    joint="shoulder",
                    priority=2,
                )
            )

        # Check elbow straightness (should be near 180 degrees)
        avg_elbow = (left_elbow + right_elbow) / 2
        elbow_variance = abs(avg_elbow - 180)
        if elbow_variance > thresholds.get("elbow_variance_max", 20):
            feedback_list.append(
                Feedback(
                    message="Keep your elbows straight.",
                    level=FeedbackLevel.WARNING,
                    joint="elbow",
                    priority=2,
                )
            )

        # Check arm symmetry
        shoulder_difference = abs(left_shoulder - right_shoulder)
        if shoulder_difference > thresholds.get("symmetry_tolerance", 15):
            feedback_list.append(
                Feedback(
                    message="Raise both arms evenly.",
                    level=FeedbackLevel.WARNING,
                    joint="shoulder",
                    priority=1,
                )
            )

        # If no errors, give positive feedback
        if not feedback_list:
            feedback_list.append(
                Feedback(
                    message="Excellent form!",
                    level=FeedbackLevel.GOOD,
                    joint="overall",
                    priority=0,
                )
            )

        return feedback_list

    def get_priority_feedback(self, feedback_list: List[Feedback]) -> List[str]:
        """
        Sort feedback by priority and return top messages.

        Args:
            feedback_list: List of Feedback objects

        Returns:
            List of feedback strings, sorted by priority (highest first)
        """
        # Sort by priority (descending) and level (error > warning > good)
        level_order = {FeedbackLevel.ERROR: 3, FeedbackLevel.WARNING: 2, FeedbackLevel.GOOD: 1}

        sorted_feedback = sorted(
            feedback_list,
            key=lambda f: (f.priority, level_order[f.level]),
            reverse=True,
        )

        # Return top 3 messages
        return [f.message for f in sorted_feedback[:3]]

    def format_feedback_display(self, feedback_messages: List[str]) -> str:
        """
        Format feedback messages for display on screen.

        Args:
            feedback_messages: List of feedback strings

        Returns:
            Formatted string for display
        """
        if not feedback_messages:
            return "No feedback available"

        return " | ".join(feedback_messages)
