"""
base.py — Abstract base class every exercise detector must implement.

This enforces a consistent interface so main.py can swap exercises
at runtime and the future ML pipeline can wrap the same API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from core.feedback_engine import FeedbackEngine
from core.pose_detector import Landmarks


class ExerciseDetector(ABC):
    """
    Base class for all exercise detectors.

    Subclasses must implement:
        • name       — human-readable exercise name.
        • evaluate() — inspect landmarks + angles, push feedback.

    The detector is given a FeedbackEngine to push messages into.
    It should NOT call engine.begin_frame() or engine.end_frame();
    the main loop handles that.
    """

    def __init__(self, feedback_engine: FeedbackEngine) -> None:
        self.fb = feedback_engine
        self._rep_count: int = 0
        self._state: str = "idle"

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the exercise (e.g. 'Squat')."""
        ...

    @abstractmethod
    def evaluate(
        self,
        landmarks: Landmarks,
        angles: Dict[str, float],
    ) -> None:
        """
        Analyse the current frame's landmarks and angles.

        Push feedback via self.fb.add(...) for each observation.
        Update self._state and self._rep_count as needed.
        """
        ...

    @property
    def state(self) -> str:
        return self._state

    @property
    def rep_count(self) -> int:
        return self._rep_count

    def reset(self) -> None:
        """Reset rep counter and state (e.g. when restarting)."""
        self._rep_count = 0
        self._state = "idle"
        self.fb.reset()
