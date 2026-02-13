"""
feedback_engine.py — Aggregates per-frame feedback and smooths it
over time so the displayed messages don't flicker.

The engine is exercise-agnostic: exercise modules push feedback items
each frame, and the engine decides what to show.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import config


class Severity(Enum):
    """How critical a feedback item is."""
    GOOD = auto()     # Correct form
    INFO = auto()     # Neutral tip
    WARNING = auto()  # Minor issue
    ERROR = auto()    # Significant form break


@dataclass
class FeedbackItem:
    """A single piece of coaching feedback."""
    message: str
    severity: Severity
    joint: str = ""   # Which joint/body part (for future robot gestures)

    @property
    def key(self) -> str:
        """Stable identity for smoothing (ignores minor wording changes)."""
        return f"{self.severity.name}:{self.joint}:{self.message}"


class FeedbackEngine:
    """
    Collects feedback items each frame and outputs a temporally
    stable list to display.  This prevents the on-screen text from
    changing every single frame due to tiny angle fluctuations.
    """

    def __init__(self, stability_frames: int = config.FEEDBACK_STABILITY_FRAMES):
        self._stability = stability_frames
        self._history: List[List[FeedbackItem]] = []
        self._current_display: List[FeedbackItem] = []

    def begin_frame(self) -> None:
        """Call at the start of each frame to reset transient feedback."""
        self._frame_items: List[FeedbackItem] = []

    def add(self, message: str, severity: Severity, joint: str = "") -> None:
        """Push a feedback item for the current frame."""
        self._frame_items.append(FeedbackItem(message, severity, joint))

    def end_frame(self) -> List[FeedbackItem]:
        """
        Finalise the frame.  Returns the smoothed list of feedback
        items that should be displayed.
        """
        self._history.append(self._frame_items)
        if len(self._history) > self._stability:
            self._history = self._history[-self._stability:]

        # Count how often each feedback key appeared in the window.
        counter: Counter = Counter()
        latest_by_key: dict[str, FeedbackItem] = {}
        for frame_items in self._history:
            for item in frame_items:
                counter[item.key] += 1
                latest_by_key[item.key] = item

        # Only show items that appeared in at least half the window.
        threshold = max(1, self._stability // 2)
        stable_items = [
            latest_by_key[key]
            for key, count in counter.items()
            if count >= threshold
        ]

        # Sort: errors first, then warnings, then good.
        severity_order = {Severity.ERROR: 0, Severity.WARNING: 1,
                          Severity.INFO: 2, Severity.GOOD: 3}
        stable_items.sort(key=lambda i: severity_order.get(i.severity, 99))

        self._current_display = stable_items
        return self._current_display

    @property
    def display(self) -> List[FeedbackItem]:
        """Last computed stable display list."""
        return self._current_display

    def reset(self) -> None:
        """Clear all history (e.g. when switching exercises)."""
        self._history.clear()
        self._current_display.clear()
