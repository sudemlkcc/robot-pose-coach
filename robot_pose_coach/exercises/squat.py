"""
squat.py — Squat form analyser with rep counting.

State machine:
    STANDING ──(knee bends past threshold)──▶ DESCENDING
    DESCENDING ──(knee reaches bottom target)──▶ BOTTOM
    BOTTOM ──(knee extends past threshold)──▶ ASCENDING
    ASCENDING ──(knee nearly straight)──▶ STANDING  (+1 rep)

Form checks at every frame:
    • Knee angle (too shallow / too deep).
    • Back lean (spine inclination).
    • Knee alignment cue (knees over toes).
"""

from __future__ import annotations

from typing import Dict

import config
from core.feedback_engine import FeedbackEngine, Severity
from core.pose_detector import Landmarks
from exercises.base import ExerciseDetector


class SquatDetector(ExerciseDetector):
    """Detects squat reps and provides real-time form correction."""

    def __init__(self, feedback_engine: FeedbackEngine) -> None:
        super().__init__(feedback_engine)
        self._state = "standing"
        self._th = config.SQUAT

    @property
    def name(self) -> str:
        return "Squat"

    def evaluate(
        self,
        landmarks: Landmarks,
        angles: Dict[str, float],
    ) -> None:
        # Average left/right for robustness.
        knee_angle = self._avg(angles, "left_knee", "right_knee")
        hip_angle  = self._avg(angles, "left_hip", "right_hip")
        spine_incl = angles.get("spine_inclination")

        if knee_angle is None:
            return  # Not enough visibility

        # ── State machine ────────────────────────────────────────
        if self._state == "standing":
            if knee_angle < self._th.knee_angle_max - 15:
                self._state = "descending"

        elif self._state == "descending":
            if knee_angle <= self._th.knee_squat_target + 10:
                self._state = "bottom"

        elif self._state == "bottom":
            if knee_angle > self._th.knee_squat_target + 15:
                self._state = "ascending"

        elif self._state == "ascending":
            if knee_angle >= self._th.knee_angle_max - 10:
                self._state = "standing"
                self._rep_count += 1

        # ── Form feedback ────────────────────────────────────────

        # 1. Depth check
        if self._state in ("descending", "bottom"):
            if knee_angle > self._th.knee_squat_target + 25:
                self.fb.add(
                    "Go deeper — bend your knees more",
                    Severity.WARNING, joint="knee",
                )
            elif knee_angle < self._th.knee_angle_min:
                self.fb.add(
                    "Too deep! Don't let knees collapse",
                    Severity.ERROR, joint="knee",
                )
            else:
                self.fb.add(
                    "Good depth!",
                    Severity.GOOD, joint="knee",
                )

        # 2. Back lean check
        if spine_incl is not None:
            if spine_incl > self._th.spine_lean_max:
                self.fb.add(
                    f"Keep your back straighter (lean: {spine_incl:.0f}°)",
                    Severity.ERROR, joint="spine",
                )
            elif spine_incl > self._th.spine_lean_max - 10:
                self.fb.add(
                    "Slight forward lean — engage your core",
                    Severity.WARNING, joint="spine",
                )
            else:
                self.fb.add("Back position looks good", Severity.GOOD, joint="spine")

        # 3. Knee symmetry
        lk = angles.get("left_knee")
        rk = angles.get("right_knee")
        if lk is not None and rk is not None:
            asymmetry = abs(lk - rk)
            if asymmetry > 15:
                self.fb.add(
                    f"Uneven knees ({asymmetry:.0f}° difference) — balance your weight",
                    Severity.WARNING, joint="knee",
                )

        # 4. Standing phase positive feedback
        if self._state == "standing" and self._rep_count > 0:
            self.fb.add(
                f"Rep {self._rep_count} complete — nice work!",
                Severity.GOOD,
            )

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _avg(angles: Dict[str, float], left: str, right: str):
        l = angles.get(left)
        r = angles.get(right)
        if l is not None and r is not None:
            return (l + r) / 2
        return l or r
