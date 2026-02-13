"""
arm_raise.py — Lateral arm raise analyser with rep counting.

State machine:
    REST ──(shoulder angle rises past 60°)──▶ RAISING
    RAISING ──(shoulder angle in good range)──▶ TOP
    TOP ──(shoulder angle drops below 60°)──▶ LOWERING
    LOWERING ──(shoulder angle < 30°)──▶ REST  (+1 rep)

Form checks:
    • Arm height (should reach ~170° shoulder angle for full range).
    • Elbow straightness (avoid excessive bend).
    • Left/right symmetry.
"""

from __future__ import annotations

from typing import Dict

import config
from core.feedback_engine import FeedbackEngine, Severity
from core.pose_detector import Landmarks
from exercises.base import ExerciseDetector


class ArmRaiseDetector(ExerciseDetector):
    """Detects lateral arm raise reps and coaches form."""

    def __init__(self, feedback_engine: FeedbackEngine) -> None:
        super().__init__(feedback_engine)
        self._state = "rest"
        self._th = config.ARM_RAISE

    @property
    def name(self) -> str:
        return "Lateral Arm Raise"

    def evaluate(
        self,
        landmarks: Landmarks,
        angles: Dict[str, float],
    ) -> None:
        ls = angles.get("left_shoulder")
        rs = angles.get("right_shoulder")
        le = angles.get("left_elbow")
        re = angles.get("right_elbow")

        shoulder_angle = self._avg_val(ls, rs)
        if shoulder_angle is None:
            return

        # ── State machine ────────────────────────────────────────
        if self._state == "rest":
            if shoulder_angle > 60:
                self._state = "raising"

        elif self._state == "raising":
            lo, hi = self._th.good_range
            if shoulder_angle >= lo:
                self._state = "top"

        elif self._state == "top":
            if shoulder_angle < 60:
                self._state = "lowering"

        elif self._state == "lowering":
            if shoulder_angle < 30:
                self._state = "rest"
                self._rep_count += 1

        # ── Form feedback ────────────────────────────────────────

        # 1. Height check
        if self._state in ("raising", "top"):
            lo, hi = self._th.good_range
            if shoulder_angle < lo:
                self.fb.add(
                    "Raise your arms higher!",
                    Severity.WARNING, joint="shoulder",
                )
            elif lo <= shoulder_angle <= hi:
                self.fb.add(
                    "Great arm height!",
                    Severity.GOOD, joint="shoulder",
                )
            else:
                self.fb.add(
                    "Arms too high — stop at shoulder level",
                    Severity.WARNING, joint="shoulder",
                )

        # 2. Elbow straightness
        elbow_angle = self._avg_val(le, re)
        if elbow_angle is not None and self._state in ("raising", "top"):
            if elbow_angle < self._th.elbow_bend_max:
                self.fb.add(
                    f"Straighten your elbows (bend: {180 - elbow_angle:.0f}°)",
                    Severity.WARNING, joint="elbow",
                )
            else:
                self.fb.add("Good — elbows are straight", Severity.GOOD, joint="elbow")

        # 3. Symmetry check
        if ls is not None and rs is not None:
            diff = abs(ls - rs)
            if diff > self._th.symmetry_tolerance:
                side = "left" if ls < rs else "right"
                self.fb.add(
                    f"Raise your {side} arm to match the other ({diff:.0f}° off)",
                    Severity.WARNING, joint="shoulder",
                )

        # 4. Rest phase rep feedback
        if self._state == "rest" and self._rep_count > 0:
            self.fb.add(
                f"Rep {self._rep_count} done!",
                Severity.GOOD,
            )

    @staticmethod
    def _avg_val(a, b):
        if a is not None and b is not None:
            return (a + b) / 2
        return a or b
