"""
visualizer.py — All OpenCV drawing logic in one place.

Draws:
  • Skeleton connections + landmark dots.
  • Joint angle annotations near each vertex.
  • A translucent feedback panel with coaching messages.
  • An exercise-name / FPS header bar.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from core.feedback_engine import FeedbackItem, Severity
from core.pose_detector import Landmarks

# MediaPipe pose connections (pairs of landmark names).
_CONNECTIONS: List[Tuple[str, str]] = [
    # Torso
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    # Left arm
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    # Right arm
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    # Left leg
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    # Right leg
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
]

# Which angles to show on screen (subset to avoid clutter).
_DISPLAY_ANGLES = {
    "left_elbow", "right_elbow",
    "left_knee", "right_knee",
    "left_shoulder", "right_shoulder",
}

# Map angle name → the landmark at the vertex (where text appears).
_ANGLE_VERTEX: Dict[str, str] = {
    "left_elbow":    "LEFT_ELBOW",
    "right_elbow":   "RIGHT_ELBOW",
    "left_knee":     "LEFT_KNEE",
    "right_knee":    "RIGHT_KNEE",
    "left_shoulder": "LEFT_SHOULDER",
    "right_shoulder":"RIGHT_SHOULDER",
    "left_hip":      "LEFT_HIP",
    "right_hip":     "RIGHT_HIP",
}


def draw_skeleton(
    frame: np.ndarray,
    landmarks: Landmarks,
    color: Tuple[int, int, int] = config.SKELETON_COLOR,
    thickness: int = config.SKELETON_THICKNESS,
) -> None:
    """Draw bones and landmark dots on the frame (in-place)."""
    h, w = frame.shape[:2]

    # Draw connections
    for name_a, name_b in _CONNECTIONS:
        pt_a = landmarks.get(name_a)
        pt_b = landmarks.get(name_b)
        if pt_a is None or pt_b is None:
            continue
        if pt_a.visibility < 0.5 or pt_b.visibility < 0.5:
            continue
        px_a = pt_a.to_pixel(w, h)
        px_b = pt_b.to_pixel(w, h)
        cv2.line(frame, px_a, px_b, color, thickness, cv2.LINE_AA)

    # Draw landmark dots
    for name, lm in landmarks.items():
        if lm.visibility < 0.5:
            continue
        px = lm.to_pixel(w, h)
        cv2.circle(frame, px, config.LANDMARK_RADIUS, color, -1, cv2.LINE_AA)


def draw_angles(
    frame: np.ndarray,
    landmarks: Landmarks,
    angles: Dict[str, float],
) -> None:
    """Annotate selected joint angles near the vertex landmark."""
    h, w = frame.shape[:2]

    for angle_name, degrees in angles.items():
        if angle_name not in _DISPLAY_ANGLES:
            continue
        vertex_name = _ANGLE_VERTEX.get(angle_name)
        if vertex_name is None:
            continue
        lm = landmarks.get(vertex_name)
        if lm is None or lm.visibility < 0.5:
            continue

        px, py = lm.to_pixel(w, h)
        label = f"{int(degrees)}"
        cv2.putText(
            frame, label, (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            config.ANGLE_TEXT_COLOR, 1, cv2.LINE_AA,
        )


def draw_feedback_panel(
    frame: np.ndarray,
    feedback_items: List[FeedbackItem],
    exercise_name: str = "",
    fps: float = 0.0,
) -> None:
    """
    Draw a translucent panel on the left side with coaching messages.
    """
    h, w = frame.shape[:2]
    panel_w = min(420, w // 3)
    panel_h = h

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, config.FEEDBACK_PANEL_ALPHA, frame,
                    1 - config.FEEDBACK_PANEL_ALPHA, 0, frame)

    # Header
    y = 35
    header = exercise_name.upper() if exercise_name else "POSE COACH"
    cv2.putText(frame, header, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS
    y += 30
    cv2.putText(frame, f"FPS: {fps:.0f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    y += 15
    cv2.line(frame, (15, y), (panel_w - 15, y), (100, 100, 100), 1)
    y += 25

    if not feedback_items:
        cv2.putText(frame, "No pose detected", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1, cv2.LINE_AA)
        return

    # Feedback messages
    severity_colors = {
        Severity.GOOD:    config.FEEDBACK_GOOD_COLOR,
        Severity.INFO:    (200, 200, 200),
        Severity.WARNING: (0, 180, 255),   # Orange
        Severity.ERROR:   config.FEEDBACK_BAD_COLOR,
    }
    severity_icons = {
        Severity.GOOD: "[OK]",
        Severity.INFO: "[i]",
        Severity.WARNING: "[!]",
        Severity.ERROR: "[X]",
    }

    for item in feedback_items:
        if y > panel_h - 20:
            break
        color = severity_colors.get(item.severity, (200, 200, 200))
        icon  = severity_icons.get(item.severity, "")

        # Word-wrap long messages
        text = f"{icon} {item.message}"
        max_chars = panel_w // 11  # rough char-per-pixel estimate
        lines = _wrap_text(text, max_chars)
        for line in lines:
            if y > panel_h - 20:
                break
            cv2.putText(frame, line, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 24
        y += 6  # gap between items


def draw_state_indicator(
    frame: np.ndarray,
    state: str,
    rep_count: int = 0,
) -> None:
    """Draw the exercise state and rep counter in the top-right corner."""
    h, w = frame.shape[:2]
    text = f"State: {state}  |  Reps: {rep_count}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x = w - tw - 20
    cv2.putText(frame, text, (x, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


# ── Helpers ──────────────────────────────────────────────────────

def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Simple word-wrap."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
