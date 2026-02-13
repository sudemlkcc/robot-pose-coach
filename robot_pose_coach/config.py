"""
config.py — Central configuration for Robot Pose Coach.

All tunable parameters live here so you never hunt through code
to change a threshold.  When the project moves to the robot,
only CAMERA_SOURCE and FRAME_* values need updating.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ── Camera / Display ─────────────────────────────────────────────
CAMERA_SOURCE = 0                # 0 = default webcam; change to robot camera URL/index later
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
FPS_TARGET    = 30
WINDOW_NAME   = "Robot Pose Coach"

# ── MediaPipe Pose Settings ──────────────────────────────────────
POSE_MIN_DETECTION_CONFIDENCE  = 0.7
POSE_MIN_TRACKING_CONFIDENCE   = 0.6
POSE_MODEL_COMPLEXITY          = 1     # 0=lite, 1=full, 2=heavy

# ── Visualiser ───────────────────────────────────────────────────
SKELETON_COLOR        = (0, 255, 128)   # BGR — green
SKELETON_THICKNESS    = 2
LANDMARK_RADIUS       = 5
ANGLE_TEXT_COLOR       = (255, 255, 255)
FEEDBACK_GOOD_COLOR   = (0, 200, 0)     # BGR
FEEDBACK_BAD_COLOR    = (0, 0, 230)
FEEDBACK_FONT_SCALE   = 0.8
FEEDBACK_THICKNESS    = 2
FEEDBACK_PANEL_ALPHA  = 0.65            # Translucent overlay panel


# ── Joint Angle Definitions ──────────────────────────────────────
# Each tuple: (first_landmark, vertex_landmark, end_landmark)
# Uses MediaPipe PoseLandmark enum names (resolved at runtime).
# This lets you add new angles without touching computation code.

@dataclass
class JointAngleDef:
    """Definition of a joint angle by three landmark names."""
    name: str
    a: str   # first point
    b: str   # vertex (where angle is measured)
    c: str   # end point


JOINT_ANGLES = [
    # ── Arms ──
    JointAngleDef("left_elbow",    "LEFT_SHOULDER",  "LEFT_ELBOW",    "LEFT_WRIST"),
    JointAngleDef("right_elbow",   "RIGHT_SHOULDER", "RIGHT_ELBOW",   "RIGHT_WRIST"),
    JointAngleDef("left_shoulder", "LEFT_ELBOW",     "LEFT_SHOULDER", "LEFT_HIP"),
    JointAngleDef("right_shoulder","RIGHT_ELBOW",    "RIGHT_SHOULDER","RIGHT_HIP"),
    # ── Legs ──
    JointAngleDef("left_knee",     "LEFT_HIP",       "LEFT_KNEE",     "LEFT_ANKLE"),
    JointAngleDef("right_knee",    "RIGHT_HIP",      "RIGHT_KNEE",    "RIGHT_ANKLE"),
    JointAngleDef("left_hip",      "LEFT_SHOULDER",   "LEFT_HIP",     "LEFT_KNEE"),
    JointAngleDef("right_hip",     "RIGHT_SHOULDER",  "RIGHT_HIP",    "RIGHT_KNEE"),
]


# ── Exercise-Specific Thresholds ─────────────────────────────────

@dataclass
class SquatThresholds:
    """Thresholds for squat correctness detection."""
    knee_angle_min: float = 70.0     # Deep squat lower bound
    knee_angle_max: float = 160.0    # Standing upper bound
    knee_squat_target: float = 100.0 # Ideal bottom-of-squat knee angle
    hip_angle_min: float = 60.0
    hip_angle_max: float = 170.0
    spine_lean_max: float = 45.0     # Max forward lean (degrees from vertical)
    knee_over_toe_warn: bool = True  # Warn if knee passes toe line


@dataclass
class ArmRaiseThresholds:
    """Thresholds for lateral arm raise correctness."""
    target_angle: float = 170.0         # Fully raised
    good_range: Tuple[float, float] = (155.0, 180.0)
    elbow_bend_max: float = 160.0       # Elbow should stay nearly straight
    symmetry_tolerance: float = 20.0    # Max asymmetry in degrees


SQUAT     = SquatThresholds()
ARM_RAISE = ArmRaiseThresholds()


# ── Feedback Smoothing ───────────────────────────────────────────
# Prevents flickering messages by requiring N consecutive identical
# feedback frames before updating the displayed message.
FEEDBACK_STABILITY_FRAMES = 5
