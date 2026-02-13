"""
Configuration and constants for the sports coach system.

This file centralizes all tunable parameters, making it easy to adjust
thresholds and settings without modifying core logic.
"""

# ============================================================================
# MEDIAPIPE POSE DETECTION SETTINGS
# ============================================================================

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ============================================================================
# ANGLE THRESHOLDS (in degrees) FOR EXERCISE DETECTION
# ============================================================================

# SQUAT DETECTION THRESHOLDS
SQUAT_KNEE_ANGLE_MIN = 60      # Knee must bend below this angle
SQUAT_KNEE_ANGLE_MAX = 175     # Knee should not be fully locked
SQUAT_BACK_INCLINATION_MAX = 50  # Back should not lean too much forward
SQUAT_HIP_KNEE_RATIO = 0.9     # Hip and knee should bend proportionally

# ARM RAISE DETECTION THRESHOLDS
ARM_RAISE_SHOULDER_MIN = 160   # Arm should raise to at least this angle
ARM_RAISE_ELBOW_STRAIGHTNESS = 20  # Elbow should be relatively straight (variance from 180)
ARM_RAISE_SYMMETRY_TOLERANCE = 15  # Left and right arms within this angle

# GENERAL POSTURE THRESHOLDS
SPINE_INCLINATION_THRESHOLD = 45  # Spine should not lean more than this

# ============================================================================
# VISUAL DISPLAY SETTINGS
# ============================================================================

# Colors in BGR format
SKELETON_COLOR = (0, 255, 0)      # Green
JOINT_COLOR = (0, 0, 255)         # Red
TEXT_COLOR = (255, 255, 255)      # White
WARNING_COLOR = (0, 165, 255)     # Orange
ERROR_COLOR = (0, 0, 255)         # Red
GOOD_COLOR = (0, 255, 0)          # Green

SKELETON_THICKNESS = 2
JOINT_RADIUS = 5
FONT_SCALE = 0.6
FONT_THICKNESS = 1

# ============================================================================
# ANGLE SMOOTHING (for reducing jitter in real-time calculations)
# ============================================================================

ANGLE_SMOOTHING_ALPHA = 0.3  # Exponential moving average factor

# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================

LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
SHOW_ANGLE_VALUES = True  # Display calculated angles on screen
SHOW_CONFIDENCE_SCORES = True  # Display pose detection confidence

# ============================================================================
# CAMERA SETTINGS
# ============================================================================

CAMERA_INDEX = 0  # Default camera (0 = webcam, change for robot camera)
CAMERA_RESOLUTION = (1280, 720)  # Width, Height
CAMERA_FPS = 30

# ============================================================================
# ML MODEL EXTENSION POINTS
# ============================================================================

# These settings prepare the system for future ML model integration
FEATURE_VECTOR_SIZE = 15  # Number of angle features to track
ENABLE_FEATURE_LOGGING = False  # Log features to CSV for training data
TRAINING_DATA_PATH = "./training_data/"

# ============================================================================
# EXERCISE PRESETS (for switching between different exercises)
# ============================================================================

EXERCISES = {
    "squat": {
        "name": "Squat",
        "feedback_keys": ["knee_depth", "back_posture", "symmetry"],
    },
    "arm_raise": {
        "name": "Arm Raise",
        "feedback_keys": ["arm_height", "elbow_straightness", "symmetry"],
    },
    "deadlift": {
        "name": "Deadlift",
        "feedback_keys": ["back_posture", "knee_angle", "hip_position"],
    },
}
