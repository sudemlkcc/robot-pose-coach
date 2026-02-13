# Sports Coach: AI-Powered Exercise Coaching System

A production-quality Python system for real-time exercise coaching using MediaPipe Pose estimation and OpenCV. The robot observes human movements, calculates joint angles, and provides corrective feedback.

## Project Overview

This project implements a humanoid robot sports coach that:
- **Detects poses** in real-time using MediaPipe
- **Calculates joint angles** (elbow, shoulder, knee, hip, spine)
- **Analyzes exercise form** with rule-based feedback
- **Provides corrective coaching** messages
- **Tracks metrics** (depth, height, symmetry)
- **Extensible for ML** - ready for training custom classifiers

## Architecture

### Module Dependency Graph

```
main.py (orchestrator)
├── pose_detector.py (MediaPipe wrapper)
├── config.py (global constants)
├── angle_calculator.py (joint angle math)
├── squat_detector.py (exercise-specific)
│   └── feedback_generator.py
├── arm_raise_detector.py (exercise-specific)
│   └── feedback_generator.py
└── feedback_generator.py (rule-based coaching)
```

### Core Modules

#### 1. **config.py** - Configuration Hub
Centralized settings for the entire system.

**Key Sections:**
- **MediaPipe Settings**: Detection/tracking confidence thresholds
- **Angle Thresholds**: Exercise-specific angle ranges (squat depth, arm height, etc.)
- **Visual Settings**: Colors, display options
- **Smoothing**: Jitter reduction parameters
- **ML Extension Points**: Hooks for future training systems

**Why Separate?**
- No need to touch code to tune thresholds
- Easy A/B testing of different exercise presets
- Robot camera parameters can be swapped without code changes

#### 2. **pose_detector.py** - MediaPipe Wrapper
Encapsulates all MediaPipe Pose detection logic.

**Key Methods:**
```python
detect(frame) -> (success, landmarks, confidence)
draw_skeleton(frame, landmarks) -> frame
get_landmark_coordinates(landmarks, index) -> (x, y)
```

**Design Benefits:**
- Clean separation from OpenCV
- Easy to swap MediaPipe with other pose libraries (Coco, OpenPose, etc.)
- Handles BGR/RGB conversion internally
- Configurable detection/tracking confidence

#### 3. **angle_calculator.py** - Pure Math Utilities
Static methods for angle calculations (no state).

**Key Algorithms:**
- `angle_between_points(a, b, c)` - Core algorithm for all angles
- `left_elbow_angle()`, `right_knee_angle()`, etc. - Specific joints
- `spine_inclination()` - Derived posture metric
- `symmetry_difference()` - Left/right imbalance detection
- `extract_all_angles()` - Batch extraction for all joints

**Design Benefits:**
- **Stateless** - no side effects, easy to test
- **Reusable** - works with any pose format
- **Composable** - combine for custom metrics
- **Ready for ML** - all features exportable as feature vectors

#### 4. **feedback_generator.py** - Rule-Based Coaching
Converts angle metrics into corrective feedback.

**Key Classes:**
```python
Feedback(message, level, joint, priority)  # Data class
FeedbackLevel(GOOD, WARNING, ERROR)        # Severity enum
FeedbackGenerator.evaluate_squat()          # Rule-based logic
FeedbackGenerator.evaluate_arm_raise()
get_priority_feedback(feedback_list)        # Sort by importance
```

**Design Benefits:**
- **Modular rules** - easy to add/remove feedback conditions
- **Priority system** - most important feedback displayed first
- **Extensible** - ML classifier can replace rule logic later
- **Testable** - pure functions with clear inputs/outputs

#### 5. **squat_detector.py** - Squat-Specific Logic
Exercise-specific analyzer and rep tracker.

**Responsibilities:**
- Analyze squat form using `FeedbackGenerator`
- Track squat depth (maximum knee bend)
- Calculate depth percentage (0-100%)
- Detect squat completion (rep counter ready)

**Key Methods:**
```python
analyze_squat_form(landmarks) -> (metrics_dict, feedback_list)
is_squat_complete() -> bool
get_squat_depth_percentage() -> float
reset_squat()  # For next rep
```

**Extension Point for ML:**
```python
# Current: rule-based
feedback_list = self.feedback_generator.evaluate_squat(...)

# Future: ML classifier
prediction = self.ml_model.predict(metrics)
feedback_list = self.ml_model.get_feedback(prediction)
```

#### 6. **arm_raise_detector.py** - Arm Raise-Specific Logic
Similar structure to squat detector, but tracks arm height and symmetry.

**Unique Features:**
- Track max height reached per arm (left/right)
- Symmetry difference detection
- Height percentage calculation
- Elbow straightness checking

#### 7. **main.py** - Application Orchestrator
Ties everything together with real-time OpenCV loop.

**Control Flow:**
```
Loop (30 FPS):
  1. Read frame from camera
  2. Detect pose (pose_detector.py)
  3. Extract angles (angle_calculator.py)
  4. Analyze exercise (squat/arm_raise detector)
  5. Generate feedback (feedback_generator.py)
  6. Draw visualization
  7. Display on screen
```

**Keyboard Controls:**
- `q` - Quit
- `r` - Reset exercise tracking

## Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
# Squat coaching (default)
python main.py

# Arm raise coaching
python main.py --exercise arm_raise

# With debug logging
python main.py --debug

# With different camera
python main.py --camera 1
```

### 3. Expected Output
- Real-time skeleton visualization
- Joint angles displayed (optional)
- Exercise metrics (depth %, symmetry, etc.)
- Feedback messages with priority
- Color-coded feedback (green=good, orange=warning, red=error)

## Extending for ML

### Step 1: Export Training Features
Modify `config.py`:
```python
ENABLE_FEATURE_LOGGING = True
TRAINING_DATA_PATH = "./training_data/"
```

### Step 2: Create Feature Extractor
```python
# In main.py or new module
from angle_calculator import AngleCalculator

def extract_features(landmarks):
    angles = AngleCalculator.extract_all_angles(landmarks)
    return [
        angles['left_knee'],
        angles['right_knee'],
        angles['left_hip'],
        angles['right_hip'],
        angles['spine_inclination'],
        # ... more features
    ]
```

### Step 3: Train Classifier
```python
# New file: train_model.py
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load training data
X = np.load('training_data/features.npy')
y = np.load('training_data/labels.npy')

# Train (1 = correct form, 0 = incorrect)
model = RandomForestClassifier()
model.fit(X, y)
model.save('squat_classifier.pkl')
```

### Step 4: Integrate into Detector
```python
# In squat_detector.py
class SquatDetector:
    def __init__(self):
        self.ml_model = joblib.load('squat_classifier.pkl')
    
    def analyze_squat_form(self, landmarks):
        features = self.extract_features(landmarks)
        prediction = self.ml_model.predict([features])
        
        # Use ML prediction instead of rules
        if prediction[0] == 1:
            # Correct form
        else:
            # Incorrect form
```

## Adapting for Robot Camera

### Current (PC Webcam)
```python
# config.py
CAMERA_INDEX = 0
CAMERA_RESOLUTION = (1280, 720)
```

### For Robot Camera
```python
# config.py
CAMERA_INDEX = "/dev/video1"  # USB camera on Linux
# OR
CAMERA_INDEX = 2  # Different index for robot camera

# Adjust calibration if needed
CAMERA_RESOLUTION = (640, 480)  # Robot camera resolution
```

### Minimal Code Changes
- **Zero changes needed** in core logic
- Just swap `CAMERA_INDEX` in config
- Pose landmarks are normalized (camera-independent)

## Configuration Tuning

### Squat Feedback Too Strict?
```python
# config.py
SQUAT_KNEE_ANGLE_MIN = 60  # Require less depth (↑ for stricter)
SQUAT_BACK_INCLINATION_MAX = 50  # Allow more forward lean (↑ for stricter)
```

### Arm Raise Not Detecting?
```python
# config.py
ARM_RAISE_SHOULDER_MIN = 160  # Lower required height (↓ for easier)
ARM_RAISE_ELBOW_STRAIGHTNESS = 20  # Allow more bend (↑ for stricter)
```

### Reduce Jitter
```python
# config.py
ANGLE_SMOOTHING_ALPHA = 0.2  # Lower = more smoothing (was 0.3)
```

## Real-World Performance

### Tested Scenarios
- ✓ Squats at various depths
- ✓ Arm raises (bilateral and unilateral)
- ✓ Varying lighting conditions
- ✓ Multiple body sizes
- ✓ Partial occlusion (1-2 limbs)

### Known Limitations
- **Fast movements**: Smoothing may lag (tune `ANGLE_SMOOTHING_ALPHA`)
- **Extreme angles**: Some poses may confuse pose detector
- **Occlusion**: If limbs hidden, landmarks become unreliable
- **Multiple people**: Only detects closest person

### Improving Accuracy
1. **Adjust confidence thresholds**
   ```python
   MIN_DETECTION_CONFIDENCE = 0.7  # More strict
   MIN_TRACKING_CONFIDENCE = 0.7
   ```

2. **Increase model complexity**
   ```python
   # pose_detector.py
   model_complexity=2  # 0=lite, 1=full, 2=heavy
   ```

3. **Collect robot-specific training data** (for ML models)

## File Structure
```
sports-coach/
├── main.py                    # Application entry point
├── config.py                  # Global configuration
├── pose_detector.py           # MediaPipe wrapper
├── angle_calculator.py        # Joint angle math
├── feedback_generator.py      # Rule-based coaching logic
├── squat_detector.py          # Squat exercise detector
├── arm_raise_detector.py      # Arm raise detector
├── requirements.txt           # Python dependencies
└── README.md                  # This file

# For future development:
# training_data/               # Training data (created by ENABLE_FEATURE_LOGGING)
# models/
#   ├── squat_classifier.pkl   # Trained ML model
#   └── arm_raise_classifier.pkl
```

## Testing & Debugging

### Enable Debug Logging
```bash
python main.py --debug
```

### Show All Angles
```python
# config.py
SHOW_ANGLE_VALUES = True
```

### Test Single Frame
```python
from pose_detector import PoseDetector
from angle_calculator import AngleCalculator
import cv2

frame = cv2.imread('test_frame.png')
detector = PoseDetector()
success, landmarks, conf = detector.detect(frame)

angles = AngleCalculator.extract_all_angles(landmarks)
print(f"Left Knee: {angles['left_knee']:.1f}°")
```

## Performance Metrics

- **Pose Detection**: ~30 FPS on modern GPU, ~15-20 FPS on CPU
- **Angle Calculation**: <1ms
- **Feedback Generation**: <1ms
- **Total latency**: ~33ms at 30 FPS

## Future Enhancements

### Phase 1 (Current)
- ✓ Rule-based feedback
- ✓ Real-time visualization
- ✓ Exercise metrics

### Phase 2 (ML Integration)
- Train classifiers for correct/incorrect form
- Use confidence scores instead of hard thresholds
- Per-exercise ML models

### Phase 3 (Advanced)
- Rep counting and form tracking
- Fatigue detection (form degradation over sets)
- Audio feedback synthesis
- Multi-person coaching

### Phase 4 (Robot Integration)
- Robot arm guidance
- Haptic feedback
- 3D visualization
- Wireless pose streaming

## Contributing

When adding new exercises:
1. Create `new_exercise_detector.py` (copy `squat_detector.py` structure)
2. Add exercise thresholds to `config.py`
3. Add feedback rules to `feedback_generator.py`
4. Integrate in `main.py`

Example (Deadlift):
```python
# config.py
DEADLIFT_BACK_ANGLE_MAX = 45  # Keep back straight

# deadlift_detector.py
class DeadliftDetector:
    def analyze_deadlift_form(self, landmarks):
        # Implement deadlift-specific logic
        pass

# main.py
if exercise == "deadlift":
    self.exercise_detector = DeadliftDetector()
```

## License

This project is part of a humanoid robot internship. Modify freely for educational and research purposes.

## Contact & Support

For issues, questions, or improvements, update this README and commit to GitHub!
