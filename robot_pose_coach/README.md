# 🤖 Robot Pose Coach

A real-time exercise coaching system that uses computer vision to detect body posture and provide corrective feedback — designed first for PC webcam development, then for deployment on a humanoid robot.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Camera     │────▶│ PoseDetector │────▶│ AngleUtils   │
│ (webcam/     │     │ (MediaPipe)  │     │ (geometry)   │
│  robot cam)  │     └──────────────┘     └──────┬───────┘
└─────────────┘                                  │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Visualizer │◀────│   Feedback   │◀────│  Exercise    │
│ (OpenCV     │     │   Engine     │     │  Detector    │
│  drawing)   │     │ (smoothing)  │     │ (rules/ML)   │
└─────────────┘     └──────────────┘     └──────────────┘
```

**Data flow per frame:**
1. **Camera** → BGR frame
2. **PoseDetector** → 33 body landmarks (normalised coordinates)
3. **AngleUtils** → Joint angles (elbow, shoulder, knee, hip, spine)
4. **ExerciseDetector** → Rule-based form analysis → feedback items
5. **FeedbackEngine** → Temporal smoothing (prevents flickering)
6. **Visualizer** → Skeleton + angles + coaching panel drawn on frame

## Project Structure

```
robot_pose_coach/
├── main.py                     # Entry point — camera loop + keyboard controls
├── config.py                   # All settings, thresholds, and constants
├── requirements.txt
├── core/
│   ├── pose_detector.py        # MediaPipe wrapper (swap-friendly)
│   ├── angle_utils.py          # Joint angle computation (pure math)
│   ├── feedback_engine.py      # Feedback aggregation + smoothing
│   └── visualizer.py           # OpenCV drawing (skeleton, panel, angles)
├── exercises/
│   ├── base.py                 # Abstract ExerciseDetector interface
│   ├── squat.py                # Squat detector (state machine + form rules)
│   └── arm_raise.py            # Lateral arm raise detector
└── ml/
    └── ml_pipeline.py          # Data collector + classifier scaffold
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USER/robot_pose_coach.git
cd robot_pose_coach

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

### Tested Versions
| Package        | Version          |
|----------------|------------------|
| Python         | 3.9 – 3.11      |
| mediapipe      | 0.10.9 – 0.10.14|
| opencv-python  | 4.8.x – 4.10.x  |
| numpy          | 1.24.x – 1.26.x |

> **Note:** MediaPipe 0.10.14 is the last release with full `solutions.pose` support. Versions ≥0.10.15 may deprecate the legacy API in favour of the Tasks API.

## Controls

| Key   | Action                        |
|-------|-------------------------------|
| `1`   | Switch to **Squat** exercise  |
| `2`   | Switch to **Arm Raise**       |
| `r`   | Reset rep counter             |
| `d`   | Toggle data recording (CSV)   |
| `q`   | Quit                          |

## Adding a New Exercise

1. Create `exercises/your_exercise.py`
2. Subclass `ExerciseDetector` from `exercises/base.py`
3. Implement `name` property and `evaluate()` method
4. Register it in `main.py`'s `EXERCISES` dict

```python
# exercises/pushup.py
from exercises.base import ExerciseDetector

class PushupDetector(ExerciseDetector):
    @property
    def name(self) -> str:
        return "Push-Up"

    def evaluate(self, landmarks, angles):
        elbow = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2
        if elbow > 160:
            self.fb.add("Lower your body more", Severity.WARNING, joint="elbow")
        elif elbow < 90:
            self.fb.add("Good depth!", Severity.GOOD, joint="elbow")
```

## ML Training Pipeline (Future)

1. **Record data:** Press `d` during exercise to save angle CSVs to `data/`
2. **Label data:** Add correct labels (currently saves as "unlabelled")
3. **Train:** Use `ml/ml_pipeline.py`'s `SklearnClassifier` scaffold
4. **Deploy:** Load the trained model and call `predict()` in your exercise detector

## Adapting for Robot Deployment

The architecture is designed so only **config.py** changes for robot deployment:

```python
# config.py — Robot camera
CAMERA_SOURCE = "http://robot-ip:8080/video"  # or device index
FRAME_WIDTH   = 640   # Match robot camera resolution
FRAME_HEIGHT  = 480
```

The `PoseDetector` wrapper isolates MediaPipe, so you can swap it for an on-device ONNX model by only modifying `core/pose_detector.py`.

## License

MIT — use freely for your internship project and beyond.
