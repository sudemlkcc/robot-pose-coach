#!/usr/bin/env python3
"""
main.py — Entry point for the Robot Pose Coach.

Pipeline per frame:
    1. Capture frame from camera.
    2. Detect pose landmarks (MediaPipe).
    3. Compute joint angles.
    4. Run exercise-specific evaluation (rule-based feedback).
    5. Smooth feedback over time.
    6. Draw skeleton, angles, and feedback panel.
    7. Display the annotated frame.

Controls:
    q / ESC  — quit
    1        — switch to Squat exercise
    2        — switch to Arm Raise exercise
    r        — reset rep counter
    d        — toggle data recording (for future ML training)
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Optional

import cv2

import config
from core.angle_utils import compute_all_angles
from core.feedback_engine import FeedbackEngine
from core.pose_detector import PoseDetector
from core.visualizer import (
    draw_angles,
    draw_feedback_panel,
    draw_skeleton,
    draw_state_indicator,
)
from exercises.arm_raise import ArmRaiseDetector
from exercises.base import ExerciseDetector
from exercises.squat import SquatDetector
from ml.ml_pipeline import DataCollector


# ─────────────────────────────────────────────────────────────────
# Exercise Registry — add new exercises here.
# ─────────────────────────────────────────────────────────────────
EXERCISES: Dict[str, type] = {
    "1": SquatDetector,
    "2": ArmRaiseDetector,
}


def build_exercise(key: str, engine: FeedbackEngine) -> ExerciseDetector:
    """Instantiate an exercise detector by its key."""
    cls = EXERCISES.get(key)
    if cls is None:
        raise ValueError(f"Unknown exercise key: {key}")
    return cls(engine)


# ─────────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Initialise camera ────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_SOURCE in config.py")
        sys.exit(1)

    # ── Initialise modules ───────────────────────────────────────
    detector = PoseDetector()
    engine   = FeedbackEngine()
    exercise: ExerciseDetector = SquatDetector(engine)  # Default exercise

    # Data collector (press 'd' to toggle recording)
    collector = DataCollector(exercise.name.lower().replace(" ", "_"))
    recording = False

    prev_time = time.time()
    fps = 0.0

    print("╔══════════════════════════════════════════════╗")
    print("║         ROBOT POSE COACH — Started           ║")
    print("╠══════════════════════════════════════════════╣")
    print("║  [1] Squat   [2] Arm Raise                  ║")
    print("║  [r] Reset   [d] Record data   [q] Quit     ║")
    print("╚══════════════════════════════════════════════╝")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying...")
                continue

            # Flip horizontally for mirror-like experience
            frame = cv2.flip(frame, 1)

            # ── FPS calculation ──────────────────────────────────
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-8))
            prev_time = now

            # ── 1. Detect pose ───────────────────────────────────
            landmarks = detector.process(frame)

            # ── 2. Compute angles ────────────────────────────────
            angles: Dict[str, float] = {}
            if landmarks:
                angles = compute_all_angles(landmarks)

            # ── 3. Exercise evaluation ───────────────────────────
            engine.begin_frame()
            if landmarks and angles:
                exercise.evaluate(landmarks, angles)

                # Record data if in recording mode
                if recording:
                    collector.record(angles, label="unlabelled")

            feedback_items = engine.end_frame()

            # ── 4. Draw everything ───────────────────────────────
            if landmarks:
                draw_skeleton(frame, landmarks)
                draw_angles(frame, landmarks, angles)
                draw_state_indicator(frame, exercise.state, exercise.rep_count)

            draw_feedback_panel(frame, feedback_items,
                                exercise_name=exercise.name, fps=fps)

            # Recording indicator
            if recording:
                cv2.circle(frame, (frame.shape[1] - 30, 30), 10,
                           (0, 0, 255), -1)  # Red dot
                cv2.putText(frame, "REC", (frame.shape[1] - 70, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ── 5. Show frame ────────────────────────────────────
            cv2.imshow(config.WINDOW_NAME, frame)

            # ── 6. Handle keyboard input ─────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):     # q or ESC
                break

            elif chr(key) in EXERCISES:   # Switch exercise
                engine.reset()
                exercise = build_exercise(chr(key), engine)
                if recording:
                    collector.end_session()
                    collector = DataCollector(exercise.name.lower().replace(" ", "_"))
                    collector.start_session()
                print(f"[INFO] Switched to: {exercise.name}")

            elif key == ord("r"):         # Reset reps
                exercise.reset()
                print("[INFO] Rep counter reset")

            elif key == ord("d"):         # Toggle recording
                recording = not recording
                if recording:
                    path = collector.start_session()
                    print(f"[REC] Recording started → {path}")
                else:
                    collector.end_session()
                    print("[REC] Recording stopped")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # ── Cleanup ──────────────────────────────────────────────
        if recording:
            collector.end_session()
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Pose Coach shut down cleanly.")


if __name__ == "__main__":
    main()
