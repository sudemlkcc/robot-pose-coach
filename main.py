"""
Sports Coach: Real-time pose-based exercise coaching system.

Main application that integrates pose detection, angle calculation,
exercise-specific detectors, and feedback generation.

Usage:
    python main.py --exercise squat      # Demo squat coaching
    python main.py --exercise arm_raise  # Demo arm raise coaching
    python main.py --help                # Show all options

Press 'q' to quit, 'r' to reset exercise tracking.
"""

import cv2
import argparse
import logging
from typing import Optional
import sys

import config
from pose_detector import PoseDetector
from angle_calculator import AngleCalculator
from squat_detector import SquatDetector
from arm_raise_detector import ArmRaiseDetector
from feedback_generator import FeedbackLevel

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SportCoachApp:
    """Main application for real-time exercise coaching."""

    def __init__(self, exercise: str = "squat"):
        """
        Initialize the sports coach application.

        Args:
            exercise: Type of exercise ("squat" or "arm_raise")
        """
        self.exercise = exercise
        self.pose_detector = PoseDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

        # Initialize exercise-specific detector
        if exercise == "squat":
            self.exercise_detector = SquatDetector()
        elif exercise == "arm_raise":
            self.exercise_detector = ArmRaiseDetector()
        else:
            raise ValueError(f"Unknown exercise: {exercise}")

        # For angle smoothing
        self.smoothed_angles = {}

        logger.info(f"SportCoachApp initialized for {exercise}")

    def run(self, camera_index: int = 0):
        """
        Run the real-time coaching application.

        Args:
            camera_index: Camera index (0 = default webcam)
        """
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_index}")
            return

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])

        logger.info(f"Camera {camera_index} opened successfully")
        logger.info("Press 'q' to quit, 'r' to reset exercise tracking")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break

                frame_count += 1

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # Detect pose
                success, landmarks, confidence = self.pose_detector.detect(frame)

                # Draw background info
                self._draw_status_bar(
                    frame, frame_count, success, confidence, self.exercise
                )

                if success:
                    # Draw skeleton
                    frame = self.pose_detector.draw_skeleton(
                        frame,
                        landmarks,
                        confidence_threshold=0.5,
                        skeleton_color=config.SKELETON_COLOR,
                        joint_color=config.JOINT_COLOR,
                        skeleton_thickness=config.SKELETON_THICKNESS,
                        joint_radius=config.JOINT_RADIUS,
                    )

                    # Analyze exercise form
                    if self.exercise == "squat":
                        metrics, feedback_list = self.exercise_detector.analyze_squat_form( # type: ignore  # mediapipe typing yok
                            landmarks
                        )
                        self._draw_squat_info(frame, metrics)
                    elif self.exercise == "arm_raise":
                        metrics, feedback_list = (
                            self.exercise_detector.analyze_arm_raise_form(landmarks) # type: ignore
                        )
                        self._draw_arm_raise_info(frame, metrics)

                    # Get prioritized feedback messages
                    if feedback_list:
                        feedback_messages = self.exercise_detector.feedback_generator.get_priority_feedback(
                            feedback_list
                        )
                        self._draw_feedback(frame, feedback_messages, feedback_list)

                    # Draw angles if enabled
                    if config.SHOW_ANGLE_VALUES:
                        self._draw_angles(frame, landmarks)

                # Display frame
                cv2.imshow("Sports Coach - Press 'q' to quit, 'r' to reset", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested by user")
                    break
                elif key == ord("r"):
                    self.exercise_detector.reset_squat() # type: ignore
                    self.exercise_detector.reset_arm_raise() # type: ignore
                    logger.info("Exercise tracking reset")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose_detector.release()
            logger.info("Application closed")

    def _draw_status_bar(
        self,
        frame,
        frame_count: int,
        pose_detected: bool,
        confidence: float,
        exercise: str,
    ):
        """Draw status information at top of frame."""
        h, w = frame.shape[:2]

        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)

        # Exercise name
        cv2.putText(
            frame,
            f"Exercise: {exercise.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            config.TEXT_COLOR,
            2,
        )

        # FPS and detection status
        status_text = "✓ Pose Detected" if pose_detected else "✗ No Pose"
        status_color = config.GOOD_COLOR if pose_detected else config.ERROR_COLOR
        cv2.putText(
            frame,
            status_text,
            (w - 250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        # Confidence
        if pose_detected:
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (w - 500, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                config.TEXT_COLOR,
                1,
            )

    def _draw_squat_info(self, frame, metrics: dict):
        """Draw squat-specific metrics on frame."""
        h, w = frame.shape[:2]
        y_offset = 70

        info_text = [
            f"Avg Knee Angle: {metrics['avg_knee']:.1f}°",
            f"Spine Inclination: {metrics['spine_inclination']:.1f}°",
            f"Depth: {self.exercise_detector.get_squat_depth_percentage():.0f}%", # type: ignore
            f"In Squat: {'Yes' if metrics['in_squat'] else 'No'}",
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.TEXT_COLOR,
                config.FONT_THICKNESS,
            )

    def _draw_arm_raise_info(self, frame, metrics: dict):
        """Draw arm raise-specific metrics on frame."""
        h, w = frame.shape[:2]
        y_offset = 70

        height_pct = self.exercise_detector.get_arm_raise_height_percentage() # type: ignore

        info_text = [
            f"Avg Shoulder Angle: {metrics['avg_shoulder']:.1f}°",
            f"Left Elbow: {metrics['left_elbow']:.1f}°",
            f"Right Elbow: {metrics['right_elbow']:.1f}°",
            f"Height %: L:{height_pct['left']:.0f}% R:{height_pct['right']:.0f}%",
            f"Symmetry Diff: {self.exercise_detector.get_symmetry_difference():.1f}°", # type: ignore
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.TEXT_COLOR,
                config.FONT_THICKNESS,
            )

    def _draw_feedback(self, frame, feedback_messages: list, feedback_list: list):
        """Draw feedback messages on frame."""
        h, w = frame.shape[:2]

        # Determine overall feedback level (highest priority)
        max_level = FeedbackLevel.GOOD
        for feedback in feedback_list:
            if feedback.level == FeedbackLevel.ERROR:
                max_level = FeedbackLevel.ERROR
                break
            elif feedback.level == FeedbackLevel.WARNING and max_level != FeedbackLevel.ERROR:
                max_level = FeedbackLevel.WARNING

        # Color based on level
        if max_level == FeedbackLevel.ERROR:
            bg_color = config.ERROR_COLOR
            text_color = (255, 255, 255)
        elif max_level == FeedbackLevel.WARNING:
            bg_color = config.WARNING_COLOR
            text_color = (0, 0, 0)
        else:
            bg_color = config.GOOD_COLOR
            text_color = (0, 0, 0)

        # Draw feedback box
        y_start = h - 80
        cv2.rectangle(frame, (0, y_start), (w, h), bg_color, -1)

        # Draw feedback messages
        for i, message in enumerate(feedback_messages[:2]):
            cv2.putText(
                frame,
                message,
                (10, y_start + 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                text_color,
                config.FONT_THICKNESS,
            )

    def _draw_angles(self, frame, landmarks: list):
        """Draw calculated angles on frame."""
        if not landmarks:
            return

        angles = AngleCalculator.extract_all_angles(landmarks)
        h, w = frame.shape[:2]
        x_offset = w - 300
        y_offset = 100

        angle_display = [
            f"Left Elbow: {angles['left_elbow']:.0f}°",
            f"Right Elbow: {angles['right_elbow']:.0f}°",
            f"Left Knee: {angles['left_knee']:.0f}°",
            f"Right Knee: {angles['right_knee']:.0f}°",
        ]

        for i, text in enumerate(angle_display):
            cv2.putText(
                frame,
                text,
                (x_offset, y_offset + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config.TEXT_COLOR,
                1,
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time sports coach powered by pose estimation"
    )
    parser.add_argument(
        "--exercise",
        type=str,
        choices=["squat", "arm_raise"],
        default="squat",
        help="Exercise to coach (default: squat)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Create and run app
    app = SportCoachApp(exercise=args.exercise)
    app.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
