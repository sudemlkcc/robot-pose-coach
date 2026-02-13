"""
ml_pipeline.py — Scaffolding for the future ML-based posture classifier.

This module provides:
  1. DataCollector  — records angle snapshots with labels for training data.
  2. PostureClassifier — interface that a trained model will implement.

Workflow (when you're ready):
  1. Run the app in "record" mode → DataCollector saves CSV files.
  2. Train a model (sklearn, PyTorch, etc.) on the CSV data.
  3. Implement PostureClassifier.load() and .predict().
  4. Plug it into the exercise detector alongside rule-based logic.
"""

from __future__ import annotations

import csv
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1.  DATA COLLECTION
# ─────────────────────────────────────────────────────────────────

class DataCollector:
    """
    Records per-frame angle dictionaries with a label to a CSV file
    for later supervised learning.

    Usage:
        collector = DataCollector("squat_data")
        collector.start_session()
        # Inside your frame loop:
        collector.record(angles_dict, label="correct")
        # When done:
        collector.end_session()
    """

    def __init__(self, exercise_name: str, output_dir: str = "data") -> None:
        self.exercise = exercise_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None
        self._file = None
        self._fieldnames: Optional[List[str]] = None

    def start_session(self) -> Path:
        """Open a new CSV file for this recording session."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"{self.exercise}_{timestamp}.csv"
        self._file = open(filepath, "w", newline="")
        self._fieldnames = None  # Will be set on first record
        return filepath

    def record(self, angles: Dict[str, float], label: str) -> None:
        """Append one row: all angle values + label + timestamp."""
        if self._file is None:
            raise RuntimeError("Call start_session() first.")

        if self._fieldnames is None:
            self._fieldnames = sorted(angles.keys()) + ["label", "timestamp"]
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        row = {k: f"{v:.2f}" for k, v in angles.items()}
        row["label"] = label
        row["timestamp"] = f"{time.time():.4f}"
        self._writer.writerow(row)

    def end_session(self) -> None:
        """Flush and close the CSV file."""
        if self._file:
            self._file.close()
            self._file = None


# ─────────────────────────────────────────────────────────────────
# 2.  CLASSIFIER INTERFACE
# ─────────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    """Output of the posture classifier."""
    label: str              # e.g. "correct", "back_lean", "shallow_squat"
    confidence: float       # 0..1
    corrections: List[str]  # Human-readable suggestions


class PostureClassifier(ABC):
    """
    Abstract classifier interface.

    Implement this with your trained model (sklearn, PyTorch, etc.).
    The exercise detector can then call classifier.predict(angles)
    alongside or instead of rule-based checks.
    """

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load a trained model from disk."""
        ...

    @abstractmethod
    def predict(self, angles: Dict[str, float]) -> Prediction:
        """Classify the current pose angles."""
        ...


# ─────────────────────────────────────────────────────────────────
# 3.  EXAMPLE SKLEARN PLACEHOLDER
# ─────────────────────────────────────────────────────────────────

class SklearnClassifier(PostureClassifier):
    """
    Example implementation using scikit-learn (uncomment when ready).

    Expects a model saved with joblib that has .predict() and
    .predict_proba() methods.
    """

    def __init__(self) -> None:
        self._model = None
        self._feature_names: List[str] = []
        self._label_map: Dict[int, str] = {}

    def load(self, model_path: str) -> None:
        # import joblib
        # data = joblib.load(model_path)
        # self._model = data["model"]
        # self._feature_names = data["feature_names"]
        # self._label_map = data["label_map"]
        raise NotImplementedError("Train a model first! See data/ folder for CSVs.")

    def predict(self, angles: Dict[str, float]) -> Prediction:
        if self._model is None:
            raise RuntimeError("Model not loaded.")

        # Build feature vector in the correct order
        features = np.array([[angles.get(f, 0.0) for f in self._feature_names]])

        pred_idx = self._model.predict(features)[0]
        proba = self._model.predict_proba(features)[0]

        label = self._label_map.get(pred_idx, "unknown")
        confidence = float(proba[pred_idx])

        # Map labels to correction messages (customise per exercise)
        correction_map = {
            "back_lean": ["Keep your back straighter"],
            "shallow_squat": ["Go deeper — bend your knees more"],
            "knee_cave": ["Push your knees outward"],
            "correct": [],
        }

        return Prediction(
            label=label,
            confidence=confidence,
            corrections=correction_map.get(label, []),
        )
