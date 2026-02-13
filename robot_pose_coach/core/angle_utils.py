"""
angle_utils.py — Geometry helpers for joint-angle computation.

All functions are pure (no side-effects) and operate on numpy arrays,
making them easy to unit-test independently of MediaPipe.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

import config
from core.pose_detector import Landmarks


def angle_between_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    Compute the angle ∠ABC at vertex B (in degrees).

    Parameters:
        a, b, c — 2D or 3D coordinate arrays.

    Returns:
        Angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def spine_inclination(
    shoulder_mid: np.ndarray, hip_mid: np.ndarray
) -> float:
    """
    Compute forward lean of the torso as the angle between the
    shoulder→hip vector and the vertical (Y-down in image coords).

    Returns:
        Angle in degrees.  0 = perfectly upright, 90 = horizontal.
    """
    spine_vec = hip_mid - shoulder_mid          # Points downward when upright
    vertical  = np.array([0.0, 1.0, 0.0])      # Y-down in image space

    cos_a = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-8)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def compute_all_angles(
    landmarks: Landmarks,
    use_3d: bool = False,
) -> Dict[str, float]:
    """
    Compute every joint angle defined in config.JOINT_ANGLES, plus
    spine inclination.

    Parameters:
        landmarks — dict of {name: LandmarkPoint} from PoseDetector.
        use_3d    — if True, use (x, y, z); otherwise (x, y) only.

    Returns:
        Dict mapping angle name → degrees.
    """
    angles: Dict[str, float] = {}

    for jdef in config.JOINT_ANGLES:
        pt_a = landmarks.get(jdef.a)
        pt_b = landmarks.get(jdef.b)
        pt_c = landmarks.get(jdef.c)

        if pt_a is None or pt_b is None or pt_c is None:
            continue

        # Minimum visibility gate — ignore low-confidence points.
        if min(pt_a.visibility, pt_b.visibility, pt_c.visibility) < 0.5:
            continue

        if use_3d:
            a, b, c = pt_a.to_array(), pt_b.to_array(), pt_c.to_array()
        else:
            a = np.array([pt_a.x, pt_a.y])
            b = np.array([pt_b.x, pt_b.y])
            c = np.array([pt_c.x, pt_c.y])

        angles[jdef.name] = angle_between_points(a, b, c)

    # ── Spine inclination (midpoint of shoulders → midpoint of hips) ──
    ls, rs = landmarks.get("LEFT_SHOULDER"), landmarks.get("RIGHT_SHOULDER")
    lh, rh = landmarks.get("LEFT_HIP"), landmarks.get("RIGHT_HIP")
    if all(p is not None for p in (ls, rs, lh, rh)):
        s_mid = (ls.to_array()[:2] + rs.to_array()[:2]) / 2
        h_mid = (lh.to_array()[:2] + rh.to_array()[:2]) / 2
        # Pad to 3D for the helper (z=0)
        s3 = np.array([s_mid[0], s_mid[1], 0.0])
        h3 = np.array([h_mid[0], h_mid[1], 0.0])
        angles["spine_inclination"] = spine_inclination(s3, h3)

    return angles


def angle_diff(current: float, target: float) -> float:
    """Signed difference: positive means current > target."""
    return current - target
