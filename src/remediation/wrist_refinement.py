"""Eye-in-hand target refinement using the calibrated physical wrist camera.

TableCam/YOLO remains authoritative for class, count and coarse identity.  A
wrist VLM supplies only a local bounding box.  Its centre is intersected with
the already known horizontal object plane, then rejected if it is too far from
the TableCam candidate.  No tray coordinates or simulator poses enter here.
"""
from __future__ import annotations

import json
import cv2
import numpy as np
import sys

from src.remediation import wrist_camera_model as M


FINGERTIP_TCP_J6_M = np.array([0.293, 0.0, 0.0], dtype=float)
WRIST_VIEW_DISTANCE_M = 0.300
MAX_WRIST_CORRECTION_M = 0.040
MAX_SUPPORT_PLANE_DELTA_M = 0.015
MAX_SUPPORT_PLANE_RMS_M = 0.004


def calibrated_home_joints_deg():
    """Physical home supplied with the wrist-camera hand-eye calibration."""
    raw = json.loads(M.CALIBRATION.read_text())
    joints = np.asarray(raw["wrist_depth"]["solved_at_joints"], dtype=float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError("wrist calibration must contain six finite home joints")
    return joints


def _flange_pose(joints):
    # Keep this module importable from Isaac, where the client package is not
    # necessarily on sys.path until runtime.
    try:
        from calibration.geometry import flange_pose
    except ModuleNotFoundError:
        client_root = str(M.CALIBRATION.parent.parent)
        if client_root not in sys.path:
            sys.path.insert(0, client_root)
        from calibration.geometry import flange_pose
    return flange_pose(joints)


def flange_rotation(joints):
    return _flange_pose(joints)[:3, :3]


def fingertip_tcp_from_joints(joints):
    """Fingertip TCP implied by the calibrated physical-arm joint chain."""
    T = _flange_pose(joints)
    return T[:3, 3] + T[:3, :3] @ FINGERTIP_TCP_J6_M


def wrist_colour_pose(joints):
    """Return T_base_colour for the joints recorded with a wrist snapshot."""
    return _flange_pose(joints) @ M.T_FLANGE_COLOUR


def project_base_to_wrist(point_base_m, joints, served_size=(1920, 1080)):
    point = np.asarray(point_base_m, dtype=float).reshape(3)
    T = wrist_colour_pose(joints)
    pc = T[:3, :3].T @ (point - T[:3, 3])
    if pc[2] <= 1e-4:
        raise ValueError("target is behind the wrist colour camera")
    K = M.colour_K(served_size)
    pinhole = (K @ pc)[:2] / pc[2]
    return M.distort_px([pinhole], served_size)[0]


def wrist_pixel_to_plane(pixel, plane_z_m, joints, served_size=(1920, 1080)):
    """Backproject a served wrist RGB pixel onto a base-frame horizontal plane."""
    W, H = tuple(map(int, served_size))
    uv = np.asarray(pixel, dtype=float).reshape(2)
    if not np.isfinite(uv).all() or not (0 <= uv[0] < W and 0 <= uv[1] < H):
        raise ValueError("wrist pixel is outside the served colour image")
    pin = M.undistort_px([uv], served_size)[0]
    K = M.colour_K(served_size)
    ray_cam = np.array([(pin[0] - K[0, 2]) / K[0, 0],
                        (pin[1] - K[1, 2]) / K[1, 1], 1.0])
    T = wrist_colour_pose(joints)
    origin, ray = T[:3, 3], T[:3, :3] @ ray_cam
    if abs(ray[2]) < 1e-6:
        raise ValueError("wrist ray is parallel to the target plane")
    distance = (float(plane_z_m) - origin[2]) / ray[2]
    if distance <= 0:
        raise ValueError("target plane is behind the wrist camera")
    return origin + distance * ray


def box_center(box_xyxy):
    box = np.asarray(box_xyxy, dtype=float).reshape(4)
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("VLM box must be finite XYXY with positive area")
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def render_wrist_refinement_annotation(image_bgr, class_name, vlm_result,
                                       expected_pixel, refinement=None):
    """Render the wrist VLM target and its coarse projected anchor.

    The selected VLM box and its centre are green.  The projected TableCam
    candidate is a blue X, and a blue segment makes the image-space residual
    visible.  Gray boxes retain any additional VLM proposals for diagnosis.
    ``refinement`` may be omitted so the raw model result can be persisted
    before geometric validation.
    """
    annotated = np.asarray(image_bgr).copy()
    if annotated.ndim != 3 or annotated.shape[2] != 3:
        raise ValueError("wrist annotation requires a BGR image")

    for candidate in vlm_result.get("all_boxes_xyxy", []):
        x0, y0, x1, y1 = [int(round(value)) for value in candidate]
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (160, 160, 160), 1)

    x0, y0, x1, y1 = [int(round(value))
                      for value in vlm_result["bbox_xyxy"]]
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 4)
    centre = (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2)))
    expected = tuple(int(round(value)) for value in expected_pixel)
    cv2.line(annotated, expected, centre, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(annotated, centre, (0, 255, 0),
                   cv2.MARKER_CROSS, 24, 3)
    cv2.drawMarker(annotated, expected, (255, 0, 0),
                   cv2.MARKER_TILTED_CROSS, 24, 3)

    latency = float(vlm_result.get("latency_s", 0.0))
    if refinement is not None and "correction_norm_m" in refinement:
        residual = f"base correction {1000.0 * float(refinement['correction_norm_m']):.1f}mm"
    else:
        residual_px = float(np.linalg.norm(np.asarray(centre) - np.asarray(expected)))
        residual = f"image residual {residual_px:.1f}px"
    labels = [
        f"{class_name} | VLM {latency:.2f}s | {residual}",
        "green +=VLM centre | blue X=coarse projected target",
    ]
    for row, label in enumerate(labels):
        origin = (30, 55 + 38 * row)
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0) if row == 0 else (255, 255, 255),
                    2, cv2.LINE_AA)
    return annotated


def fit_local_support_plane(points_base_m, colour_pixels, target_box_xyxy,
                            target_xy_m, nominal_z_m, *, min_points=24):
    """Fit the table/paper surrounding a wrist-RGB target box.

    The loose cap itself can be absent from passive stereo.  This deliberately
    uses an image-space annulus around its VLM box, rejects points far from the
    calibrated table height, and fits ``z = ax + by + c`` in robot-base space.
    It raises when the result is not safe to use; callers then retain the
    configured work-surface plane.
    """
    points = np.asarray(points_base_m, dtype=float).reshape(-1, 3)
    pixels = np.asarray(colour_pixels, dtype=float).reshape(-1, 2)
    if len(points) != len(pixels):
        raise ValueError("wrist depth points and projected colour pixels must align")
    box = np.asarray(target_box_xyxy, dtype=float).reshape(4)
    target_xy = np.asarray(target_xy_m, dtype=float).reshape(2)
    if (not np.isfinite(points).all() or not np.isfinite(pixels).all() or
            not np.isfinite(box).all() or not np.isfinite(target_xy).all() or
            box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("support-plane inputs must be finite and the box non-empty")

    width, height = box[2] - box[0], box[3] - box[1]
    outer = np.array([box[0] - width, box[1] - height,
                      box[2] + width, box[3] + height])
    inner = np.array([box[0] - 0.15 * width, box[1] - 0.15 * height,
                      box[2] + 0.15 * width, box[3] + 0.15 * height])
    in_outer = ((pixels[:, 0] >= outer[0]) & (pixels[:, 0] <= outer[2]) &
                (pixels[:, 1] >= outer[1]) & (pixels[:, 1] <= outer[3]))
    in_inner = ((pixels[:, 0] >= inner[0]) & (pixels[:, 0] <= inner[2]) &
                (pixels[:, 1] >= inner[1]) & (pixels[:, 1] <= inner[3]))
    near_nominal = np.abs(points[:, 2] - float(nominal_z_m)) <= 0.020
    selected = points[in_outer & ~in_inner & near_nominal]
    if len(selected) < int(min_points):
        raise ValueError(f"only {len(selected)} valid wrist-depth support points")

    # Two robust least-squares passes discard cap edges, gripper pixels and
    # isolated stereo mismatches while retaining a gently non-level paper.
    kept = selected
    for _ in range(2):
        A = np.column_stack((kept[:, 0], kept[:, 1], np.ones(len(kept))))
        coeff, *_ = np.linalg.lstsq(A, kept[:, 2], rcond=None)
        residual = kept[:, 2] - A @ coeff
        median = np.median(residual)
        good = np.abs(residual - median) <= MAX_SUPPORT_PLANE_RMS_M
        if good.sum() < int(min_points):
            raise ValueError(f"only {int(good.sum())} inlier wrist-depth support points")
        kept = kept[good]
    A = np.column_stack((kept[:, 0], kept[:, 1], np.ones(len(kept))))
    coeff, *_ = np.linalg.lstsq(A, kept[:, 2], rcond=None)
    residual = kept[:, 2] - A @ coeff
    rms = float(np.sqrt(np.mean(residual * residual)))
    slope = float(np.linalg.norm(coeff[:2]))
    support_z = float(coeff[0] * target_xy[0] + coeff[1] * target_xy[1] + coeff[2])
    if rms > MAX_SUPPORT_PLANE_RMS_M:
        raise ValueError(f"wrist support-plane RMS is {rms * 1000:.1f} mm")
    if slope > np.tan(np.radians(5.0)):
        raise ValueError(f"wrist support-plane tilt is {np.degrees(np.arctan(slope)):.1f} deg")
    if abs(support_z - float(nominal_z_m)) > MAX_SUPPORT_PLANE_DELTA_M:
        raise ValueError(
            f"wrist support plane differs from calibration by "
            f"{(support_z - float(nominal_z_m)) * 1000:.1f} mm")
    return {
        "support_z_base_m": support_z,
        "nominal_support_z_base_m": float(nominal_z_m),
        "support_z_correction_m": support_z - float(nominal_z_m),
        "point_count": int(len(kept)),
        "rms_m": rms,
        "tilt_deg": float(np.degrees(np.arctan(slope))),
        "source": "wrist stereo plane fit around VLM box",
    }


def refine_from_vlm_box(box_xyxy, expected_base_m, plane_z_m, snapshot_joints,
                        served_size=(1920, 1080), max_correction_m=MAX_WRIST_CORRECTION_M):
    """Convert a VLM box into a bounded base-frame target correction."""
    expected = np.asarray(expected_base_m, dtype=float).reshape(3)
    refined = wrist_pixel_to_plane(
        box_center(box_xyxy), plane_z_m, snapshot_joints, served_size)
    correction = refined[:2] - expected[:2]
    distance = float(np.linalg.norm(correction))
    if not np.isfinite(distance) or distance > float(max_correction_m):
        raise ValueError(
            f"wrist refinement is {distance * 1000:.1f} mm from its TableCam candidate; "
            f"limit is {float(max_correction_m) * 1000:.1f} mm")
    refined[2] = float(plane_z_m)
    return {
        "base_pos_m": refined.tolist(),
        "bbox_xyxy": list(map(float, box_xyxy)),
        "pixel": box_center(box_xyxy).tolist(),
        "correction_base_m": [float(correction[0]), float(correction[1]), 0.0],
        "correction_norm_m": distance,
        "plane_z_m": float(plane_z_m),
        "source": "wrist LocateAnything box-centre ray/plane intersection",
    }


def wrist_view_tcp(target_base_m, command_rotation, distance_m=WRIST_VIEW_DISTANCE_M,
                   j6_rotation=None, centre_camera=True):
    """Fingertip TCP base position that centres a target in the wrist camera.

    ``command_rotation`` is the controller TCP rotation.  The conversion to the
    J6 frame mirrors the shared controller without importing it (and creating a
    circular dependency).
    """
    if j6_rotation is None:
        R = np.asarray(command_rotation, dtype=float).reshape(3, 3)
        R_j6 = np.column_stack((R[:, 2], R[:, 1], -R[:, 0]))
    else:
        R_j6 = np.asarray(j6_rotation, dtype=float).reshape(3, 3)
    R_cam = R_j6 @ M.T_FLANGE_COLOUR[:3, :3]
    optical_axis = R_cam[:, 2]
    eye = np.asarray(target_base_m, dtype=float).reshape(3) - optical_axis * float(distance_m)
    flange = eye - R_j6 @ M.T_FLANGE_COLOUR[:3, 3]
    tcp = flange + R_j6 @ FINGERTIP_TCP_J6_M
    if not centre_camera:
        # Fast coarse hover: put the fingertip TCP above the TableCam target
        # while retaining the Z needed for the requested optical distance.
        # Hand-eye projection supplies the off-centre expected image pixel, so
        # a camera-only XY centring move is unnecessary.
        tcp[:2] = np.asarray(target_base_m, dtype=float).reshape(3)[:2]
    return tcp
