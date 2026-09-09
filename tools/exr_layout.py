"""Shared helpers for the offline tools: sequence layout detection, EXR channel discovery,
camera loading. Works for both the original UE output and the Blender output.

UE layout      : images/-0004.exr .. 0035.exr, one EXR holds all cameras
                 (FinalImage_CusCamera_XX.R, FinalImageMovieRenderQueue_WorldDepth_CusCamera_XX.R, ...)
Blender layout : images/cam_XX/-0004.exr .. 0035.exr, one EXR per camera
                 (<ViewLayer>.Combined.R, <ViewLayer>.Depth.Z, <ViewLayer>.IndexOB.X)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blender_engine.common.exr_header import exr_channel_names, pick_blender_channels  # noqa: E402,F401


def detect_layout(sequence_path):
    images_dir = os.path.join(sequence_path, "images")
    if os.path.isdir(os.path.join(images_dir, "cam_00")):
        return "blender"
    return "ue"


def frame_name(frame_id):
    return f"{frame_id:04d}" if frame_id >= 0 else f"-{abs(frame_id):04d}"


def camera_json_path(sequence_path):
    name = os.path.basename(os.path.normpath(sequence_path))
    return os.path.join(sequence_path, name.lower() + ".json")


def exr_path_for(sequence_path, frame_id, camera_id, layout):
    images_dir = os.path.join(sequence_path, "images")
    if layout == "blender":
        return os.path.join(images_dir, f"cam_{camera_id:02d}", frame_name(frame_id) + ".exr")
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(".exr"))
    idx = frame_id + 4
    if idx >= len(files):
        raise IndexError(f"frame_id + 4 out of range: {frame_id} + 4 >= {len(files)}")
    return os.path.join(images_dir, files[idx])


def vertex_bin_for(sequence_path, frame_id):
    return os.path.join(sequence_path, "vertex_data", f"frame_{frame_id:04d}.bin")


def blender_channels(available):
    """Pick rgb / depth / mask channel names from a Blender multi-layer EXR header."""
    return pick_blender_channels(available)


def ue_channels(camera_id):
    cam = f"{camera_id:02d}"
    if camera_id == 0:
        rgb = ["R", "G", "B"]
    else:
        rgb = [f"FinalImage_CusCamera_{cam}.R", f"FinalImage_CusCamera_{cam}.G", f"FinalImage_CusCamera_{cam}.B"]
    return rgb, f"FinalImageMovieRenderQueue_WorldDepth_CusCamera_{cam}.R", f"FinalImageCustomMask_CusCamera_{cam}.R"


def compute_camera_intrinsic(hfovs, image_shape):
    hfovs = np.asarray(hfovs, dtype=np.float32)
    h, w = image_shape
    fx = w * 0.5 / np.tan(np.radians(hfovs * 0.5))
    K = np.zeros((len(hfovs), 3, 3), dtype=np.float32)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fx
    K[:, 0, 2] = w / 2.0
    K[:, 1, 2] = h / 2.0
    K[:, 2, 2] = 1.0
    return K


def compute_camera_c2w_ue(camera_locations, camera_rotations):
    """Original UE roll/pitch/yaw (left-handed, cm) -> OpenCV c2w. Kept verbatim for UE data."""
    camera_locations = np.asarray(camera_locations, dtype=np.float32)
    rot = np.radians(np.asarray(camera_rotations, dtype=np.float32))
    roll, pitch, yaw = rot[:, 0], rot[:, 1], rot[:, 2]
    z = np.zeros_like(yaw)
    o = np.ones_like(yaw)
    R_yaw = np.stack([np.stack([np.cos(yaw), -np.sin(yaw), z], 1), np.stack([np.sin(yaw), np.cos(yaw), z], 1),
                      np.stack([z, z, o], 1)], 1)
    R_pitch = np.stack([np.stack([np.cos(pitch), z, -np.sin(pitch)], 1), np.stack([z, o, z], 1),
                        np.stack([np.sin(pitch), z, np.cos(pitch)], 1)], 1)
    R_roll = np.stack([np.stack([o, z, z], 1), np.stack([z, np.cos(roll), -np.sin(roll)], 1),
                       np.stack([z, np.sin(roll), np.cos(roll)], 1)], 1)
    R_ue = R_yaw @ R_pitch @ R_roll
    forward = R_ue @ np.array([1, 0, 0], dtype=np.float32)
    right = R_ue @ np.array([0, 1, 0], dtype=np.float32)
    up = R_ue @ np.array([0, 0, 1], dtype=np.float32)
    R_cv = np.stack([-right, -up, forward], axis=2)
    c2w = np.zeros((len(camera_locations), 4, 4), dtype=np.float32)
    c2w[:, :3, :3] = R_cv
    c2w[:, :3, 3] = camera_locations
    c2w[:, 3, 3] = 1.0
    return c2w


def load_camera_data(camera_json_path, image_shape):
    """Returns (fov[num_cam], K[num_cam,3,3], c2w[num_cam,num_frame,4,4]) in OpenCV convention.
    Blender specs carry an explicit c2w; UE annotations are converted from roll/pitch/yaw."""
    with open(camera_json_path, "r") as f:
        cam = json.load(f)
    fov = np.asarray(cam["fov"], dtype=np.float32)
    K = compute_camera_intrinsic(fov, image_shape)
    if "c2w" in cam:
        c2w = np.asarray(cam["c2w"], dtype=np.float32)
    else:
        pos = np.asarray(cam["camera_position"], dtype=np.float32)
        rot = np.asarray(cam["rotation"], dtype=np.float32)
        num_camera, num_frame = rot.shape[:2]
        c2w = compute_camera_c2w_ue(pos.reshape(-1, 3), rot.reshape(-1, 3)).reshape(num_camera, num_frame, 4, 4)
    return fov, K, c2w

