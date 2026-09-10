"""Scene construction (port of construct_scene/main.py).

Samples dynamic objects, a ground region, trajectories and 16 cameras against the static scene,
then writes one JSON spec per sequence:  <anno_base>/SEQUENCE_xxxxxxxx/sequence_xxxxxxxx.json
Nothing is spawned in the Blender scene here; `assemble.build_sequence` rebuilds it from the spec.
"""
import json
import math
import os
import random
import re

from mathutils import Vector

from .camera import place_camera, CAMERA_POS_LOOKAT_COMBINATION
from .common.convert import opencv_c2w
from .common.geom import (interpolate_bounding_boxes, get_bbox_size, get_bbox_center, rescale_bbox_list,
                          get_bbox_list_radius_and_height)
from .common.log import log, warn, error
from .scene_analysis import update_density_map, sample_nav_region, generate_initial_point
from .trajectory import generate_trajectories

SPEC_FORMAT = "omnix-blender-v1"

DEFAULT_CONFIG = {
    "MAX_ATTEMPT_NAV": 2000,
    "MAX_ATTEMPT_SCENE": 10,
    "MAX_ATTEMPT_INITIAL_GROUND_POINT": 1000,
    "MAX_ATTEMPT_TRAJECTORY": 1000,
    "MOVEMENT_SCALE": 1.5,
    "MAX_GROUND_TILT": 10.0,
    "CAMERA_FOV_RANGE": (60.0, 100.0),
    "CAMERA_TILT_RANGE": (0.0, 30.0),
    "CAMERA_DISTANCE_SCALE_RANGE": (1.0, 1.2),
    "CAMERA_TRAJ_SPHERE_ANGLE_RANGE": (45.0, 225.0),
    "CAMERA_TRAJ_TRANSLATION_SCALE_RANGE": (0.5, 0.75),
    "CAMERA_TRAJ_PLANE_TILE_RANGE": (0.0, 10.0),
    "NOISE_SCALE_RANGE": (0.0, 0.005),
    "NOISE_ANGLE_RANGE": (0.0, 1.0),
    "IMAGE_RATIO": (16, 9),
    "MAX_FRAME": 36,
    "NUM_OBJECT_RANGE": (1, 3),
    "WARMUP_FRAMES": 4,
    "FPS": 24,
    "NUM_CAMERAS": 16,
    "MIN_VALID_CAMERA_RATIO": 0.9,
    "STATIC_TRAJECTORY_PROB": 0.5,
    "INDOOR_PROB_OUTDOOR_SCENE": 0.05,
    "RANDOM_OBJECT_SCALE_RANGE": (0.2, 5.0),   # UE: random.uniform(0.2, 5); use ~(0.8, 1.25) for realistic sizes
}


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# ----------------------------------------------------------------------------- meta (read_meta.py)
def read_meta_file(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)


def convert_meta_to_bbox_list(meta_list):
    bbox_list, frame_number_list = [], []
    for frames in meta_list:
        frames = sorted(frames, key=lambda x: x["frame"])
        traj = [[[fd["min"]["x"], fd["min"]["y"], fd["min"]["z"]],
                 [fd["max"]["x"], fd["max"]["y"], fd["max"]["z"]]] for fd in frames]
        bbox_list.append(traj[:-1])              # UE: sampled keys = frames + 1, drop the last
        frame_number_list.append(len(traj) - 1)
    return bbox_list, frame_number_list


def sample_bbox_sequences(bbox_list, frame_number_list, max_frame, start_frame=0):
    out_b, out_n, out_s = [], [], []
    for traj, n in zip(bbox_list, frame_number_list):
        if n <= max_frame:
            out_b.append(traj)
            out_n.append(n)
            out_s.append(start_frame)
        else:
            s = random.randint(0, n - max_frame)
            out_b.append(traj[s:s + max_frame])
            out_n.append(max_frame)
            out_s.append(s)
    return out_b, out_n, out_s


def load_object_data(object_file):
    """object_data.json: list of objects, each a list of animation variants:
        {"asset_path": "...", "action_name": "Run", "bbox_name": "dog__Run.json"}
    Relative asset paths are resolved against the json's directory."""
    base = os.path.dirname(os.path.abspath(object_file))
    with open(object_file, "r") as f:
        data = json.load(f)
    for variants in data:
        for v in variants:
            p = v["asset_path"]
            if not os.path.isabs(p):
                v["asset_path"] = os.path.normpath(os.path.join(base, p))
    return data


# ----------------------------------------------------------------------------- main loop
def construct_scene(bvh, scene_info, object_data, bbox_folder, anno_base_path, expect_sequence_num=3,
                    sequence_group_index=0, sequence_log_path=None, scene_type="outdoor", config=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(config or {})

    wb_min = scene_info["world_bound_min"]
    wb_max = scene_info["world_bound_max"]
    world_bound_min = Vector((wb_min["x"], wb_min["y"], wb_min["z"]))
    world_bound_max = Vector((wb_max["x"], wb_max["y"], wb_max["z"]))
    density_map = [row[:] for row in scene_info["density_map"]]
    masks = [scene_info.get(k) for k in ("ppv_mask", "reflection_mask", "camera_mask") if scene_info.get(k)]
    density_map = update_density_map(density_map, masks, threshold_percent=75, multiply_factor=6.0)
    cell_size = float(scene_info["cell_size"])

    if scene_type == "indoor":
        indoor_prob = 1.0
        max_portion, min_portion = 1 / 2, 1 / 6
    else:
        indoor_prob = cfg["INDOOR_PROB_OUTDOOR_SCENE"]
        max_portion, min_portion = 1 / 6, 1 / 10

    combos = CAMERA_POS_LOOKAT_COMBINATION[:int(cfg["NUM_CAMERAS"])]
    scene_size = Vector((cell_size, cell_size, cell_size)).length
    max_frame = int(cfg["MAX_FRAME"])
    movement_scale = cfg["MOVEMENT_SCALE"]
    written = []
    sequence_index = 0

    for nav_index in range(int(cfg["MAX_ATTEMPT_NAV"])):
        lo, hi = cfg["NUM_OBJECT_RANGE"]
        num_object = random.choices(range(lo, hi + 1), weights=range(lo, hi + 1))[0]
        num_object = min(num_object, len(object_data))
        sampled_objects = random.sample(object_data, num_object)
        sampled_anims = [random.choice(v) for v in sampled_objects]

        metas = [read_meta_file(os.path.join(bbox_folder, a["bbox_name"])) for a in sampled_anims]
        bbox_list, frame_number_list = convert_meta_to_bbox_list([m["frames"] for m in metas])
        bbox_list, frame_number_list, start_frame_list = sample_bbox_sequences(bbox_list, frame_number_list, max_frame)
        on_the_ground = [True] * num_object

        imported_scales = []
        for obj_bbox in bbox_list:
            ratio = get_bbox_size(obj_bbox[0]) / scene_size
            if ratio > max_portion:
                imported_scales.append(max_portion / ratio)
            elif ratio < min_portion:
                imported_scales.append(min_portion / ratio)
            else:
                imported_scales.append(1.0)

        anno_index = sequence_index + expect_sequence_num * sequence_group_index
        nav_location, nav_radius = sample_nav_region(bvh, world_bound_min, world_bound_max, density_map, cell_size)

        random_scale = random.uniform(*cfg["RANDOM_OBJECT_SCALE_RANGE"])
        random_scales = [s * random_scale for s in imported_scales]
        success = False

        for attempt_scene in range(int(cfg["MAX_ATTEMPT_SCENE"])):
            new_scales = [s * (0.9 ** attempt_scene) for s in random_scales]
            rescaled = rescale_bbox_list(bbox_list, new_scales)
            interpolated = interpolate_bounding_boxes(rescaled, frame_number_list)
            max_radius, max_height = get_bbox_list_radius_and_height(rescaled)
            search_radius = max_radius * (1 + movement_scale * 0.2)
            search_height = max_height * (1 + movement_scale * 0.2) * 0.5
            log(f"[nav {nav_index} scene {attempt_scene}] scales {[round(s, 3) for s in new_scales]} "
                f"search_radius {search_radius:.3f} search_height {search_height:.3f}")

            ground = generate_initial_point(
                bvh, nav_location, nav_search_radius=nav_radius, search_radius=search_radius,
                height_offset=search_height, max_attempts=int(cfg["MAX_ATTEMPT_INITIAL_GROUND_POINT"]),
                ground_tilt_angle=cfg["MAX_GROUND_TILT"], indoor_prob=indoor_prob)
            if ground is None:
                continue

            traj_radius = max_radius * movement_scale * math.sqrt(num_object)
            traj_height = max_height * movement_scale * 0.5
            traj_result = generate_trajectories(
                bvh, interpolated, trajectory_radius=traj_radius, trajectory_height=traj_height,
                on_the_ground=on_the_ground, initial_ground_point=ground,
                max_attempts=int(cfg["MAX_ATTEMPT_TRAJECTORY"]), static_prob=cfg["STATIC_TRAJECTORY_PROB"])
            if traj_result is None:
                continue
            trajectories, global_bboxes = traj_result

            cam_result = place_camera(
                bvh, global_bboxes, camera_fov_range=cfg["CAMERA_FOV_RANGE"],
                camera_elevation_range=cfg["CAMERA_TILT_RANGE"],
                camera_distance_scale_range=cfg["CAMERA_DISTANCE_SCALE_RANGE"],
                camera_traj_sphere_angle_range=cfg["CAMERA_TRAJ_SPHERE_ANGLE_RANGE"],
                camera_traj_translation_scale_range=cfg["CAMERA_TRAJ_TRANSLATION_SCALE_RANGE"],
                camera_traj_plane_tile_range=cfg["CAMERA_TRAJ_PLANE_TILE_RANGE"],
                noise_scale_range=cfg["NOISE_SCALE_RANGE"], noise_angle_range=cfg["NOISE_ANGLE_RANGE"],
                image_ratio=cfg["IMAGE_RATIO"], min_valid_ratio=cfg["MIN_VALID_CAMERA_RATIO"],
                combinations=combos)
            if cam_result is None:
                continue
            fov_all, pos_all, rot_all, look_all, check_all = cam_result

            objects = []
            for oi, anim in enumerate(sampled_anims):
                meta = metas[oi]
                objects.append({
                    "name": f"obj{oi:02d}_{sanitize(os.path.splitext(os.path.basename(anim['asset_path']))[0])}",
                    "asset_path": anim["asset_path"],
                    "action_name": meta.get("action_name", anim.get("action_name")),
                    "bbox_name": anim["bbox_name"],
                    "action_frame_start": meta.get("action_frame_start", 0),
                    "scale": new_scales[oi],
                    "start_frame": start_frame_list[oi],
                    "frame_number": frame_number_list[oi],
                    "bbox_center0": list(get_bbox_center(interpolated[oi][0])),
                    "trajectory": [[p.x, p.y, p.z] for p in trajectories[oi]],
                    "global_bbox": [b.to_list() for b in global_bboxes[oi]],
                    "pass_index": 1,
                })

            c2w = [[opencv_c2w(pos_all[ci][f], rot_all[ci][f]) for f in range(len(pos_all[ci]))]
                   for ci in range(len(pos_all))]
            spec = {
                "format": SPEC_FORMAT,
                "units": "meters",
                "fps": int(cfg["FPS"]),
                "warmup_frames": int(cfg["WARMUP_FRAMES"]),
                "max_frame": max(frame_number_list),
                "image_ratio": list(cfg["IMAGE_RATIO"]),
                "scene_type": scene_type,
                "blend_file": scene_info.get("blend_file"),
                "initial_ground_point": [ground.x, ground.y, ground.z],
                "camera_modes": [list(c) for c in combos],
                "fov": fov_all,
                "camera_position": pos_all,
                "rotation": rot_all,
                "camera_look_at": look_all,
                "check_camera_result": check_all,
                "c2w": c2w,
                "objects": objects,
                # legacy keys (UE annotation layout)
                "skeletal_mesh": [o["asset_path"] for o in objects],
                "anim_sequence": [o["action_name"] for o in objects],
                "imported_scales": new_scales,
                "start_frame": start_frame_list,
            }
            seq_dir = os.path.join(anno_base_path, f"SEQUENCE_{anno_index:08d}")
            os.makedirs(seq_dir, exist_ok=True)
            spec_path = os.path.join(seq_dir, f"sequence_{anno_index:08d}.json")
            with open(spec_path, "w") as f:
                json.dump(spec, f, indent=1)
            written.append(spec_path)
            log(f"sequence {anno_index:08d} written: {num_object} object(s), {len(fov_all)} cameras -> {spec_path}")
            if sequence_log_path:
                with open(sequence_log_path, "a+") as f:
                    f.write(spec_path + "\n")
            success = True
            sequence_index += 1
            break

        if not success:
            warn(f"scene validation failed for nav attempt {nav_index}")
        if sequence_index >= expect_sequence_num:
            break

    if sequence_index < expect_sequence_num:
        error(f"only {sequence_index}/{expect_sequence_num} sequences constructed")
    return written
