"""Construct sequences (replaces construct_scene/main.py).

    blender -b env.blend -P scripts/run_construct.py -- \
        --scene_info logs/scene_info/DEBUG/scene_info.json \
        --object_file object_data/collected_object.json \
        --bbox_folder object_data/collected_bbox_info \
        --anno_base render_output/DEBUG --expect_sequence_num 3 --group_index 0 \
        --scene_type outdoor [--seed 0] [--max_frame 36] [--num_cameras 16] [--config overrides.json]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import script_args  # noqa: E402

import bpy  # noqa: E402

from blender_engine.construct_scene import construct_scene, load_object_data  # noqa: E402
from blender_engine.scene_query import SceneBVH  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_info", required=True)
    p.add_argument("--object_file", required=True)
    p.add_argument("--bbox_folder", required=True)
    p.add_argument("--anno_base", required=True)
    p.add_argument("--expect_sequence_num", type=int, default=3)
    p.add_argument("--group_index", type=int, default=0)
    p.add_argument("--sequence_log_folder", default=None)
    p.add_argument("--scene_type", default="outdoor", choices=["outdoor", "indoor"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_frame", type=int, default=None)
    p.add_argument("--num_cameras", type=int, default=None)
    p.add_argument("--config", default=None, help="json overriding DEFAULT_CONFIG keys")
    a = p.parse_args(script_args())

    if a.seed is not None:
        random.seed(a.seed)
    cfg = {}
    if a.config:
        with open(a.config) as f:
            cfg.update(json.load(f))
    if a.max_frame:
        cfg["MAX_FRAME"] = a.max_frame
    if a.num_cameras:
        cfg["NUM_CAMERAS"] = a.num_cameras

    with open(a.scene_info) as f:
        scene_info = json.load(f)
    object_data = load_object_data(a.object_file)

    log_path = None
    if a.sequence_log_folder:
        os.makedirs(a.sequence_log_folder, exist_ok=True)
        log_path = os.path.join(a.sequence_log_folder, f"QUEUE_{a.group_index:04d}.txt")
        open(log_path, "w").close()

    bvh = SceneBVH(bpy.context.scene)
    written = construct_scene(bvh, scene_info, object_data, a.bbox_folder, a.anno_base,
                              expect_sequence_num=a.expect_sequence_num, sequence_group_index=a.group_index,
                              sequence_log_path=log_path, scene_type=a.scene_type, config=cfg)
    print(json.dumps({"written": written}))


if __name__ == "__main__":
    main()
