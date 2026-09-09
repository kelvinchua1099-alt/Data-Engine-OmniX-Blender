"""Analyse the static environment (replaces the SceneSetup automation test).

    blender -b env.blend -P scripts/run_scene_setup.py -- --config config/template.json \
        --output logs/scene_info/DEBUG/scene_info.json --status logs/scene_info/DEBUG/status.txt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import script_args  # noqa: E402

from blender_engine.scene_analysis import run_scene_setup  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="json with cell_count / exclude_num / z_percent / max_nav_extent")
    p.add_argument("--output", required=True)
    p.add_argument("--status", default=None)
    p.add_argument("--cell_count", type=int, default=None)
    p.add_argument("--exclude_num", type=int, default=None)
    p.add_argument("--z_percent", type=float, default=None)
    p.add_argument("--max_nav_extent", type=float, default=None)
    a = p.parse_args(script_args())

    cfg = {}
    if a.config:
        with open(a.config) as f:
            cfg.update(json.load(f))
    for k in ("cell_count", "exclude_num", "z_percent", "max_nav_extent"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    cfg["output_path"] = a.output
    cfg["status_path"] = a.status or os.path.join(os.path.dirname(a.output), "status.txt")
    run_scene_setup(cfg)


if __name__ == "__main__":
    main()
