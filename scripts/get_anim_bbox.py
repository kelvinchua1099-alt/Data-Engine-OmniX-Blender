"""Per-frame bounding boxes of an animated asset (replaces BoundingBoxTool + tools/get_anim_bbox.py).

    blender -b -P scripts/get_anim_bbox.py -- --asset assets/dog.glb --action Run \
        --out_dir object_data/collected_bbox_info
Omit --action to export every action found in the asset.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import script_args  # noqa: E402

import bpy  # noqa: E402

from blender_engine.assets import import_asset, remove_asset  # noqa: E402
from blender_engine.bbox_tool import export_animation_bboxes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", required=True)
    p.add_argument("--action", default=None)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args(script_args())

    if a.action:
        export_animation_bboxes(a.asset, a.action, a.out_dir)
        return
    probe = import_asset(a.asset)
    names = [act.name for act in probe.actions]
    remove_asset(probe)
    if not names:
        raise SystemExit(f"no actions in {a.asset}")
    for n in names:
        export_animation_bboxes(a.asset, n, a.out_dir)


if __name__ == "__main__":
    main()
