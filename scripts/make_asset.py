"""Turn an animated glTF/FBX/.blend into a pipeline-ready asset:
normalise the root scale to a target height, save as .blend, export per-action bounding boxes and
register every action in collected_object.json.

    blender -b -P scripts/make_asset.py -- --src assets/fox/Fox.glb --dst assets/fox/fox.blend --height 0.7 \
        --bbox_dir assets/collected_bbox_info --object_json assets/collected_object.json [--actions Run,Walk]
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import script_args  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from blender_engine.assets import import_asset, base_name  # noqa: E402
from blender_engine.bbox_tool import export_animation_bboxes, bbox_name_for  # noqa: E402
from blender_engine.common.log import log  # noqa: E402
from blender_engine.scene_query import DYNAMIC_PROP  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True, help="output .blend")
    p.add_argument("--height", type=float, default=None, help="target height in meters (default: keep)")
    p.add_argument("--bbox_dir", required=True)
    p.add_argument("--object_json", required=True)
    p.add_argument("--actions", default=None, help="comma separated subset of actions (default: all)")
    p.add_argument("--exclude", default=r"^Icosphere",
                   help="regex of object names to delete after import (the glTF importer adds placeholder "
                        "'Icosphere' meshes for some node types; default removes them)")
    a = p.parse_args(script_args())

    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    bpy.ops.wm.read_homefile(use_empty=True)
    asset = import_asset(src)
    if not asset.actions:
        raise SystemExit(f"{src} has no animations")
    if a.exclude:
        rx = re.compile(a.exclude)
        dropped = [o for o in asset.objects if rx.search(o.name)]
        for o in dropped:
            asset.objects.remove(o)
            bpy.data.objects.remove(o, do_unlink=True)
        if dropped:
            log(f"removed {len(dropped)} object(s) matching /{a.exclude}/")
    pts = [o.matrix_world @ Vector(c) for o in asset.meshes for c in o.bound_box]
    h = max(pt.z for pt in pts) - min(pt.z for pt in pts)
    if a.height and h > 0:
        s = a.height / h
        asset.root.scale = (s, s, s)
        log(f"{os.path.basename(src)}: height {h:.3f} -> {a.height} (scale {s:.4f})")
    for o in asset.objects + [asset.root]:
        if DYNAMIC_PROP in o:
            del o[DYNAMIC_PROP]          # re-added on import
    for act in asset.actions:
        act.use_fake_user = True         # keep actions alive in the saved file
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=dst)

    wanted = [n.strip() for n in a.actions.split(",")] if a.actions else None
    names = [base_name(act.name) for act in asset.actions if not wanted or base_name(act.name) in wanted]
    bpy.ops.wm.read_homefile(use_empty=True)
    entries = []
    for n in names:
        export_animation_bboxes(dst, n, a.bbox_dir)
        entries.append({"asset_path": os.path.relpath(dst, os.path.dirname(os.path.abspath(a.object_json))),
                        "action_name": n, "bbox_name": bbox_name_for(dst, n)})

    data = json.load(open(a.object_json)) if os.path.exists(a.object_json) else []
    data = [v for v in data if not (v and v[0]["asset_path"] == entries[0]["asset_path"])] + [entries]
    os.makedirs(os.path.dirname(os.path.abspath(a.object_json)), exist_ok=True)
    with open(a.object_json, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps({"blend": dst, "actions": names, "object_json": a.object_json}))


if __name__ == "__main__":
    main()
