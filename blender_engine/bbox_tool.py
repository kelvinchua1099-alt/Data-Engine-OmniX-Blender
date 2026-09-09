"""Per-frame world-space bounding boxes of an animated asset (port of BoundingBoxTool C++).

Output JSON is the same shape the UE plugin wrote plus a few extra fields:
    {"frames": [{"min": {x,y,z}, "max": {x,y,z}, "frame": i}, ...],
     "asset_path", "action_name", "action_frame_start", "action_frame_end", "units": "meters"}
Frame i of the JSON corresponds to action frame (action_frame_start + i).
"""
import json
import os
import re

import bpy
import numpy as np

from .assets import import_asset, find_action, assign_action, remove_asset, base_name
from .common.log import log


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def bbox_name_for(asset_path, action_name):
    stem = os.path.splitext(os.path.basename(asset_path))[0]
    return f"{sanitize(stem)}__{sanitize(action_name)}.json"


def evaluated_world_vertices(obj, depsgraph):
    """(N,3) float64 world-space vertex positions of the evaluated (deformed) mesh."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        n = len(me.vertices)
        if n == 0:
            return np.zeros((0, 3))
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(n, 3)
        mw = np.array(ev.matrix_world, dtype=np.float64)
        return co @ mw[:3, :3].T + mw[:3, 3]
    finally:
        ev.to_mesh_clear()


def evaluated_triangles(obj, depsgraph):
    """(M,3) int triangle indices of the evaluated mesh (local vertex indexing)."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        m = len(me.loop_triangles)
        if m == 0:
            return np.zeros((0, 3), dtype=np.int64)
        tris = np.empty(m * 3, dtype=np.int64)
        me.loop_triangles.foreach_get("vertices", tris)
        return tris.reshape(m, 3)
    finally:
        ev.to_mesh_clear()


def compute_animation_bboxes(asset, action, scene=None):
    scene = scene or bpy.context.scene
    assign_action(asset, action)
    f0, f1 = action.frame_range
    f0, f1 = int(round(f0)), int(round(f1))
    frames = []
    for i, f in enumerate(range(f0, f1 + 1)):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()
        mn = np.full(3, np.inf)
        mx = np.full(3, -np.inf)
        for obj in asset.meshes:
            pts = evaluated_world_vertices(obj, dg)
            if len(pts):
                mn = np.minimum(mn, pts.min(axis=0))
                mx = np.maximum(mx, pts.max(axis=0))
        if not np.all(np.isfinite(mn)):
            raise RuntimeError("asset has no mesh vertices")
        frames.append({
            "min": {"x": float(mn[0]), "y": float(mn[1]), "z": float(mn[2])},
            "max": {"x": float(mx[0]), "y": float(mx[1]), "z": float(mx[2])},
            "frame": i,
        })
    return frames, (f0, f1)


def export_animation_bboxes(asset_path, action_name, out_dir, scene=None, keep_asset=False):
    asset = import_asset(asset_path, scene=scene)
    action = find_action(asset, action_name)
    frames, (f0, f1) = compute_animation_bboxes(asset, action, scene)
    # file name and stored action name use the base name ('Run', never 'Run.002'): the assembler
    # resolves it again at import time, where the suffix may differ.
    canonical = base_name(action_name or action.name)
    data = {
        "frames": frames,
        "asset_path": os.path.abspath(asset_path),
        "action_name": canonical,
        "action_frame_start": f0,
        "action_frame_end": f1,
        "units": "meters",
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, bbox_name_for(asset_path, canonical))
    with open(out_path, "w") as f:
        json.dump(data, f)
    log(f"bbox for {os.path.basename(asset_path)} / {action.name}: {len(frames)} frames -> {out_path}")
    if not keep_asset:
        remove_asset(asset)
    return out_path
