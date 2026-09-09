"""Assemble one sequence spec in Blender and render it (replaces Movie Render Queue + VertexTracker).

    blender -b env.blend -P scripts/run_render.py -- --sequence_dir render_output/DEBUG/SEQUENCE_00000000 \
        --width 1280 --height 720 --engine CYCLES --samples 64 [--device GPU|CPU] [--cameras 0-15] \
        [--frames=-4-35] [--no_images] [--no_vertices] [--save_blend out.blend]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import script_args  # noqa: E402

import bpy  # noqa: E402

from blender_engine.assemble import build_sequence, clear_dynamic  # noqa: E402
from blender_engine.render import configure_render, render_sequence  # noqa: E402


def parse_range(text, default):
    if text is None:
        return default
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        # allow negative numbers: "-4-35" means -4..35
        if part.count("-") >= 1 and not part.lstrip("-").isdigit():
            body = part
            neg = body.startswith("-")
            if neg:
                body = body[1:]
            lo, hi = body.split("-", 1)
            lo = -int(lo) if neg else int(lo)
            out.extend(range(lo, int(hi) + 1))
        else:
            out.append(int(part))
    return out


def find_spec(seq_dir):
    cands = sorted(glob.glob(os.path.join(seq_dir, "sequence_*.json")))
    if not cands:
        raise SystemExit(f"no sequence_*.json in {seq_dir}")
    return cands[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence_dir", required=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--device", default="GPU")
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--cameras", default=None, help="e.g. 0-15 or 0,3,5")
    p.add_argument("--frames", default=None,
                   help="data frames, e.g. 0,5,10 or 0-35; ranges starting negative need the '=' form: --frames=-4-35")
    p.add_argument("--no_images", action="store_true")
    p.add_argument("--no_vertices", action="store_true")
    p.add_argument("--use_dof", action="store_true")
    p.add_argument("--save_blend", default=None)
    a = p.parse_args(script_args())

    with open(find_spec(a.sequence_dir)) as f:
        spec = json.load(f)
    scene = bpy.context.scene
    clear_dynamic(scene)
    built = build_sequence(scene, spec, use_dof=a.use_dof)
    vl = configure_render(scene, a.width, a.height, engine=a.engine, samples=a.samples, device=a.device,
                          denoise=not a.no_denoise)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend), copy=True)
    meta = render_sequence(scene, built, a.sequence_dir,
                           camera_ids=parse_range(a.cameras, None),
                           frames=parse_range(a.frames, None),
                           render_images=not a.no_images, export_vertices=not a.no_vertices, view_layer=vl)
    print(json.dumps({"rendered_frames": meta["frames"], "cameras": meta["cameras"]}))


if __name__ == "__main__":
    main()
