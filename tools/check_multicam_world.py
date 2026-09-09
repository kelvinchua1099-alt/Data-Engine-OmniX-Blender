"""Visualise multi-camera output as world-space point clouds (port of ue_code/tools/check_multicam_world.py).

    python tools/check_multicam_world.py --sequence_path /path/to/SEQUENCE_00000000 \
        --frame_id 0 --camera_ids 0-15 --out_dir /tmp/multicam_check --stride 2 [--use_mask]
All cameras of one frame should reproject onto the same geometry; the exported vertex PLY of the
dynamic objects should sit exactly on top of the foreground points.
"""
import argparse
import json
import os
import sys

import cv2
import Imath
import numpy as np
import OpenEXR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exr_layout import (detect_layout, exr_path_for, vertex_bin_for, camera_json_path,  # noqa: E402
                        load_camera_data, blender_channels, ue_channels)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blender_engine.common.binio import read_vertex_frame, read_faces  # noqa: E402
from compute_foreground_motion import depth_to_world_points_zdepth, depth_to_vis, save_point_ply  # noqa: E402


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_exr_camera(exr_file, available, width, height, camera_id, layout, flip_x, apply_gamma=False):
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    if layout == "blender":
        rgb_names, depth_name, mask_name = blender_channels(available)
    else:
        rgb_names, depth_name, mask_name = ue_channels(camera_id)
    if not all(n and n in available for n in rgb_names):
        print(f"[cam {camera_id:02d}] missing RGB channels, skip")
        return None
    if not depth_name or depth_name not in available:
        print(f"[cam {camera_id:02d}] missing depth channel, skip")
        return None
    bufs = exr_file.channels(rgb_names, FLOAT)
    rgb = np.stack([np.frombuffer(b, dtype=np.float32) for b in bufs], axis=-1).reshape(height, width, 3)
    depth = np.frombuffer(exr_file.channel(depth_name, FLOAT), dtype=np.float32).reshape(height, width)
    if mask_name and mask_name in available:
        mask = np.frombuffer(exr_file.channel(mask_name, FLOAT), dtype=np.float32).reshape(height, width)
    else:
        print(f"[cam {camera_id:02d}] mask channel not found")
        mask = np.zeros((height, width), dtype=np.float32)
    if flip_x:
        rgb, depth, mask = cv2.flip(rgb, 1), cv2.flip(depth, 1), cv2.flip(mask, 1)
    if apply_gamma:
        rgb = np.power(np.clip(rgb, 0, None), 1.0 / 2.2)
    return ((rgb * 255).clip(0, 255).astype(np.uint8), depth.astype(np.float32),
            (mask * 255).clip(0, 255).astype(np.uint8))


def read_exr_multicam(sequence_path, frame_id, camera_ids, layout, flip_x, apply_gamma=False):
    results, shape = {}, None
    opened = {}
    for cid in camera_ids:
        path = exr_path_for(sequence_path, frame_id, cid, layout)
        if path not in opened:
            f = OpenEXR.InputFile(path)
            h = f.header()
            dw = h["dataWindow"]
            opened[path] = (f, set(h["channels"].keys()), dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            print("EXR:", path, "size:", opened[path][2], opened[path][3])
        f, available, w, h = opened[path]
        shape = (h, w)
        data = read_exr_camera(f, available, w, h, cid, layout, flip_x, apply_gamma)
        if data is not None:
            results[cid] = data
    for f, *_ in opened.values():
        f.close()
    return results, shape


def save_vertices_world_ply(vertices_dict, output_path):
    pts = np.concatenate([v for v in vertices_dict.values() if len(v)], axis=0) if vertices_dict else np.zeros((0, 3))
    save_point_ply(pts, np.full((len(pts), 3), (255, 200, 0), dtype=np.uint8), output_path)
    print("saved:", output_path, "vertices:", len(pts))


def save_as_obj(vertices, faces, output_path):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
    print("saved:", output_path, "vertices:", len(vertices), "faces:", len(faces))


def save_meshes_as_obj(vertices_dict, faces_dict, output_folder):
    ensure_dir(output_folder)
    for name, verts in vertices_dict.items():
        faces = faces_dict.get(name, np.zeros((0, 3), dtype=np.int64))
        if len(verts) and len(faces):
            safe = "".join(x for x in name if x.isalnum() or x in "._- ")
            save_as_obj(verts, faces, os.path.join(output_folder, f"{safe}.obj"))
        else:
            print(f"Warning: missing data for {name}")


def parse_camera_ids(text):
    if text == "all":
        return None
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence_path", required=True)
    p.add_argument("--frame_id", type=int, required=True)
    p.add_argument("--camera_ids", default="0")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--use_mask", action="store_true")
    p.add_argument("--flip_x", action="store_true", default=None)
    p.add_argument("--no_flip_x", action="store_false", dest="flip_x")
    p.add_argument("--apply_gamma", action="store_true")
    a = p.parse_args()
    ensure_dir(a.out_dir)

    layout = detect_layout(a.sequence_path)
    flip_x = (layout == "ue") if a.flip_x is None else a.flip_x
    cam_json = camera_json_path(a.sequence_path)
    vertex_bin = vertex_bin_for(a.sequence_path, a.frame_id)
    faces_bin = os.path.join(a.sequence_path, "faces.bin")

    with open(cam_json) as f:
        cam = json.load(f)
    camera_ids = parse_camera_ids(a.camera_ids)
    if camera_ids is None:
        camera_ids = list(range(len(cam["fov"])))

    exr_data, shape = read_exr_multicam(a.sequence_path, a.frame_id, camera_ids, layout, flip_x, a.apply_gamma)
    fov, K_all, c2w_all = load_camera_data(cam_json, image_shape=shape)

    _, actors = read_vertex_frame(vertex_bin)
    vertices_dict = {k: np.asarray(v, dtype=np.float32).reshape(-1, 3) for k, v in actors.items()}
    if os.path.exists(faces_bin):
        faces_dict = {k: np.asarray(v, dtype=np.int64).reshape(-1, 3) for k, v in read_faces(faces_bin).items()}
        save_meshes_as_obj(vertices_dict, faces_dict, os.path.join(a.out_dir, f"frame_{a.frame_id:04d}_meshes"))
    save_vertices_world_ply(vertices_dict, os.path.join(a.out_dir, f"frame_{a.frame_id:04d}_vertices_world.ply"))

    for cid in camera_ids:
        if cid not in exr_data:
            continue
        rgb, depth, mask = exr_data[cid]
        K, c2w = K_all[cid], c2w_all[cid, a.frame_id]
        cam_dir = os.path.join(a.out_dir, f"cam_{cid:02d}")
        ensure_dir(cam_dir)
        cv2.imwrite(os.path.join(cam_dir, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(cam_dir, "mask.png"), mask)
        cv2.imwrite(os.path.join(cam_dir, "depth_vis.png"), depth_to_vis(depth))
        pts, colors = depth_to_world_points_zdepth(depth, rgb, K, c2w, mask=mask if a.use_mask else None,
                                                  stride=a.stride)
        save_point_ply(pts, colors, os.path.join(cam_dir, f"frame_{a.frame_id:04d}_cam_{cid:02d}_world_zdepth.ply"))
        print(f"[cam {cid:02d}] hfov {float(fov[cid]):.2f}, points {len(pts)}")
        print(f"[cam {cid:02d}] c2w:\n{c2w}")
    print("Done. Output:", a.out_dir)


if __name__ == "__main__":
    main()
