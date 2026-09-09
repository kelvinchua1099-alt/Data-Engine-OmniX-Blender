"""Dense foreground trajectory from a source frame to a target frame (port of ue_code/tools/compute_foreground_motion.py).

Works on both the original UE output and the Blender output (layout auto-detected, see exr_layout.py).

    python tools/compute_foreground_motion.py --sequence_path /path/to/SEQUENCE_00000000 \
        --start_frame 0 --target_frame 10 --camera_id 0 --out_dir /tmp/fg_motion_check --stride 1
"""
import argparse
import collections
import os
import sys

import cv2
import Imath
import numpy as np
import OpenEXR
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exr_layout import (detect_layout, exr_path_for, vertex_bin_for, camera_json_path,  # noqa: E402
                        load_camera_data, blender_channels, ue_channels)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blender_engine.common.binio import read_vertex_frame, read_faces  # noqa: E402


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def read_binary_vertex_file(file_path):
    _, actors = read_vertex_frame(file_path)
    return {k: np.asarray(v, dtype=np.float32).reshape(-1, 3) for k, v in actors.items()}


def read_binary_faces(file_path):
    return {k: np.asarray(v, dtype=np.int32).reshape(-1, 3) for k, v in read_faces(file_path).items()}


def read_exr_camera(exr_file, available, width, height, camera_id, layout, flip_x, apply_gamma=False):
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    if layout == "blender":
        rgb_names, depth_name, mask_name = blender_channels(available)
    else:
        rgb_names, depth_name, mask_name = ue_channels(camera_id)
    if not all(n and n in available for n in rgb_names):
        raise KeyError(f"Missing RGB channels for camera {camera_id}: {rgb_names} in {sorted(available)}")
    if not depth_name or depth_name not in available:
        raise KeyError(f"Missing depth channel for camera {camera_id}: {depth_name} in {sorted(available)}")
    if not mask_name or mask_name not in available:
        raise KeyError(f"Missing foreground mask channel for camera {camera_id}: {mask_name} in {sorted(available)}")

    bufs = exr_file.channels(rgb_names, FLOAT)
    rgb = np.stack([np.frombuffer(b, dtype=np.float32) for b in bufs], axis=-1).reshape(height, width, 3)
    depth = np.frombuffer(exr_file.channel(depth_name, FLOAT), dtype=np.float32).reshape(height, width)
    mask = np.frombuffer(exr_file.channel(mask_name, FLOAT), dtype=np.float32).reshape(height, width)
    if flip_x:
        rgb, depth, mask = cv2.flip(rgb, 1), cv2.flip(depth, 1), cv2.flip(mask, 1)
    if apply_gamma:
        rgb = np.power(np.clip(rgb, 0, None), 1.0 / 2.2)
    rgb_u8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    mask_u8 = (mask * 255.0).clip(0, 255).astype(np.uint8)
    return rgb_u8, depth.astype(np.float32), mask_u8


def read_exr_single_camera(exr_path, camera_id, layout, flip_x, apply_gamma=False):
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    available = set(header["channels"].keys())
    data = read_exr_camera(exr_file, available, width, height, camera_id, layout, flip_x, apply_gamma)
    exr_file.close()
    return data, (height, width)


def depth_to_world_points_zdepth(depth, rgb, K, c2w, mask=None, stride=1, min_depth=0.01, max_depth=100000.0):
    h, w = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    if stride > 1:
        u_grid, v_grid = u_grid[::stride, ::stride], v_grid[::stride, ::stride]
        depth_s, rgb_s = depth[::stride, ::stride], rgb[::stride, ::stride]
        mask_s = mask[::stride, ::stride] if mask is not None else None
    else:
        depth_s, rgb_s, mask_s = depth, rgb, mask

    u = u_grid.reshape(-1).astype(np.float32)
    v = v_grid.reshape(-1).astype(np.float32)
    z = depth_s.reshape(-1).astype(np.float32)
    colors = rgb_s.reshape(-1, 3)

    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    finite = z[np.isfinite(z)]
    if finite.size > 0:
        max_value = np.max(finite)
        sky = np.isclose(z, max_value, rtol=1e-6, atol=max(1e-3, abs(max_value) * 1e-7))
        valid &= ~sky
    if mask_s is not None:
        valid &= mask_s.reshape(-1) > 0

    u, v, z, colors = u[valid], v[valid], z[valid], colors[valid]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    points_cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1).astype(np.float32)
    R = c2w[:3, :3].astype(np.float32)
    t = c2w[:3, 3].astype(np.float32)
    return points_cam @ R.T + t[None, :], colors


def merge_multiple_object(vertex_dict_list, faces_dict):
    names = [n for n in faces_dict.keys() if n in vertex_dict_list[0]]
    faces_all, offset = [], 0
    for n in names:
        faces = np.asarray(faces_dict[n], dtype=np.int32)
        verts = vertex_dict_list[0][n]
        if len(verts) == 0 or len(faces) == 0:
            continue
        faces_all.append(faces + offset)
        offset += verts.shape[0]
    if not faces_all:
        raise ValueError("No valid faces found after merging objects.")
    faces_all = np.vstack(faces_all).astype(np.int32)
    frames = []
    for vd in vertex_dict_list:
        parts = []
        for n in names:
            if n not in vd:
                raise KeyError(f"Missing actor {n} in vertex frame.")
            parts.append(vd[n].astype(np.float32))
        frames.append(np.vstack(parts))
    return np.stack(frames, axis=0), faces_all


def vectorized_assign_points_to_faces(vertices, faces, query_points):
    if query_points.ndim == 1:
        query_points = query_points[None, :]
    nbrs = NearestNeighbors(n_neighbors=1).fit(vertices)
    _, indices = nbrs.kneighbors(query_points)
    nearest = indices[:, 0]

    vertex_face_map = collections.defaultdict(list)
    for fi, face in enumerate(faces):
        for vi in face:
            vertex_face_map[int(vi)].append(fi)

    max_faces = max(len(vertex_face_map[v]) for v in nearest)
    candidate = np.zeros((len(query_points), max_faces), dtype=np.int32)
    valid = np.zeros((len(query_points), max_faces), dtype=bool)
    for i, vi in enumerate(nearest):
        fl = vertex_face_map[int(vi)]
        candidate[i, :len(fl)] = fl
        valid[i, :len(fl)] = True

    fv = vertices[faces[candidate[valid]]]
    pts = np.repeat(query_points[:, None, :], max_faces, axis=1)[valid]
    normals = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    degenerate = norms.flatten() < 1e-8
    normals = normals / np.where(degenerate[:, None], 1.0, norms)
    signed = np.sum((pts - fv[:, 0]) * normals, axis=1)
    dist_plane = np.abs(signed)
    proj = pts - signed[:, None] * normals

    v0, v1, v2 = fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], proj - fv[:, 0]
    d00, d01, d11 = np.sum(v0 * v0, 1), np.sum(v0 * v1, 1), np.sum(v1 * v1, 1)
    d20, d21 = np.sum(v2 * v0, 1), np.sum(v2 * v1, 1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    bv = (d11 * d20 - d01 * d21) / denom
    bw = (d00 * d21 - d01 * d20) / denom
    bu = 1.0 - bv - bw
    eps = 1e-6
    inside = (bu >= -eps) & (bv >= -eps) & (bw >= -eps)
    inside[degenerate] = False

    dist_r = np.full((len(query_points), max_faces), np.inf, dtype=np.float32)
    dist_r[valid] = np.where(degenerate, np.inf, dist_plane)
    inside_r = np.zeros((len(query_points), max_faces), dtype=bool)
    inside_r[valid] = inside
    dist_r[~inside_r] += 1e6
    mi = np.argmin(dist_r, axis=1)
    assignments = candidate[np.arange(len(query_points)), mi]
    md = dist_r[np.arange(len(query_points)), mi]
    return assignments, np.minimum(md - 1e6, md)


def compute_start_to_target_points(vertices_mat, faces, start_index, target_index, foreground_points, assignments):
    fv = vertices_mat[:, faces[assignments]]
    sfv, tfv = fv[start_index], fv[target_index]
    dist_sq = np.sum((sfv - foreground_points[:, None, :]) ** 2, axis=-1)
    w = 1.0 / (dist_sq + 1e-10)
    w = w / w.sum(axis=1, keepdims=True)
    sc = np.sum(w[:, :, None] * sfv, axis=1)
    tc = np.sum(w[:, :, None] * tfv, axis=1)
    qs, qt = sfv - sc[:, None, :], tfv - tc[:, None, :]
    cov = np.matmul(qt.transpose(0, 2, 1), w[:, :, None] * qs)
    U, _, Vh = np.linalg.svd(cov)
    det = np.linalg.det(U @ Vh)
    U[det < 0, :, -1] *= -1
    R = np.matmul(U, Vh)
    return (tc + np.matmul(R, (foreground_points - sc)[:, :, None]).squeeze(-1)).astype(np.float32)


def _write_ply_header(f, n_vertex, n_face=0, n_edge=0):
    f.write("ply\nformat ascii 1.0\n")
    f.write(f"element vertex {n_vertex}\nproperty float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    if n_face:
        f.write(f"element face {n_face}\nproperty list uchar int vertex_indices\n")
    if n_edge:
        f.write(f"element edge {n_edge}\nproperty int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    f.write("end_header\n")


def save_point_ply(points, colors, output_path):
    ensure_dir(os.path.dirname(output_path))
    colors = np.full((len(points), 3), 255, dtype=np.uint8) if colors is None else colors.astype(np.uint8)
    with open(output_path, "w") as f:
        _write_ply_header(f, len(points))
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_mesh_ply(vertices, faces, output_path, color=(180, 180, 180)):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        _write_ply_header(f, len(vertices), n_face=len(faces))
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_motion_lines_ply(start_points, target_points, output_path):
    ensure_dir(os.path.dirname(output_path))
    points = np.concatenate([start_points, target_points], axis=0)
    n = len(start_points)
    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colors[:n] = (255, 60, 60)
    colors[n:] = (60, 180, 255)
    with open(output_path, "w") as f:
        _write_ply_header(f, len(points), n_edge=n)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        for i in range(n):
            f.write(f"{i} {i + n} 255 255 0\n")


def depth_to_vis(depth):
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1e6)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], 1), np.percentile(depth[valid], 99)
    d = (np.clip(depth, lo, hi) - lo) / max(hi - lo, 1e-6)
    return cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence_path", required=True)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--target_frame", type=int, required=True)
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--mask_threshold", type=int, default=1)
    p.add_argument("--flip_x", action="store_true", default=None, help="default: UE yes, Blender no")
    p.add_argument("--no_flip_x", action="store_false", dest="flip_x")
    p.add_argument("--apply_gamma", action="store_true")
    a = p.parse_args()
    ensure_dir(a.out_dir)

    layout = detect_layout(a.sequence_path)
    flip_x = (layout == "ue") if a.flip_x is None else a.flip_x
    cam_json = camera_json_path(a.sequence_path)
    start_exr = exr_path_for(a.sequence_path, a.start_frame, a.camera_id, layout)
    start_bin = vertex_bin_for(a.sequence_path, a.start_frame)
    target_bin = vertex_bin_for(a.sequence_path, a.target_frame)
    faces_bin = os.path.join(a.sequence_path, "faces.bin")
    print("layout:", layout, "| camera_json:", cam_json)
    print("start_exr:", start_exr, "| start_bin:", start_bin, "| target_bin:", target_bin)

    (rgb, depth, mask), shape = read_exr_single_camera(start_exr, a.camera_id, layout, flip_x, a.apply_gamma)
    _, K_all, c2w_all = load_camera_data(cam_json, image_shape=shape)
    K = K_all[a.camera_id]
    start_c2w = c2w_all[a.camera_id, a.start_frame]
    fg_mask = (mask > a.mask_threshold).astype(np.uint8) * 255

    start_points, start_colors = depth_to_world_points_zdepth(depth, rgb, K, start_c2w, mask=fg_mask, stride=a.stride)
    if len(start_points) == 0:
        raise SystemExit("no foreground points found (empty mask?)")

    vertices_mat, faces = merge_multiple_object(
        [read_binary_vertex_file(start_bin), read_binary_vertex_file(target_bin)], read_binary_faces(faces_bin))
    start_vertices, target_vertices = vertices_mat[0], vertices_mat[1]

    assignments, min_distances = vectorized_assign_points_to_faces(start_vertices, faces, start_points)
    target_points = compute_start_to_target_points(vertices_mat, faces, 0, 1, start_points, assignments)

    s, t, c = a.start_frame, a.target_frame, a.camera_id
    save_mesh_ply(start_vertices, faces, os.path.join(a.out_dir, f"frame_{s:04d}_vertices_mesh.ply"), (220, 80, 80))
    save_mesh_ply(target_vertices, faces, os.path.join(a.out_dir, f"frame_{t:04d}_vertices_mesh.ply"), (80, 160, 255))
    save_point_ply(start_points, start_colors, os.path.join(a.out_dir, f"frame_{s:04d}_foreground_points.ply"))
    save_point_ply(target_points, start_colors, os.path.join(a.out_dir, f"frame_{s:04d}_to_{t:04d}_foreground_points.ply"))
    save_motion_lines_ply(start_points, target_points, os.path.join(a.out_dir, f"frame_{s:04d}_to_{t:04d}_motion_lines.ply"))
    cv2.imwrite(os.path.join(a.out_dir, f"frame_{s:04d}_cam_{c:02d}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(a.out_dir, f"frame_{s:04d}_cam_{c:02d}_mask.png"), fg_mask)
    cv2.imwrite(os.path.join(a.out_dir, f"frame_{s:04d}_cam_{c:02d}_depth_vis.png"), depth_to_vis(depth))
    np.savez_compressed(
        os.path.join(a.out_dir, f"frame_{s:04d}_to_{t:04d}_foreground_motion.npz"),
        start_points=start_points.astype(np.float32), target_points=target_points.astype(np.float32),
        colors=start_colors.astype(np.uint8), assignments=assignments.astype(np.int32),
        assignment_distances=min_distances.astype(np.float32), start_vertices=start_vertices.astype(np.float32),
        target_vertices=target_vertices.astype(np.float32), faces=faces.astype(np.int32),
        K=K.astype(np.float32), start_c2w=start_c2w.astype(np.float32))

    print("foreground points:", len(start_points))
    nn_dist, _ = NearestNeighbors(n_neighbors=1).fit(start_vertices).kneighbors(start_points)
    print(f"sanity: median distance from reprojected foreground points to nearest exported vertex: "
          f"{float(np.median(nn_dist)):.4f} (scene units)")
    print("saved to:", a.out_dir)


if __name__ == "__main__":
    main()
