"""Multi-camera rendering + per-frame vertex export (replaces Movie Render Queue + VertexTracker).

Output layout of one sequence directory:
    images/cam_XX/-0004.exr .. -0001.exr   warm-up frames (static scene, dynamic objects hidden)
    images/cam_XX/0000.exr .. NNNN.exr     active frames
    vertex_data/frame_NNNN.bin             VTXD, world-space vertices of every dynamic object
    faces.bin                              FACE, triangle topology (constant)
    render_meta.json                       resolution, engine, EXR channel names, frame list

Each EXR is a Blender multi-layer EXR written straight from the Render Result, so channels are
    <ViewLayer>.Combined.R/G/B/A   <ViewLayer>.Depth.Z   <ViewLayer>.IndexOB.X
Depth is planar camera-space z (same meaning as UE WorldDepth); IndexOB is the object pass index
(dynamic objects carry pass_index 1) and serves as the foreground mask.
"""
import json
import os

import bpy
import numpy as np

from .bbox_tool import evaluated_world_vertices, evaluated_triangles
from .common.binio import write_vertex_frame, write_faces
from .common.exr_header import exr_channel_names, pick_blender_channels
from .common.log import log, warn

GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")


def frame_name(f):
    return f"{f:04d}" if f >= 0 else f"-{abs(f):04d}"


def enable_gpu(preferred=None):
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is None:
        return None
    cp = prefs.preferences
    backends = [preferred] if preferred else list(GPU_BACKENDS)
    for backend in backends:
        try:
            cp.compute_device_type = backend
            cp.get_devices()
            gpus = [d for d in cp.devices if d.type != "CPU"]
            if gpus:
                for d in cp.devices:
                    d.use = True
                log(f"cycles GPU backend {backend}: {[d.name for d in gpus]}")
                return backend
        except Exception:  # noqa: BLE001
            continue
    cp.compute_device_type = "NONE"
    return None


def configure_render(scene, width, height, engine="CYCLES", samples=64, device="GPU", denoise=True,
                     motion_blur=False, view_layer=None):
    r = scene.render
    r.resolution_x = int(width)
    r.resolution_y = int(height)
    r.resolution_percentage = 100
    r.engine = engine
    r.use_compositing = False
    r.use_sequencer = False
    r.use_file_extension = True
    r.use_overwrite = True
    r.use_placeholder = False
    r.use_motion_blur = bool(motion_blur)
    ims = r.image_settings
    if hasattr(ims, "media_type"):
        # Blender >= 5.0: multilayer EXR is media_type MULTI_LAYER_IMAGE + OPEN_EXR; the default is a
        # multi-part file, the legacy interleaved single-part layout is what OpenEXR.InputFile reads.
        ims.media_type = "MULTI_LAYER_IMAGE"
        ims.file_format = "OPEN_EXR_MULTILAYER"
        if hasattr(ims, "use_exr_interleave"):
            ims.use_exr_interleave = True
    else:
        ims.file_format = "OPEN_EXR_MULTILAYER"
    ims.color_depth = "32"
    ims.exr_codec = "ZIP"

    if engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = bool(denoise)
        scene.cycles.use_adaptive_sampling = True
        if device.upper() == "GPU" and enable_gpu() is not None:
            scene.cycles.device = "GPU"
        else:
            scene.cycles.device = "CPU"
    elif engine.startswith("BLENDER_EEVEE"):
        scene.eevee.taa_render_samples = int(samples)
        warn("EEVEE does not write an object-index pass; the foreground mask channel will be missing")

    vl = view_layer or scene.view_layers[0]
    vl.use_pass_combined = True
    vl.use_pass_z = True
    vl.use_pass_object_index = True
    return vl


def channel_names(view_layer_name, sample_exr=None):
    """Channel names as written by this Blender; read from a rendered file when one is available."""
    if sample_exr and os.path.exists(sample_exr):
        try:
            avail = exr_channel_names(sample_exr)
            rgb, depth, mask = pick_blender_channels(avail)
            return {"rgb": rgb, "depth": depth, "mask": mask, "all": avail}
        except Exception as e:  # noqa: BLE001
            warn(f"could not parse EXR header of {sample_exr}: {e}")
    return {
        "rgb": [f"{view_layer_name}.Combined.R", f"{view_layer_name}.Combined.G", f"{view_layer_name}.Combined.B"],
        "depth": f"{view_layer_name}.Depth.Z",
        "mask": f"{view_layer_name}.Object Index.X",
    }


def export_vertex_frame(built, f, seq_dir, depsgraph):
    actors = []
    for asset in built.assets:
        parts = [evaluated_world_vertices(m, depsgraph) for m in asset.meshes]
        pts = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3))
        actors.append((asset.name, pts))
    path = os.path.join(seq_dir, "vertex_data", f"frame_{f:04d}.bin")
    write_vertex_frame(path, f, actors)
    return path


def export_faces(built, seq_dir, depsgraph):
    actors = []
    for asset in built.assets:
        offset = 0
        faces = []
        for m in asset.meshes:
            tris = evaluated_triangles(m, depsgraph)
            faces.append(tris + offset)
            offset += len(evaluated_world_vertices(m, depsgraph))
        allf = np.concatenate(faces, axis=0) if faces else np.zeros((0, 3), dtype=np.int64)
        actors.append((asset.name, allf))
    path = os.path.join(seq_dir, "faces.bin")
    write_faces(path, actors)
    return path


def render_sequence(scene, built, seq_dir, camera_ids=None, frames=None, render_images=True,
                    export_vertices=True, view_layer=None):
    """frames: iterable of data-frame indices f (negative = warm-up); default all."""
    vl = view_layer or scene.view_layers[0]
    images_dir = os.path.join(seq_dir, "images")
    camera_ids = list(camera_ids) if camera_ids is not None else list(range(len(built.cameras)))
    all_frames = list(range(-built.warmup, built.max_frame))
    frames = [f for f in (frames if frames is not None else all_frames) if -built.warmup <= f < built.max_frame]
    rendered = []
    for f in frames:
        b = built.frame_to_blender(f)
        built.set_dynamic_visible(f >= 0)
        scene.frame_set(b)
        if f >= 0 and export_vertices:
            dg = bpy.context.evaluated_depsgraph_get()
            export_vertex_frame(built, f, seq_dir, dg)
            if f == 0 or not os.path.exists(os.path.join(seq_dir, "faces.bin")):
                export_faces(built, seq_dir, dg)
        if render_images:
            for ci in camera_ids:
                scene.camera = built.cameras[ci]
                scene.render.filepath = os.path.join(images_dir, f"cam_{ci:02d}", frame_name(f))
                bpy.ops.render.render(write_still=True)
            log(f"frame {f:+d} rendered ({len(camera_ids)} cameras)")
        rendered.append(f)

    meta = {
        "layout": "per_camera",
        "units": "meters",
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "engine": scene.render.engine,
        "samples": int(getattr(scene.cycles, "samples", 0)) if scene.render.engine == "CYCLES" else int(scene.eevee.taa_render_samples),
        "warmup_frames": built.warmup,
        "max_frame": built.max_frame,
        "frames": rendered,
        "cameras": camera_ids,
        "view_layer": vl.name,
        "channels": channel_names(vl.name, sample_exr=(
            os.path.join(images_dir, f"cam_{camera_ids[0]:02d}", frame_name(rendered[0]) + ".exr")
            if rendered and camera_ids and render_images else None)),
        "depth_convention": "planar camera-space z, meters",
        "mask_convention": "object pass index (>0.5 = dynamic foreground)",
        "flip_x": False,
    }
    with open(os.path.join(seq_dir, "render_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
