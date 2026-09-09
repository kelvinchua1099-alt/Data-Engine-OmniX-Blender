"""End-to-end smoke test on synthetic data. Run headless:

    /Applications/Blender.app/Contents/MacOS/Blender -b -P tests/smoke_test.py -- --out /tmp/omnix_smoke \
        [--engine CYCLES] [--samples 8] [--cameras 2] [--res 192x108] [--frames all|few] [--seed 0]

Builds a synthetic environment .blend and an animated character .blend, then runs
bbox_tool -> scene_analysis -> construct_scene -> assemble -> render and validates the outputs.
"""
import argparse
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bpy  # noqa: E402
import bmesh  # noqa: E402
import numpy as np  # noqa: E402

from blender_engine.assemble import build_sequence, clear_dynamic  # noqa: E402
from blender_engine.bbox_tool import export_animation_bboxes, bbox_name_for  # noqa: E402
from blender_engine.common.binio import read_vertex_frame, read_faces  # noqa: E402
from blender_engine.construct_scene import construct_scene, load_object_data  # noqa: E402
from blender_engine.render import configure_render, render_sequence  # noqa: E402
from blender_engine.scene_analysis import run_scene_setup  # noqa: E402
from blender_engine.scene_query import SceneBVH  # noqa: E402
from blender_engine.common.exr_header import exr_channel_names  # noqa: E402


def script_args():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def new_file():
    bpy.ops.wm.read_homefile(use_empty=True)


def add_material(obj, color):
    mat = bpy.data.materials.new(f"mat_{obj.name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    obj.data.materials.append(mat)


# ----------------------------------------------------------------------------- synthetic assets
def build_character_asset(path):
    new_file()
    scene = bpy.context.scene
    arm_data = bpy.data.armatures.new("CharArm")
    arm = bpy.data.objects.new("CharArm", arm_data)
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    lower = arm_data.edit_bones.new("lower")
    lower.head, lower.tail = (0, 0, 0), (0, 0, 0.9)
    upper = arm_data.edit_bones.new("upper")
    upper.head, upper.tail = (0, 0, 0.9), (0, 0, 1.8)
    upper.parent = lower
    upper.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.22, radius2=0.18, depth=1.8)
    for z in (-0.6, -0.3, 0.0, 0.3, 0.6):
        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, z), plane_no=(0, 0, 1))
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 0.9))
    mesh = bpy.data.meshes.new("CharMesh")
    bm.to_mesh(mesh)
    bm.free()
    body = bpy.data.objects.new("CharBody", mesh)
    scene.collection.objects.link(body)
    body.parent = arm
    add_material(body, (0.9, 0.2, 0.1))

    vg_l = body.vertex_groups.new(name="lower")
    vg_u = body.vertex_groups.new(name="upper")
    for v in mesh.vertices:
        w = min(1.0, max(0.0, (v.co.z - 0.7) / 0.4))
        vg_u.add([v.index], w, "REPLACE")
        vg_l.add([v.index], 1.0 - w, "REPLACE")
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = arm

    pb_upper = arm.pose.bones["upper"]
    pb_lower = arm.pose.bones["lower"]
    pb_upper.rotation_mode = "XYZ"
    for frame, angle, bob in ((1, 0.0, 0.0), (12, 60.0, 0.12), (24, 0.0, 0.0)):
        pb_upper.rotation_euler = (math.radians(angle), 0, 0)
        pb_upper.keyframe_insert("rotation_euler", frame=frame)
        pb_lower.location = (0, bob, 0)  # bone-local Y is along the bone -> vertical bob
        pb_lower.keyframe_insert("location", frame=frame)
    arm.animation_data.action.name = "Wave"
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return path


def build_environment(path, seed=0):
    new_file()
    random.seed(seed)
    scene = bpy.context.scene
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    add_material(ground, (0.35, 0.45, 0.25))
    for i in range(8):
        sx, sy, sz = random.uniform(0.8, 3.0), random.uniform(0.8, 3.0), random.uniform(1.0, 4.0)
        x, y = random.uniform(-12, 12), random.uniform(-12, 12)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, sz / 2))
        cube = bpy.context.active_object
        cube.name = f"Building_{i}"
        cube.scale = (sx, sy, sz)
        add_material(cube, (random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)))
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 7, 1.25))
    wall = bpy.context.active_object
    wall.name = "Wall"
    wall.scale = (8, 0.3, 2.5)
    add_material(wall, (0.7, 0.7, 0.6))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=150, location=(0, 0, 0))
    sky = bpy.context.active_object
    sky.name = "SkyDome"
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 30))
    bpy.context.active_object.data.energy = 3.0
    bpy.context.active_object.rotation_euler = (math.radians(50), 0, math.radians(30))
    bpy.ops.object.light_add(type="POINT", location=(3, -3, 4))
    bpy.context.active_object.data.energy = 500
    bpy.ops.object.camera_add(location=(15, -15, 8))
    bpy.context.active_object.name = "PreviewCam"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.55, 0.7, 0.95, 1.0)
        bg.inputs[1].default_value = 0.6
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return path


# ----------------------------------------------------------------------------- pipeline
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--res", default="192x108")
    p.add_argument("--frames", default="few", choices=["all", "few"])
    p.add_argument("--max_frame", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(script_args())

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    random.seed(a.seed)
    report = {}

    os.makedirs(os.path.join(out, "assets"), exist_ok=True)
    asset_path = build_character_asset(os.path.join(out, "assets", "character.blend"))
    env_path = build_environment(os.path.join(out, "env.blend"), seed=a.seed)
    report["asset"] = asset_path
    report["env"] = env_path

    # 1. bbox tool (asset is imported into the env scene temporarily and removed again)
    bbox_dir = os.path.join(out, "object_data", "collected_bbox_info")
    bbox_path = export_animation_bboxes(asset_path, "Wave", bbox_dir)
    with open(bbox_path) as f:
        bbox = json.load(f)
    report["bbox_frames"] = len(bbox["frames"])
    report["bbox_frame0"] = bbox["frames"][0]
    object_file = os.path.join(out, "object_data", "collected_object.json")
    with open(object_file, "w") as f:
        json.dump([[{"asset_path": asset_path, "action_name": "Wave",
                     "bbox_name": bbox_name_for(asset_path, "Wave")}]], f, indent=2)

    # 2. scene analysis
    scene_info_path = os.path.join(out, "logs", "scene_info.json")
    scene_info = run_scene_setup({"output_path": scene_info_path,
                                  "status_path": os.path.join(out, "logs", "status.txt"),
                                  "cell_count": 8, "exclude_num": 10, "z_percent": 0.9})
    report["scene_bounds"] = [scene_info["world_bound_min"], scene_info["world_bound_max"]]
    report["cell_size"] = scene_info["cell_size"]
    report["excluded_sky"] = scene_info["excluded_sky_objects"]

    # 3. construct
    scene = bpy.context.scene
    bvh = SceneBVH(scene)
    anno_base = os.path.join(out, "render_output")
    cfg = {"MAX_FRAME": a.max_frame, "NUM_CAMERAS": a.cameras, "MAX_ATTEMPT_NAV": 40,
           "MAX_ATTEMPT_INITIAL_GROUND_POINT": 200, "MAX_ATTEMPT_TRAJECTORY": 200}
    specs = construct_scene(bvh, scene_info, load_object_data(object_file), bbox_dir, anno_base,
                            expect_sequence_num=1, scene_type="outdoor", config=cfg)
    if not specs:
        raise SystemExit("construct_scene produced no sequence")
    with open(specs[0]) as f:
        spec = json.load(f)
    report["spec"] = specs[0]
    report["num_cameras"] = len(spec["fov"])
    report["camera_checks"] = spec["check_camera_result"]
    report["object_scale"] = spec["objects"][0]["scale"]
    report["trajectory_start"] = spec["objects"][0]["trajectory"][0]

    # 4. assemble + render
    seq_dir = os.path.dirname(specs[0])
    w, h = (int(v) for v in a.res.lower().split("x"))
    clear_dynamic(scene)
    built = build_sequence(scene, spec)
    vl = configure_render(scene, w, h, engine=a.engine, samples=a.samples, device="GPU", denoise=False)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "assembled.blend"), copy=True)
    frames = None if a.frames == "all" else [-4, -1, 0, a.max_frame // 2, a.max_frame - 1]
    meta = render_sequence(scene, built, seq_dir, camera_ids=range(a.cameras), frames=frames, view_layer=vl)
    report["rendered_frames"] = meta["frames"]

    # 5. validation
    exr0 = os.path.join(seq_dir, "images", "cam_00", "0000.exr")
    report["exr_exists"] = os.path.exists(exr0)
    report["exr_channels"] = exr_channel_names(exr0) if report["exr_exists"] else None
    fn, actors = read_vertex_frame(os.path.join(seq_dir, "vertex_data", "frame_0000.bin"))
    faces = read_faces(os.path.join(seq_dir, "faces.bin"))
    name = next(iter(actors))
    verts0 = np.asarray(actors[name])
    f_last = max(f for f in meta["frames"] if f >= 0)
    _, actors_last = read_vertex_frame(os.path.join(seq_dir, "vertex_data", f"frame_{f_last:04d}.bin"))
    verts_last = np.asarray(actors_last[name])
    fidx = np.asarray(faces[name])
    report["vertex_actor"] = name
    report["vertex_count"] = int(len(verts0))
    report["face_count"] = int(len(fidx))
    report["faces_in_range"] = bool(fidx.max() < len(verts0)) if len(fidx) else False
    report["vertex_count_constant"] = bool(len(verts0) == len(verts_last))
    report["vertex_bbox_frame0"] = [verts0.min(axis=0).round(3).tolist(), verts0.max(axis=0).round(3).tolist()]
    report["spec_global_bbox_frame0"] = spec["objects"][0]["global_bbox"][0]
    report["mean_vertex_motion"] = float(np.linalg.norm(verts_last - verts0, axis=1).mean())
    print("SMOKE_REPORT " + json.dumps(report, default=str))


if __name__ == "__main__":
    main()
