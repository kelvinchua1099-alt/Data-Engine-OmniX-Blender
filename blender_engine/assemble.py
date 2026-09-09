"""Rebuild a sequence spec (written by construct_scene) inside Blender.

Replaces the UE Level Sequence: dynamic objects are imported, scaled, given per-frame location
keyframes on their root Empty and an NLA strip that retimes the chosen action; one camera object per
spec camera receives per-frame location / rotation keyframes.

Blender frame b  <->  data frame f = b - warmup_frames.  Frames f < 0 are the static warm-up frames.
"""
import math
import os

import bpy
from mathutils import Vector

from .assets import import_asset, find_action, set_nla_strip, mark_dynamic, set_asset_visibility, remove_asset
from .common.convert import blender_camera_matrix
from .common.log import log, warn
from .scene_query import DYNAMIC_PROP

CAMERA_COLLECTION = "OmniX_Cameras"


class BuiltSequence:
    def __init__(self, spec):
        self.spec = spec
        self.assets = []
        self.cameras = []
        self.warmup = int(spec.get("warmup_frames", 4))
        self.max_frame = int(spec["max_frame"])

    @property
    def dynamic_meshes(self):
        return [m for a in self.assets for m in a.meshes]

    def frame_to_blender(self, f):
        return int(f) + self.warmup

    def set_dynamic_visible(self, visible):
        for a in self.assets:
            set_asset_visibility(a, visible)


def _collection(scene, name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in [c.name for c in scene.collection.children]:
        scene.collection.children.link(coll)
    return coll


class _LinearKeyframes:
    """Context manager: newly inserted keyframes use LINEAR interpolation."""

    def __enter__(self):
        self.prev = bpy.context.preferences.edit.keyframe_new_interpolation_type
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
        return self

    def __exit__(self, *exc):
        bpy.context.preferences.edit.keyframe_new_interpolation_type = self.prev
        return False


def clear_dynamic(scene):
    """Remove everything a previous build_sequence created (objects, cameras, orphan data)."""
    for obj in list(scene.objects):
        if obj.get(DYNAMIC_PROP) or obj.name.startswith("CusCamera_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.cameras, bpy.data.actions):
        for datablock in list(block):
            if datablock.users == 0 and not datablock.use_fake_user:
                try:
                    block.remove(datablock)
                except Exception:  # noqa: BLE001
                    pass


def build_objects(scene, built, spec_objects):
    for idx, ospec in enumerate(spec_objects):
        asset = import_asset(ospec["asset_path"], name=ospec["name"], scene=scene)
        action = find_action(asset, ospec.get("action_name"))
        mark_dynamic(asset, pass_index=int(ospec.get("pass_index", 1)))

        root = asset.root
        s = float(ospec["scale"])
        root.scale = (s, s, s)
        c0 = Vector(ospec["bbox_center0"])
        for f, p in enumerate(ospec["trajectory"]):
            root.location = Vector(p) - c0
            root.keyframe_insert("location", frame=built.frame_to_blender(f))

        act_start = float(ospec.get("action_frame_start", action.frame_range[0]))
        start = act_start + float(ospec["start_frame"])
        end = start + float(ospec["frame_number"])
        scale = built.max_frame / float(ospec["frame_number"])
        for target in asset.animated_targets:
            set_nla_strip(target, action, built.warmup, start, end, scale)
        built.assets.append(asset)
        log(f"object {idx} '{asset.name}': scale {s:.3f}, action {action.name} [{start:.0f},{end:.0f}] x{scale:.3f}")


def build_cameras(scene, built, spec, use_dof=False, clip_end=1000.0):
    coll = _collection(scene, CAMERA_COLLECTION)
    fovs = spec["fov"]
    positions = spec["camera_position"]
    rotations = spec["rotation"]
    look_ats = spec.get("camera_look_at")
    for ci, (fov, pos_list, rot_list) in enumerate(zip(fovs, positions, rotations)):
        name = f"CusCamera_{ci:02d}"
        cam_data = bpy.data.cameras.new(name)
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.angle = math.radians(float(fov))
        cam_data.clip_start = 0.01
        cam_data.clip_end = clip_end
        cam_data.dof.use_dof = bool(use_dof)
        cam = bpy.data.objects.new(name, cam_data)
        coll.objects.link(cam)
        cam.rotation_mode = "QUATERNION"
        for f, (p, r) in enumerate(zip(pos_list, rot_list)):
            b = built.frame_to_blender(f)
            m = blender_camera_matrix(p, r)
            cam.location = m.translation
            cam.rotation_quaternion = m.to_quaternion()
            cam.keyframe_insert("location", frame=b)
            cam.keyframe_insert("rotation_quaternion", frame=b)
            if use_dof and look_ats:
                cam_data.dof.focus_distance = (Vector(look_ats[ci][f]) - Vector(p)).length
                cam_data.dof.keyframe_insert("focus_distance", frame=b)
        built.cameras.append(cam)
    log(f"created {len(built.cameras)} cameras")


def build_sequence(scene, spec, use_dof=False):
    built = BuiltSequence(spec)
    scene.render.fps = int(spec.get("fps", 24))
    scene.frame_start = 0
    scene.frame_end = built.warmup + built.max_frame - 1
    with _LinearKeyframes():
        build_objects(scene, built, spec["objects"])
        build_cameras(scene, built, spec, use_dof=use_dof)
    scene.frame_set(built.warmup)
    return built


def teardown_sequence(scene, built):
    for a in built.assets:
        remove_asset(a)
    for cam in built.cameras:
        data = cam.data
        bpy.data.objects.remove(cam, do_unlink=True)
        if data.users == 0:
            bpy.data.cameras.remove(data)
