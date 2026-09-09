"""Dynamic asset import (replaces UE skeletal-mesh / anim-sequence assets).

Supported: .blend (all objects + actions appended), .glb/.gltf, .fbx.
Every imported object is parented under a new Empty root placed at the world origin and tagged with
the custom property ``omnix_dynamic`` so SceneBVH ignores it.
"""
import os
import re

import bpy

from .common.log import log, warn
from .scene_query import DYNAMIC_PROP

DYNAMIC_COLLECTION = "OmniX_Dynamic"


class ImportedAsset:
    def __init__(self, name, path, root, objects, actions):
        self.name = name
        self.path = path
        self.root = root
        self.objects = objects
        self.actions = actions

    @property
    def meshes(self):
        return [o for o in self.objects if o.type == "MESH"]

    @property
    def armatures(self):
        return [o for o in self.objects if o.type == "ARMATURE"]

    @property
    def animated_targets(self):
        """Objects that should receive the action: armatures, else objects with existing animation."""
        arms = self.armatures
        if arms:
            return arms
        return [o for o in self.objects if o.animation_data is not None] or self.objects[:1]


def ensure_collection(scene, name=DYNAMIC_COLLECTION):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in [c.name for c in scene.collection.children]:
        scene.collection.children.link(coll)
    return coll


def _move_to_collection(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def import_asset(path, name=None, scene=None):
    scene = scene or bpy.context.scene
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    stem = os.path.splitext(os.path.basename(path))[0]
    name = name or stem

    objs_before = set(bpy.data.objects)
    acts_before = set(bpy.data.actions)

    if ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
            dst.actions = list(src.actions)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"unsupported asset type: {ext}")

    new_objs = [o for o in bpy.data.objects if o not in objs_before]
    new_acts = [a for a in bpy.data.actions if a not in acts_before]
    if not new_objs:
        raise RuntimeError(f"no objects imported from {path}")

    coll = ensure_collection(scene)
    for o in new_objs:
        _move_to_collection(o, coll)

    root = bpy.data.objects.new(f"{name}_root", None)
    root.empty_display_type = "PLAIN_AXES"
    coll.objects.link(root)
    for o in new_objs:
        if o.parent is None or o.parent not in new_objs:
            o.parent = root
    for o in new_objs + [root]:
        o[DYNAMIC_PROP] = True
    log(f"imported asset {name}: {len(new_objs)} objects, {len(new_acts)} actions "
        f"({[a.name for a in new_acts][:6]})")
    return ImportedAsset(name, path, root, new_objs, new_acts)


def base_name(name):
    """'Run.002' -> 'Run' (Blender appends .NNN to duplicated datablock names)."""
    return re.sub(r"\.\d{3}$", "", name)


def find_action(asset, action_name=None):
    if not asset.actions and action_name is None:
        raise RuntimeError(f"asset {asset.name} has no animation actions")
    if action_name is None:
        return asset.actions[0]
    for a in asset.actions:
        if a.name == action_name:
            return a
    for a in asset.actions:
        if base_name(a.name) == base_name(action_name):
            return a
    low = action_name.lower()
    for a in asset.actions:
        if low in a.name.lower():
            return a
    a = bpy.data.actions.get(action_name)
    if a is not None:
        return a
    raise KeyError(f"action '{action_name}' not found; available: {[a.name for a in asset.actions]}")


def _assign_slot(adt_or_strip, action):
    """Blender >= 4.4 slotted actions: pick the first slot when none is assigned."""
    if hasattr(adt_or_strip, "action_slot") and getattr(adt_or_strip, "action_slot", None) is None:
        slots = getattr(action, "slots", None)
        if slots:
            try:
                adt_or_strip.action_slot = slots[0]
            except Exception as e:  # noqa: BLE001
                warn(f"could not assign action slot: {e}")


def assign_action(asset, action):
    """Directly assign the action (used by bbox_tool; no retiming)."""
    for obj in asset.animated_targets:
        adt = obj.animation_data or obj.animation_data_create()
        for t in list(adt.nla_tracks):
            adt.nla_tracks.remove(t)
        adt.action = action
        _assign_slot(adt, action)


def set_nla_strip(obj, action, scene_start, action_start, action_end, scale):
    """Play action frames [action_start, action_end] starting at scene frame `scene_start`,
    time-stretched by `scale` (UE play_rate = 1/scale). Returns the strip."""
    adt = obj.animation_data or obj.animation_data_create()
    adt.action = None
    for t in list(adt.nla_tracks):
        adt.nla_tracks.remove(t)
    track = adt.nla_tracks.new()
    track.name = "omnix"
    strip = track.strips.new(action.name, int(scene_start), action)
    _assign_slot(strip, action)
    strip.blend_type = "REPLACE"
    strip.extrapolation = "HOLD"
    strip.use_auto_blend = False
    strip.frame_start = float(scene_start)
    strip.action_frame_start = float(action_start)
    strip.action_frame_end = float(max(action_end, action_start + 1e-3))
    strip.scale = float(scale)
    strip.frame_start = float(scene_start)
    return strip


def mark_dynamic(asset, pass_index=1):
    for o in asset.meshes:
        o.pass_index = pass_index
        o.hide_render = False
    return [o.name for o in asset.meshes]


def set_asset_visibility(asset, visible):
    for o in asset.objects:
        o.hide_render = not visible
        o.hide_viewport = not visible


def remove_asset(asset):
    """Remove the imported objects and the actions that came with them (so a re-import of the same
    file does not create 'Run.001'-style duplicates)."""
    for o in list(asset.objects) + [asset.root]:
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:  # noqa: BLE001
            pass
    for a in list(asset.actions):
        try:
            a.use_fake_user = False
            if a.users == 0:
                bpy.data.actions.remove(a)
        except Exception:  # noqa: BLE001
            pass
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images):
        for datablock in list(block):
            if datablock.users == 0 and not datablock.use_fake_user:
                try:
                    block.remove(datablock)
                except Exception:  # noqa: BLE001
                    pass
