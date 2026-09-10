"""Build a photoreal forest-clearing environment from the Poly Haven assets fetched by
assets/download_forest_assets.py, and save it as a pipeline-ready environment .blend.

    blender -b -P scripts/build_forest_env.py -- --assets assets/polyhaven --out assets/env/forest_clearing.blend \
        [--seed 1] [--size 90] [--trees 45] [--rocks 24] [--ferns 70] [--grass 2500] [--preview preview.png]

Scene: displaced ground with a tiled PBR material, HDRI sky, scanned trees in a ring around a clearing,
rocks / logs / ferns scattered with spacing constraints, grass clumps as a hair particle system.
Grass and ferns are tagged `omnix_no_collision` so they are rendered but ignored by the collision BVH.
"""
import argparse
import glob
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import script_args  # noqa: E402

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from blender_engine.common.log import log, warn  # noqa: E402
from blender_engine.scene_query import IGNORE_PROP  # noqa: E402

import re

# asset id -> (sampling weight, regex selecting the complete-model variants inside the .blend).
# Poly Haven tree files contain LOD0..LOD2 of each species plus trunk / twig parts; LOD0 is 2-6M
# triangles per tree so the mid LODs are used and every other object is deleted after selection.
TREES = {
    "fir_tree_01": (3, r"^fir_tree_01_[abc]_LOD1$"),
    "pine_tree_01": (3, r"^pine_tree_01_[abc]_LOD2$"),
    "tree_small_02": (2, r"^tree_small_02_LOD1$"),
    "fir_sapling_medium": (2, r"^fir_sapling_medium_[abc]_LOD1$"),
}
ROCKS = {
    "rock_moss_set_01": (2, r"^rock_moss_set_01_rock\d+$"),
    "rock_moss_set_02": (2, r"^rock_moss_set_02_rock\d+$"),
    "boulder_01": (1, r"^boulder_01_LOD1$"),
    "dead_tree_trunk_02": (1, r"^dead_tree_trunk_02_LOD1$"),
}
FERNS = {"fern_02": (1, r"^fern_02_[abcd]$")}
GRASS = ("grass_medium_01", r"^grass_medium_01_(large|mid)_[abc]_LOD1$")
GROUND_TEX = "forrest_ground_01"
HDRI = "meadow_2"


# ----------------------------------------------------------------------------- helpers
def collection(name, parent=None):
    coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if coll.name not in [c.name for c in parent.children]:
        parent.children.link(coll)
    return coll


def relink_images(asset_dir):
    for img in bpy.data.images:
        if img.source != "FILE" or not img.filepath:
            continue
        p = bpy.path.abspath(img.filepath)
        if os.path.exists(p):
            continue
        hits = glob.glob(os.path.join(asset_dir, "**", os.path.basename(img.filepath)), recursive=True)
        if hits:
            img.filepath = hits[0]
            img.reload()
        else:
            warn(f"texture not found: {img.filepath}")


def append_polyhaven(asset_id, pattern, assets_root, library):
    """Append the objects of the asset's .blend whose names match `pattern` into the hidden Library
    collection; everything else in the file is discarded. Returns the selected variants."""
    d = os.path.join(assets_root, asset_id)
    blends = sorted(glob.glob(os.path.join(d, "*.blend")))
    if not blends:
        raise FileNotFoundError(f"no .blend for {asset_id} in {d}")
    rx = re.compile(pattern)
    with bpy.data.libraries.load(blends[0]) as (src, _):
        wanted = [n for n in src.objects if rx.match(n)]
    if not wanted:
        raise RuntimeError(f"{asset_id}: no object matches {pattern}; file has {list(src.objects)[:10]}...")
    before = set(bpy.data.objects)
    with bpy.data.libraries.load(blends[0], link=False) as (src, dst):
        dst.objects = wanted
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        library.objects.link(o)
        o.hide_render = True
        o.hide_viewport = True
        o.parent = None
    relink_images(d)
    variants = [o for o in new if o.type == "MESH"]
    log(f"appended {asset_id}: {[o.name for o in variants]}")
    return variants


def purge_orphans():
    for _ in range(3):
        for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.node_groups, bpy.data.textures):
            for db in list(block):
                if db.users == 0 and not db.use_fake_user:
                    try:
                        block.remove(db)
                    except Exception:  # noqa: BLE001
                        pass


def instance(src, coll, location, yaw, scale, tag_ignore=False):
    ob = src.copy()
    ob.name = f"{src.name}_i"
    ob.hide_render = False
    ob.hide_viewport = False
    ob.location = location
    ob.rotation_euler = (0, 0, yaw)
    ob.scale = (scale, scale, scale)
    if tag_ignore:
        ob[IGNORE_PROP] = True
    coll.objects.link(ob)
    return ob


def setup_world(hdri_path, strength=1.0):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:  # noqa: BLE001
        pass
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path)
    mapping = nt.nodes.new("ShaderNodeMapping")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(random.uniform(0, 360)))
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def make_ground(size, maps, tiles=16):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=220, y_subdivisions=220, size=size, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    # gentle terrain: large undulation + small bumps (Displace modifiers on procedural textures)
    for name, noise_scale, strength in (("undulation", 28.0, 1.1), ("bumps", 4.0, 0.18)):
        tex = bpy.data.textures.new(f"ground_{name}", "CLOUDS")
        tex.noise_scale = noise_scale
        tex.noise_depth = 2
        mod = ground.modifiers.new(name, "DISPLACE")
        mod.texture = tex
        mod.strength = strength
        mod.mid_level = 0.5
        mod.texture_coords = "GLOBAL"
    # PBR material
    mat = bpy.data.materials.new("ForestGround")
    try:
        mat.use_nodes = True
    except Exception:  # noqa: BLE001
        pass
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (tiles, tiles, 1)
    nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])

    def tex_node(path, non_color):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        n.projection = "FLAT"
        nt.links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    if "Diffuse" in maps:
        nt.links.new(tex_node(maps["Diffuse"], False).outputs["Color"], bsdf.inputs["Base Color"])
    if "Rough" in maps:
        nt.links.new(tex_node(maps["Rough"], True).outputs["Color"], bsdf.inputs["Roughness"])
    if "nor_gl" in maps:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 0.8
        nt.links.new(tex_node(maps["nor_gl"], True).outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    ground.data.materials.append(mat)
    return ground


def add_grass(ground, grass_obj, count, seed):
    grass_obj.hide_render = False       # instance source must be renderable for particles
    grass_obj.hide_viewport = False
    grass_obj.location = (0, 0, -50)    # park the source far below the terrain
    grass_obj[IGNORE_PROP] = True
    mod = ground.modifiers.new(f"Grass_{grass_obj.name}", "PARTICLE_SYSTEM")
    ps = mod.particle_system
    ps.seed = seed
    st = ps.settings
    st.type = "HAIR"
    st.count = count
    st.hair_length = 1.0
    st.render_type = "OBJECT"
    st.instance_object = grass_obj
    st.particle_size = 1.0
    st.size_random = 0.45
    st.use_advanced_hair = True
    st.use_rotations = True
    st.rotation_mode = "NOR"
    st.phase_factor_random = 2.0
    st.emit_from = "FACE"
    st.distribution = "RAND"
    st.use_modifier_stack = True
    return ps


def ground_z(scene, dg, x, y):
    hit, loc, *_ = scene.ray_cast(dg, Vector((x, y, 60.0)), Vector((0, 0, -1)))
    return loc.z if hit else 0.0


def sample_positions(n, r_min, r_max, spacing, occupied, rng, tries=4000):
    out = []
    for _ in range(tries):
        if len(out) >= n:
            break
        ang = rng.uniform(0, 2 * math.pi)
        r = math.sqrt(rng.uniform(r_min ** 2, r_max ** 2))
        x, y = r * math.cos(ang), r * math.sin(ang)
        if all((x - ox) ** 2 + (y - oy) ** 2 >= (spacing + osp) ** 2 for ox, oy, osp in occupied):
            out.append((x, y))
            occupied.append((x, y, spacing))
    return out


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "assets", "polyhaven"))
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--size", type=float, default=90.0)
    p.add_argument("--clearing", type=float, default=7.0)
    p.add_argument("--trees", type=int, default=90)
    p.add_argument("--rocks", type=int, default=28)
    p.add_argument("--ferns", type=int, default=90)
    p.add_argument("--grass", type=int, default=3000)
    p.add_argument("--preview", default=None)
    p.add_argument("--preview_samples", type=int, default=64)
    a = p.parse_args(script_args())
    rng = random.Random(a.seed)
    random.seed(a.seed)

    bpy.ops.wm.read_homefile(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.unit_settings.system = "METRIC"

    hdri = glob.glob(os.path.join(a.assets, HDRI, "*.hdr")) + glob.glob(os.path.join(a.assets, HDRI, "*.exr"))
    setup_world(hdri[0], strength=1.0)

    maps = json.load(open(os.path.join(a.assets, GROUND_TEX, "maps.json")))
    ground = make_ground(a.size, maps, tiles=int(a.size / 5.5))
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()

    library = collection("Library")
    library.hide_render = True
    forest = collection("Forest")
    rocks_coll = collection("Rocks")
    plants = collection("Plants")

    variants = {}
    specs = dict(TREES)
    specs.update(ROCKS)
    specs.update(FERNS)
    specs[GRASS[0]] = (1, GRASS[1])
    for asset_id, (_, pattern) in specs.items():
        try:
            variants[asset_id] = append_polyhaven(asset_id, pattern, a.assets, library)
        except Exception as e:  # noqa: BLE001
            warn(f"skipping {asset_id}: {e}")
    purge_orphans()

    def pick(weights):
        ids = [i for i in weights if variants.get(i)]
        w = [weights[i][0] for i in ids]
        asset_id = rng.choices(ids, weights=w)[0]
        return rng.choice(variants[asset_id])

    occupied = []
    r_out = a.size * 0.42
    for (x, y) in sample_positions(a.trees, a.clearing + 1.5, r_out, 2.8, occupied, rng):
        src = pick(TREES)
        instance(src, forest, (x, y, ground_z(scene, dg, x, y) - 0.05), rng.uniform(0, 2 * math.pi),
                 rng.uniform(0.85, 1.25))
    for (x, y) in sample_positions(a.rocks, 1.5, r_out, 1.2, occupied, rng):
        src = pick(ROCKS)
        instance(src, rocks_coll, (x, y, ground_z(scene, dg, x, y) - 0.03), rng.uniform(0, 2 * math.pi),
                 rng.uniform(0.7, 1.4))
    for (x, y) in sample_positions(a.ferns, 1.0, r_out, 0.4, list(occupied), rng):
        src = pick(FERNS)
        instance(src, plants, (x, y, ground_z(scene, dg, x, y) - 0.02), rng.uniform(0, 2 * math.pi),
                 rng.uniform(0.7, 1.3), tag_ignore=True)
    if variants.get(GRASS[0]) and a.grass > 0:
        # a few particle systems, one per grass variant, so the clumps vary
        gvars = variants[GRASS[0]]
        for i, gv in enumerate(gvars):
            add_grass(ground, gv, max(1, a.grass // len(gvars)), a.seed + i)

    # a preview camera at eye height looking across the clearing
    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_data.lens = 28
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0.0, -a.clearing * 1.3, ground_z(scene, dg, 0.0, -a.clearing * 1.3) + 1.6)
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = (Vector((0, 0, 1.0)) - cam.location).to_track_quat("-Z", "Y")
    scene.camera = cam

    scene.cycles.samples = a.preview_samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in
                                                            bpy.types.ColorManagedViewSettings.bl_rna.properties[
                                                                "view_transform"].enum_items] else "Filmic"

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    bpy.ops.file.make_paths_absolute()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    n_static = len([o for o in scene.objects if o.type == "MESH" and not o.hide_render])
    log(f"saved {a.out}: {n_static} renderable mesh objects "
        f"(trees {len(forest.objects)}, rocks {len(rocks_coll.objects)}, ferns {len(plants.objects)}, grass {a.grass})")

    if a.preview:
        from blender_engine.render import enable_gpu
        if enable_gpu():
            scene.cycles.device = "GPU"
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = os.path.abspath(a.preview)
        bpy.ops.render.render(write_still=True)
        log(f"preview rendered to {a.preview}")


if __name__ == "__main__":
    main()
