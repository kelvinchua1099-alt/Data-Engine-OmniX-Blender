"""Static environment analysis (port of the SceneSetup C++ plugin + parse_scene.py helpers).

Produces scene_info.json with the same keys the UE version wrote:
    world_bound_min / world_bound_max : {"x","y","z"}   (meters)
    cell_size, density_map, ppv_mask, reflection_mask, camera_mask, light_mask
Post-process volumes and reflection captures do not exist in Blender, those masks are all zeros.
The Recast navmesh is replaced by ray casts against SceneBVH (see sample_nav_region / generate_initial_point).
"""
import json
import math
import os
import random

import bpy
from mathutils import Vector

from .common.geom import Box
from .common.log import log, warn
from .scene_query import SceneBVH

EXCLUDE_NAME_KEYWORDS = {"fog", "light", "sky", "reflection", "volume", "post", "atmospheric",
                         "weather", "wind", "environment", "settings"}


def is_excluded_name(name):
    n = name.lower()
    return any(k in n for k in EXCLUDE_NAME_KEYWORDS)


# ----------------------------------------------------------------------------- bounds
def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _std(v, m):
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0


def compute_scene_bounds(boxes, exclude_num=10, z_percent=0.9):
    """boxes: list[Box]. Port of FSceneSetupTest::CalculateSceneBounds."""
    items = sorted(boxes, key=lambda b: -b.diagonal)
    if exclude_num > 0 and items:
        one_percent = max(1, len(items) // 100)
        n_remove = min(exclude_num, one_percent)
        log(f"bounds: {len(items)} objects, removing {n_remove} largest")
        items = items[n_remove:]
    if not items:
        return None, []

    cols = {
        "min_x": [b.min.x for b in items], "max_x": [b.max.x for b in items],
        "min_y": [b.min.y for b in items], "max_y": [b.max.y for b in items],
        "min_z": [b.min.z for b in items], "cz": [b.center.z for b in items],
    }
    stats = {k: (_mean(v), _std(v, _mean(v))) for k, v in cols.items()}

    def is_outlier(b):
        checks = [("min_x", b.min.x), ("max_x", b.max.x), ("min_y", b.min.y),
                  ("max_y", b.max.y), ("min_z", b.min.z), ("cz", b.center.z)]
        for k, val in checks:
            m, s = stats[k]
            if s > 1e-9 and abs(val - m) > 3.0 * s:
                return True
        return False

    kept = [b for b in items if not is_outlier(b)]
    if not kept:
        return None, []

    min_x = min(b.min.x for b in kept)
    max_x = max(b.max.x for b in kept)
    min_y = min(b.min.y for b in kept)
    max_y = max(b.max.y for b in kept)
    min_z = min(b.min.z for b in kept)
    czs = sorted(b.center.z for b in kept)
    idx = max(0, min(len(czs) - 1, int(math.floor(len(czs) * z_percent))))
    max_z = czs[idx]
    if max_z <= min_z:
        max_z = max(b.max.z for b in kept)
    bounds = Box((min_x, min_y, min_z), (max_x, max_y, max_z))
    if bounds.size.length < 1e-6:
        return None, kept
    return bounds, kept


def compute_density_map(scene_bounds, boxes, cell_count=16):
    """Port of GenerateDensityMap: log(1 + number of object boxes covering each cell)."""
    ext = scene_bounds.extent
    cell_size = 2.0 * min(ext.x, ext.y) / cell_count
    if cell_size <= 0:
        cell_size = max(ext.x, ext.y, 1e-3) * 2.0 / cell_count
    width = max(1, int(math.floor((scene_bounds.max.x - scene_bounds.min.x) / cell_size)))
    height = max(1, int(math.floor((scene_bounds.max.y - scene_bounds.min.y) / cell_size)))
    density = [[0.0] * width for _ in range(height)]
    bmin = scene_bounds.min
    for b in boxes:
        if scene_bounds.overlap(b) is None:
            continue
        gx0 = max(0, int(math.floor((b.min.x - bmin.x) / cell_size)))
        gx1 = min(width - 1, int(math.floor((b.max.x - bmin.x) / cell_size)))
        gy0 = max(0, int(math.floor((b.min.y - bmin.y) / cell_size)))
        gy1 = min(height - 1, int(math.floor((b.max.y - bmin.y) / cell_size)))
        for y in range(gy0, gy1 + 1):
            for x in range(gx0, gx1 + 1):
                density[y][x] += 1.0
    for y in range(height):
        density[y] = [math.log(v + 1.0) for v in density[y]]
    return cell_size, density


def refine_bounds_by_density(scene_bounds, density, cell_size, max_extent):
    """Port of RefineBoundsByDensity (only used when max_extent > 0)."""
    ext = scene_bounds.extent
    if max_extent <= 0 or (ext.x <= max_extent and ext.y <= max_extent):
        return scene_bounds
    cells = [(density[y][x], x, y) for y in range(len(density)) for x in range(len(density[0])) if density[y][x] > 0]
    if not cells:
        c = scene_bounds.center
        target = Box(c - Vector((max_extent,) * 3), c + Vector((max_extent,) * 3))
        return scene_bounds.overlap(target) or scene_bounds
    cells.sort(key=lambda t: -t[0])
    top = max(1, int(math.ceil(len(cells) * 0.3)))
    _, cx_i, cy_i = random.choice(cells[:top])
    cx = scene_bounds.min.x + (cx_i + 0.5) * cell_size
    cy = scene_bounds.min.y + (cy_i + 0.5) * cell_size
    nx0, nx1 = cx - max_extent, cx + max_extent
    ny0, ny1 = cy - max_extent, cy + max_extent
    if nx1 > scene_bounds.max.x:
        d = nx1 - scene_bounds.max.x
        nx0, nx1 = nx0 - d, nx1 - d
    if nx0 < scene_bounds.min.x:
        d = scene_bounds.min.x - nx0
        nx0, nx1 = nx0 + d, nx1 + d
    if ny1 > scene_bounds.max.y:
        d = ny1 - scene_bounds.max.y
        ny0, ny1 = ny0 - d, ny1 - d
    if ny0 < scene_bounds.min.y:
        d = scene_bounds.min.y - ny0
        ny0, ny1 = ny0 + d, ny1 + d
    refined = Box((nx0, ny0, scene_bounds.min.z), (nx1, ny1, scene_bounds.max.z))
    return scene_bounds.overlap(refined) or scene_bounds


def compute_masks(scene, scene_bounds, cell_size, width, height):
    """camera / light masks from existing camera and light objects; PPV / reflection are zeros."""
    zeros = lambda: [[0.0] * width for _ in range(height)]
    ppv, refl, cam, light = zeros(), zeros(), zeros(), zeros()

    def mark(mask, pos, radius_cells=0):
        gx = int(math.floor((pos.x - scene_bounds.min.x) / cell_size))
        gy = int(math.floor((pos.y - scene_bounds.min.y) / cell_size))
        for y in range(gy - radius_cells, gy + radius_cells + 1):
            for x in range(gx - radius_cells, gx + radius_cells + 1):
                if 0 <= x < width and 0 <= y < height:
                    mask[y][x] = 1.0

    for obj in scene.objects:
        if obj.type == "CAMERA":
            mark(cam, obj.matrix_world.translation)
        elif obj.type == "LIGHT":
            if obj.data.type == "SUN":
                continue  # unbound, like an unbound PPV it would be ignored anyway
            radius = 0
            if getattr(obj.data, "use_custom_distance", False):
                radius = int(obj.data.cutoff_distance / cell_size)
            mark(light, obj.matrix_world.translation, min(radius, 3))
    return ppv, refl, cam, light


def analyse_scene(scene=None, cell_count=16, exclude_num=10, z_percent=0.9, max_nav_extent=0.0):
    scene = scene or bpy.context.scene
    bvh = SceneBVH(scene)
    boxes = [b for name, b in bvh.object_bounds if not is_excluded_name(name)]
    if not boxes:
        raise RuntimeError("no static mesh objects found in the scene")
    bounds, kept = compute_scene_bounds(boxes, exclude_num=exclude_num, z_percent=z_percent)
    if bounds is None:
        raise RuntimeError("invalid scene bounds")
    cell_size, density = compute_density_map(bounds, kept, cell_count)
    bounds = refine_bounds_by_density(bounds, density, cell_size, max_nav_extent)
    if max_nav_extent > 0:
        cell_size, density = compute_density_map(bounds, kept, cell_count)
    height, width = len(density), len(density[0])
    ppv, refl, cam, light = compute_masks(scene, bounds, cell_size, width, height)
    info = {
        "units": "meters",
        "blend_file": bpy.data.filepath,
        "world_bound_min": {"x": bounds.min.x, "y": bounds.min.y, "z": bounds.min.z},
        "world_bound_max": {"x": bounds.max.x, "y": bounds.max.y, "z": bounds.max.z},
        "cell_size": cell_size,
        "cell_count": cell_count,
        "density_map": density,
        "ppv_mask": ppv,
        "reflection_mask": refl,
        "camera_mask": cam,
        "light_mask": light,
        "static_object_count": len(bvh.object_bounds),
        "excluded_sky_objects": bvh.sky_objects,
    }
    log(f"scene bounds {bounds}, cell_size {cell_size:.3f}, grid {width}x{height}")
    return info, bvh


def run_scene_setup(config, scene=None):
    """config keys: output_path, status_path, cell_count, exclude_num, z_percent, max_nav_extent."""
    status_path = config.get("status_path")
    try:
        info, _ = analyse_scene(scene, cell_count=int(config.get("cell_count", 16)),
                                exclude_num=int(config.get("exclude_num", 10)),
                                z_percent=float(config.get("z_percent", 0.9)),
                                max_nav_extent=float(config.get("max_nav_extent", 0.0)))
    except Exception as e:  # noqa: BLE001
        if status_path:
            os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
            with open(status_path, "w") as f:
                f.write(f"FAILED: {e}\n")
        raise
    out = config["output_path"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    if status_path:
        os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
        with open(status_path, "w") as f:
            f.write("SUCCESS\n")
    log(f"scene_info written to {out}")
    return info


# ----------------------------------------------------------------------------- runtime helpers (parse_scene.py)
def update_density_map(density_map, masks, threshold_percent=75.0, multiply_factor=3.0):
    if not masks or not density_map:
        return density_map
    height, width = len(density_map), len(density_map[0])
    merged = [[0] * width for _ in range(height)]
    for idx, mask in enumerate(masks):
        ones = sum(1 for row in mask for v in row if v == 1)
        total = height * width
        if total and (ones / total) * 100.0 > threshold_percent:
            log(f"mask {idx} ignored: {(ones / total) * 100:.1f}% cells set")
            continue
        for y in range(height):
            for x in range(width):
                if mask[y][x] == 1:
                    merged[y][x] = 1
    for y in range(height):
        for x in range(width):
            if merged[y][x] == 1:
                density_map[y][x] *= multiply_factor
    return density_map


def sample_from_density_map(density_map, bound_min, bound_max, cell_size, density_power=2.4):
    weights = [(x, y, d ** density_power) for y, row in enumerate(density_map) for x, d in enumerate(row)]
    total = sum(w for _, _, w in weights)
    if total <= 0:
        x, y, _ = random.choice(weights)
    else:
        r = random.random() * total
        acc = 0.0
        x, y = weights[-1][0], weights[-1][1]
        for cx, cy, w in weights:
            acc += w
            if r <= acc:
                x, y = cx, cy
                break
    return Vector((bound_min.x + (x + 0.5) * cell_size, bound_min.y + (y + 0.5) * cell_size,
                   (bound_min.z + bound_max.z) * 0.5))


def sample_nav_region(bvh, world_bound_min, world_bound_max, density_map, cell_size, max_tries=200):
    """Pick a density-weighted cell that actually has walkable ground. Returns (location, radius)."""
    for _ in range(max_tries):
        nav_location = sample_from_density_map(density_map, world_bound_min, world_bound_max, cell_size)
        nav_radius = cell_size * 0.5
        if bvh.random_navigable_point(nav_location, nav_radius) is not None:
            return nav_location, nav_radius
    raise RuntimeError("could not find any navigable cell in the density map")


def generate_initial_point(bvh, nav_location, nav_search_radius=200.0, search_radius=1.0, height_offset=1.0,
                           max_attempts=100, num_rays=16, inclination_angle=15.0, ground_tilt_angle=10.0,
                           indoor_prob=0.1, require_surroundings=False, ground_check_slack=3.0):
    """Port of parse_scene.generate_initial_point with the navmesh replaced by ray casts.
    Returns a ground point (Vector) or None.

    Note: in the UE original the 'ground' and 'surroundings' ray checks were effectively no-ops
    (the HitResult struct is never None). Here the ground check is active but lenient: the slanted
    rays may travel `ground_check_slack` times the nominal length before they must hit, so flat
    roofs are accepted while real cliff edges are rejected. The surroundings check is opt-in."""
    ray_length1 = search_radius / math.cos(math.radians(inclination_angle))
    ray_length2 = math.hypot(height_offset + search_radius * math.tan(math.radians(ground_tilt_angle)), search_radius)
    ray_length3 = search_radius * random.randint(7, 11)
    inclination_angle2 = -math.degrees(math.atan(
        (height_offset + search_radius * math.tan(math.radians(ground_tilt_angle))) / search_radius))
    indoor = random.uniform(0, 1) < indoor_prob
    hit_mid_fail = 0
    reasons = {"no_point": 0, "roof": 0, "project": 0, "blocked": 0, "ledge": 0, "surroundings": 0}
    radius = nav_search_radius

    for attempt in range(max_attempts):
        # widen the search disk gradually when the cell turns out to be hard (keeps the density bias)
        if attempt and attempt % max(1, max_attempts // 4) == 0:
            radius *= 1.5
        point = bvh.random_navigable_point(nav_location, radius, pick="random" if indoor else "first")
        if point is None:
            reasons["no_point"] += 1
            continue
        if not indoor:
            # must not be under a roof
            ok = False
            for _ in range(30):
                if bvh.trace_down(point, -search_radius * 100.0, lift=0.05) is None:
                    ok = True
                    break
                point = bvh.random_navigable_point(nav_location, radius, pick="first")
                if point is None:
                    break
            if not ok or point is None:
                reasons["roof"] += 1
                continue
        else:
            ok = False
            for _ in range(200):
                if bvh.trace_down(point, -search_radius * 100.0, lift=0.05) is not None:
                    ok = True
                    break
                point = bvh.random_navigable_point(nav_location, radius, pick="random")
                if point is None:
                    break
            if not ok or point is None:
                reasons["roof"] += 1
                break

        ground = bvh.project_to_ground(point, up=0.05, down=10.0)
        if ground is None:
            reasons["project"] += 1
            continue
        ray_start = Vector((ground.x, ground.y, ground.z + height_offset))

        # 1. slightly-upward rays must be free (nothing around the object at its height)
        valid = True
        for i in range(num_rays):
            ang = 2 * math.pi * i / num_rays
            d = Vector((math.cos(ang) * math.cos(math.radians(inclination_angle)),
                        math.sin(ang) * math.cos(math.radians(inclination_angle)),
                        math.sin(math.radians(inclination_angle))))
            if bvh.ray_cast(ray_start, d, ray_length1) is not None:
                valid = False
                reasons["blocked"] += 1
                break
        # 2. downward-slanted rays must hit ground (not standing on a ledge)
        if valid:
            for i in range(num_rays):
                ang = 2 * math.pi * i / num_rays
                d = Vector((math.cos(ang) * math.cos(math.radians(inclination_angle2)),
                            math.sin(ang) * math.cos(math.radians(inclination_angle2)),
                            math.sin(math.radians(inclination_angle2))))
                if bvh.ray_cast(ray_start, d, ray_length2 * ground_check_slack) is None:
                    valid = False
                    reasons["ledge"] += 1
                    break
        # 3. optionally require some structure around (45 deg upward rays hit something)
        if valid and require_surroundings:
            any_hit = False
            for i in range(num_rays):
                ang = 2 * math.pi * i / num_rays
                d = Vector((math.cos(ang), math.sin(ang), 1.0))
                if bvh.ray_cast(ray_start, d, ray_length3 * math.sqrt(2)) is not None:
                    any_hit = True
                    break
            if not any_hit:
                valid = False
                hit_mid_fail += 1
                reasons["surroundings"] += 1
                if hit_mid_fail > 10:
                    break
        if valid:
            log(f"initial ground point {tuple(round(v, 3) for v in ground)} after {attempt + 1} attempt(s)")
            return ground
    warn(f"could not find an appropriate initial point (search_radius {search_radius:.2f}, "
         f"rejections: {reasons})")
    return None
