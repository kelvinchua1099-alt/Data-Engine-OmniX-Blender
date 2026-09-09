"""Static-scene spatial queries: replaces UE line traces, box traces and the Recast navmesh.

A single BVH is built from every static mesh object in the scene (world space, evaluated).
Dynamic objects carry the custom property ``omnix_dynamic`` and are excluded automatically,
as are sky domes / backgrounds (by name keyword or by being absurdly large and roughly cubic).
"""
import math
import random

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .common.geom import Box
from .common.log import log, warn

SKY_KEYWORDS = ("sky", "dome", "background", "hdri", "atmosphere", "skybox")
DYNAMIC_PROP = "omnix_dynamic"


def world_box(obj):
    mw = obj.matrix_world
    return Box.from_points([mw @ Vector(c) for c in obj.bound_box])


def is_sky_like(obj, box, median_diag):
    name = obj.name.lower()
    if any(k in name for k in SKY_KEYWORDS):
        return True
    if median_diag > 0 and box.diagonal > 20.0 * median_diag:
        e = box.extent
        if min(e) > 0.5 * max(e):  # roughly spherical / cubic and huge -> dome
            return True
    return False


def iter_static_mesh_objects(scene):
    for obj in scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if obj.get(DYNAMIC_PROP):
            continue
        yield obj


def _box_bvh(box):
    c = box.corners()
    # corners(): index bits = (x, y, z) -> 0..7 with x major
    faces = [
        (0, 1, 3, 2), (4, 6, 7, 5),  # -x, +x
        (0, 4, 5, 1), (2, 3, 7, 6),  # -y, +y
        (0, 2, 6, 4), (1, 5, 7, 3),  # -z, +z
    ]
    return BVHTree.FromPolygons([tuple(v) for v in c], faces, all_triangles=False)


class SceneBVH:
    def __init__(self, scene=None, depsgraph=None, exclude_sky=True):
        self.scene = scene or bpy.context.scene
        dg = depsgraph or bpy.context.evaluated_depsgraph_get()
        objs = list(iter_static_mesh_objects(self.scene))
        boxes = [(o, world_box(o)) for o in objs]
        diags = sorted(b.diagonal for _, b in boxes)
        median = diags[len(diags) // 2] if diags else 0.0

        self.object_bounds = []   # [(name, Box)]
        self.sky_objects = []
        verts, polys = [], []
        for o, b in boxes:
            if exclude_sky and is_sky_like(o, b, median):
                self.sky_objects.append(o.name)
                continue
            self.object_bounds.append((o.name, b))
            ev = o.evaluated_get(dg)
            me = ev.to_mesh()
            if me is None:
                continue
            mw = ev.matrix_world
            base = len(verts)
            verts.extend((mw @ v.co).to_tuple() for v in me.vertices)
            polys.extend(tuple(base + i for i in p.vertices) for p in me.polygons)
            ev.to_mesh_clear()

        self.n_verts, self.n_polys = len(verts), len(polys)
        self.bvh = BVHTree.FromPolygons(verts, polys, all_triangles=False) if polys else None
        if self.object_bounds:
            self.bounds = Box.from_points([c for _, b in self.object_bounds for c in (b.min, b.max)])
        else:
            self.bounds = Box((-1, -1, -1), (1, 1, 1))
        log(f"SceneBVH: {len(self.object_bounds)} static objects, {self.n_polys} polygons, "
            f"skipped sky-like: {self.sky_objects}")

    # ------------------------------------------------------------------ primitives
    def ray_cast(self, origin, direction, distance):
        """Returns (location, normal, poly_index, distance) or None."""
        if self.bvh is None:
            return None
        d = Vector(direction)
        if d.length < 1e-12:
            return None
        loc, nrm, idx, dist = self.bvh.ray_cast(Vector(origin), d.normalized(), float(distance))
        if loc is None:
            return None
        return loc, nrm, idx, dist

    def trace_down(self, point, distance, lift=0.0):
        """Port of UE trace_down: positive distance traces downward, negative traces upward.
        Returns the hit location or None. `lift` raises the ray origin by that many meters so a
        `point` lying on a surface neither self-hits (upward) nor starts below it (downward)."""
        direction = Vector((0, 0, -1)) if distance > 0 else Vector((0, 0, 1))
        origin = Vector(point) + Vector((0, 0, lift))
        length = abs(distance) + (lift if distance > 0 else 0.0)
        hit = self.ray_cast(origin, direction, length)
        return hit[0] if hit else None

    def project_to_ground(self, point, up=1.0, down=10.0, max_slope_deg=60.0):
        """Nearest walkable surface below `point` (replaces navmesh projection)."""
        hit = self.ray_cast(Vector(point) + Vector((0, 0, up)), (0, 0, -1), up + down)
        if hit is None:
            return None
        loc, nrm, _, _ = hit
        if nrm.z < math.cos(math.radians(max_slope_deg)):
            return None
        return loc

    def surface_hits_below(self, x, y, max_hits=8, max_slope_deg=45.0):
        """All upward-facing surfaces along the vertical line (x, y), top to bottom."""
        top = self.bounds.max.z + 1.0
        bottom = self.bounds.min.z - 1.0
        hits = []
        origin = Vector((x, y, top))
        remaining = top - bottom
        cos_slope = math.cos(math.radians(max_slope_deg))
        while remaining > 0 and len(hits) < max_hits:
            hit = self.ray_cast(origin, (0, 0, -1), remaining)
            if hit is None:
                break
            loc, nrm, _, dist = hit
            if nrm.z >= cos_slope:
                hits.append(loc)
            origin = loc + Vector((0, 0, -1e-3))
            remaining -= dist + 1e-3
        return hits

    def random_navigable_point(self, center, radius, max_tries=32, pick="first"):
        """Replaces NavigationSystem.get_random_point_in_navigable_radius.
        pick='first' -> top-most surface (outdoor), 'random' -> any surface (allows indoor floors)."""
        for _ in range(max_tries):
            theta = random.uniform(0, 2 * math.pi)
            r = radius * math.sqrt(random.uniform(0, 1))
            x, y = center[0] + r * math.cos(theta), center[1] + r * math.sin(theta)
            hits = self.surface_hits_below(x, y)
            if not hits:
                continue
            return hits[0] if pick == "first" else random.choice(hits)
        return None

    def box_overlaps(self, box):
        """True if the axis-aligned box intersects any static geometry (UE box_trace with start==end)."""
        if self.bvh is None:
            return False
        return bool(self.bvh.overlap(_box_bvh(box)))

    def segment_hit(self, start, end, half_extent=0.0):
        """Approximation of UE box_trace_single along a segment: centre ray plus four offset rays."""
        s, e = Vector(start), Vector(end)
        d = e - s
        length = d.length
        if length < 1e-9:
            if half_extent > 0:
                ext = Vector((half_extent,) * 3)
                return self.box_overlaps(Box(s - ext, s + ext))
            return False
        dn = d / length
        offsets = [Vector((0, 0, 0))]
        if half_extent > 0:
            u = dn.cross(Vector((0, 0, 1)))
            if u.length < 1e-6:
                u = Vector((1, 0, 0))
            u.normalize()
            v = dn.cross(u).normalized()
            offsets += [u * half_extent, -u * half_extent, v * half_extent, -v * half_extent]
        for off in offsets:
            if self.ray_cast(s + off, dn, length) is not None:
                return True
        return False
