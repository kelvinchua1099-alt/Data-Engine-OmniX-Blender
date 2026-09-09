"""Bounding-box helpers (port of bbox_util.py / math_util.py, unreal.Vector -> mathutils.Vector).

A "bbox" in list form is [[min_x, min_y, min_z], [max_x, max_y, max_z]] exactly like the UE code;
`Box` is the object form (replaces unreal.Box).
"""
import copy
import math
import random

from mathutils import Vector


class Box:
    __slots__ = ("min", "max")

    def __init__(self, bmin, bmax):
        self.min = Vector(bmin)
        self.max = Vector(bmax)

    @property
    def center(self):
        return (self.min + self.max) * 0.5

    @property
    def extent(self):
        return (self.max - self.min) * 0.5

    @property
    def size(self):
        return self.max - self.min

    @property
    def diagonal(self):
        return (self.max - self.min).length

    def corners(self):
        mn, mx = self.min, self.max
        return [
            Vector((mn.x, mn.y, mn.z)), Vector((mn.x, mn.y, mx.z)),
            Vector((mn.x, mx.y, mn.z)), Vector((mn.x, mx.y, mx.z)),
            Vector((mx.x, mn.y, mn.z)), Vector((mx.x, mn.y, mx.z)),
            Vector((mx.x, mx.y, mn.z)), Vector((mx.x, mx.y, mx.z)),
        ]

    def to_list(self):
        return [[self.min.x, self.min.y, self.min.z], [self.max.x, self.max.y, self.max.z]]

    @staticmethod
    def from_points(points):
        pts = list(points)
        mn = Vector((min(p[0] for p in pts), min(p[1] for p in pts), min(p[2] for p in pts)))
        mx = Vector((max(p[0] for p in pts), max(p[1] for p in pts), max(p[2] for p in pts)))
        return Box(mn, mx)

    def intersects(self, other, tolerance=1e-4):
        return boxes_intersect(self, other, tolerance)

    def overlap(self, other):
        mn = Vector((max(self.min.x, other.min.x), max(self.min.y, other.min.y), max(self.min.z, other.min.z)))
        mx = Vector((min(self.max.x, other.max.x), min(self.max.y, other.max.y), min(self.max.z, other.max.z)))
        if mn.x > mx.x or mn.y > mx.y or mn.z > mx.z:
            return None
        return Box(mn, mx)

    def __repr__(self):
        return f"Box(min={tuple(round(v, 3) for v in self.min)}, max={tuple(round(v, 3) for v in self.max)})"


def lerp(a, b, t):
    return a + (b - a) * t


def interpolate_bounding_boxes(bbox_list, frame_number_list):
    """Stretch every object's bbox trajectory to the longest one (linear interpolation)."""
    max_frames = max(frame_number_list)
    out = []
    for bbox_trajectory, frame_count in zip(bbox_list, frame_number_list):
        if frame_count == max_frames:
            out.append(bbox_trajectory)
            continue
        interpolated = []
        for i in range(max_frames):
            t = i / (max_frames - 1) if max_frames > 1 else 0.0
            original_index = t * (frame_count - 1)
            lo = int(original_index)
            hi = min(lo + 1, frame_count - 1)
            frac = original_index - lo
            lo_b, hi_b = bbox_trajectory[lo], bbox_trajectory[hi]
            interpolated.append([
                [lerp(lo_b[0][j], hi_b[0][j], frac) for j in range(3)],
                [lerp(lo_b[1][j], hi_b[1][j], frac) for j in range(3)],
            ])
        out.append(interpolated)
    return out


def get_bbox_size(bbox):
    return (Vector(bbox[1]) - Vector(bbox[0])).length


def get_bbox_center(bbox):
    return Vector((
        (bbox[0][0] + bbox[1][0]) * 0.5,
        (bbox[0][1] + bbox[1][1]) * 0.5,
        (bbox[0][2] + bbox[1][2]) * 0.5,
    ))


def get_bbox_height(bbox):
    return bbox[1][2] - bbox[0][2]


def get_bbox_list_radius_and_height(bbox_list):
    """Max horizontal radius / height over all objects, over their whole trajectories."""
    max_radius, max_height = 0.0, 0.0
    for obj_bbox_list in bbox_list:
        mn = list(obj_bbox_list[0][0])
        mx = list(obj_bbox_list[0][1])
        for b_min, b_max in obj_bbox_list[1:]:
            mn = [min(b_min[i], mn[i]) for i in range(3)]
            mx = [max(b_max[i], mx[i]) for i in range(3)]
        radius = math.hypot(mx[0] - mn[0], mx[1] - mn[1]) * 0.5
        max_radius = max(radius, max_radius)
        max_height = max(mx[2] - mn[2], max_height)
    return max_radius, max_height


def move_bbox_to_center(bbox, new_center):
    extent = Vector((
        (bbox[1][0] - bbox[0][0]) * 0.5,
        (bbox[1][1] - bbox[0][1]) * 0.5,
        (bbox[1][2] - bbox[0][2]) * 0.5,
    ))
    c = Vector(new_center)
    return Box(c - extent, c + extent)


def rescale_bbox_list(bbox_list, scales):
    out = copy.deepcopy(bbox_list)
    for oi, s in enumerate(scales):
        if s == 1:
            continue
        for fi in range(len(out[oi])):
            out[oi][fi][0] = [v * s for v in out[oi][fi][0]]
            out[oi][fi][1] = [v * s for v in out[oi][fi][1]]
    return out


def shrink_box(box, scale=1.0):
    """Shrink a Box horizontally (x, y) about its center."""
    center = box.center
    ext = box.extent.copy()
    ext.x *= scale
    ext.y *= scale
    return Box(center - ext, center + ext)


def boxes_intersect(box1, box2, tolerance=1e-4):
    return (
        box1.min.x <= box2.max.x + tolerance and box1.max.x >= box2.min.x - tolerance and
        box1.min.y <= box2.max.y + tolerance and box1.max.y >= box2.min.y - tolerance and
        box1.min.z <= box2.max.z + tolerance and box1.max.z >= box2.min.z - tolerance
    )


def sample_point_in_cylinder(radius, height):
    theta = random.uniform(0, 2 * math.pi)
    r = random.uniform(0, radius)
    z = random.uniform(0, height)
    return Vector((r * math.cos(theta), r * math.sin(theta), z))


def bezier_point(t, p0, p1, p2, p3):
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    return p0 * mt3 + p1 * (3 * mt2 * t) + p2 * (3 * mt * t2) + p3 * t3
