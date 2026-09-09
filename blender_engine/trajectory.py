"""Object trajectory generation (port of trajectory.py). Traces go through SceneBVH."""
import math
import random

from mathutils import Vector

from .common.geom import (Box, get_bbox_center, get_bbox_height, move_bbox_to_center, shrink_box,
                          boxes_intersect, sample_point_in_cylinder, bezier_point)
from .common.log import log

TRACE_DOWN_DISTANCE = 10.0   # meters (UE: 1000 cm)


def check_trajectory_on_the_ground(bvh, obj_trajectories, obj_bboxes, initial_ground_point_z):
    projected_points = []
    for trajectory, obj_bbox in zip(obj_trajectories, obj_bboxes):
        new_center = trajectory + (get_bbox_center(obj_bbox) - get_bbox_center(obj_bboxes[0]))
        h = get_bbox_height(obj_bbox)
        projected = bvh.project_to_ground(new_center, up=h * 0.5 + 0.5, down=max(h * 2.5, 2.0))
        if projected is None:
            return None
        # avoid sudden jump to an upper floor
        if (projected.z - initial_ground_point_z) > h * 0.5:
            down = bvh.trace_down(new_center, get_bbox_height(obj_bboxes[0]) * 10.0)
            if down is None:
                return None
            projected_points.append(Vector((down.x, down.y, down.z + 0.001)))
            continue
        projected_points.append(projected)
    return projected_points


def check_box_scene_collision(bvh, obj_global_bboxes):
    for gb in obj_global_bboxes:
        ext = gb.extent.copy()
        ext.z *= 0.95
        c = gb.center
        if bvh.box_overlaps(Box(c - ext, c + ext)):
            return True
    return False


def generate_trajectories(bvh, interpolated_bboxes, complexity=0.5, trajectory_radius=1.0,
                          trajectory_height=0.5, on_the_ground=None, initial_ground_point=None,
                          max_attempts=100, static_prob=0.5):
    """Returns (trajectories, global_bboxes) or None.
    trajectories[i][f] : world position of object i's frame-0 bbox centre at frame f.
    global_bboxes[i][f]: Box of object i at frame f (horizontally shrunk by `shrink` = 1.0)."""
    initial_ground_point = Vector(initial_ground_point or (0, 0, 0))
    if on_the_ground is None:
        on_the_ground = [True] * len(interpolated_bboxes)

    def generate_control_points(start, end):
        distance = (end - start).length
        max_offset = min(trajectory_radius, distance * 0.5) * complexity
        t1 = random.uniform(0.2, 0.4)
        t2 = random.uniform(0.6, 0.8)
        dir_vector = end - start

        def limited_offset():
            theta = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, max_offset)
            return Vector((r * math.cos(theta), r * math.sin(theta),
                           random.uniform(-trajectory_height * 0.2, trajectory_height * 0.2)))

        return start + dir_vector * t1 + limited_offset(), start + dir_vector * t2 + limited_offset()

    def simple_trajectory(start, end, frames):
        p1, p2 = generate_control_points(start, end)
        if frames <= 1:
            return [start.copy()]
        return [bezier_point(t / (frames - 1), start, p1, p2, end) for t in range(frames)]

    def find_ground_height(points):
        heights = []
        for p in points:
            g = bvh.trace_down(p, TRACE_DOWN_DISTANCE, lift=0.05)
            if g is None:
                return None
            heights.append(g.z)
        return heights

    def check_box_box_collision(bboxes1, prev_bboxes):
        for bboxes2 in prev_bboxes:
            for b1, b2 in zip(bboxes1, bboxes2):
                if boxes_intersect(b1, b2):
                    return True
        return False

    total_frames = len(interpolated_bboxes[0])
    trajectories, global_bboxes = [], []

    for obj_index, obj_bboxes in enumerate(interpolated_bboxes):
        attempt = 0
        while attempt < max_attempts:
            start = sample_point_in_cylinder(trajectory_radius, trajectory_height)
            end = sample_point_in_cylinder(trajectory_radius, trajectory_height)
            obj_traj = simple_trajectory(start, end, total_frames)
            if random.random() < static_prob:
                obj_traj = [obj_traj[0].copy() for _ in range(total_frames)]
            half_h0 = get_bbox_height(obj_bboxes[0]) * 0.5
            obj_traj = [Vector((p.x + initial_ground_point.x, p.y + initial_ground_point.y,
                                p.z + initial_ground_point.z + half_h0)) for p in obj_traj]

            if on_the_ground[obj_index]:
                projected = check_trajectory_on_the_ground(bvh, obj_traj, obj_bboxes, initial_ground_point.z)
                if projected is None:
                    attempt += 1
                    continue
                ground_heights = find_ground_height(projected)
                if ground_heights is None:
                    attempt += 1
                    continue
                c0 = get_bbox_center(obj_bboxes[0]).z
                for p, gh, bb in zip(obj_traj, ground_heights, obj_bboxes):
                    # faithful to UE: put the object's bottom on the ground each frame
                    p.z = gh
                    cf = get_bbox_center(bb).z
                    half_h = get_bbox_height(bb) * 0.5
                    if p.z + (cf - c0) - half_h < gh:
                        p.z = gh + half_h - (cf - c0)

            obj_global = []
            for p, bb in zip(obj_traj, obj_bboxes):
                new_center = p + (get_bbox_center(bb) - get_bbox_center(obj_bboxes[0]))
                obj_global.append(shrink_box(move_bbox_to_center(bb, new_center)))

            if check_box_box_collision(obj_global, global_bboxes[:obj_index]):
                attempt += 1
                continue
            if check_box_scene_collision(bvh, obj_global):
                attempt += 1
                continue

            trajectories.append(obj_traj)
            global_bboxes.append(obj_global)
            log(f"trajectory for object {obj_index} found after {attempt + 1} attempt(s)")
            break

        if len(trajectories) != obj_index + 1:
            return None

    return trajectories, global_bboxes
