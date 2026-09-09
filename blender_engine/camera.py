"""Camera placement (port of camera_util.py). Meters, Z-up, right-handed.

Orientation is [roll, pitch, yaw] degrees, see common/convert.py. Occlusion checks go through SceneBVH.
"""
import math
import random

from mathutils import Vector

from .common.convert import look_at_rpy, camera_basis_from_forward
from .common.log import log

MIN_CAMERA_DISTANCE = 0.3   # meters (UE: 30 cm)

CAMERA_POS_LOOKAT_COMBINATION = [
    ("static", "static"),
    ("static", "follow_obj"),
    ("follow_obj", "follow_obj"),
    ("sphere_azi", "static"),
    ("sphere_elv", "static"),
    ("sphere_azi", "follow_obj"),
    ("sphere_elv", "follow_obj"),
    ("translation_lr", "static"),
    ("translation_ud", "static"),
    ("translation_fb", "static"),
    ("translation_lr", "follow_obj"),
    ("translation_ud", "follow_obj"),
    ("translation_fb", "follow_obj"),
    ("translation_lr", "follow_position"),
    ("translation_ud", "follow_position"),
    ("translation_fb", "follow_position"),
]


# ----------------------------------------------------------------------------- basic geometry
def calculate_bboxes_center(bboxes):
    c = Vector((0, 0, 0))
    for b in bboxes:
        c += b.center
    return c / max(len(bboxes), 1)


def calculate_normal_vector(camera_direction, center, point):
    d = Vector(camera_direction).normalized()
    rel = Vector(point) - Vector(center)
    projection_point = Vector(center) + d * rel.dot(d)
    return -(projection_point - Vector(point))


def calculate_rotation(camera_position, center):
    return look_at_rpy(camera_position, center)


def calculate_length_ratio(angle_with_up, image_ratio=(16, 9)):
    assert 0 <= angle_with_up <= 180, "angle_with_up range not correct"
    if angle_with_up > 90:
        angle_with_up = 180 - angle_with_up
    a = math.radians(angle_with_up)
    width_ratio, height_ratio = image_ratio
    if abs(math.tan(a)) <= (width_ratio / height_ratio):
        return (height_ratio / width_ratio) * (1 / max(math.cos(a), 1e-9))
    return 1 / max(math.sin(a), 1e-9)


def calculate_new_fov(old_fov, k):
    return math.degrees(2 * math.atan(math.tan(math.radians(old_fov) / 2) * k))


def calculate_angle(v1, v2):
    denom = v1.length * v2.length
    if denom < 1e-12:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, v1.dot(v2) / denom))))


def generate_camera_direction(elevation_angle, azimuth_angle):
    """Unit vector from the object centre towards the camera."""
    el, az = math.radians(elevation_angle), math.radians(azimuth_angle)
    return Vector((math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)))


def generate_camera_up(camera_direction):
    _, _, up = camera_basis_from_forward(-Vector(camera_direction))
    return up


def calculate_min_camera_distance(fov, center, camera_direction, bboxes, image_ratio=(16, 9)):
    camera_up = generate_camera_up(camera_direction)
    min_distance = 0.0
    for bbox in bboxes:
        for corner in bbox.corners():
            normal_vector = calculate_normal_vector(camera_direction, center, corner)
            length_ratio = calculate_length_ratio(calculate_angle(normal_vector, camera_up), image_ratio)
            new_fov = calculate_new_fov(fov, length_ratio)
            corner_ray = corner - Vector(center)
            corner_camera_angle = calculate_angle(corner_ray, Vector(camera_direction))
            target_angle = 180.0 - corner_camera_angle - new_fov * 0.5
            if target_angle <= 0:
                continue
            target_length = (math.sin(math.radians(target_angle)) /
                             math.sin(math.radians(new_fov * 0.5)) * corner_ray.length)
            min_distance = max(target_length, min_distance)
    return min_distance


# ----------------------------------------------------------------------------- path generation (tuple math)
def normalize(v):
    mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rotate_around_axis(vector, axis, angle_rad):
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    k = normalize(axis)
    dot_kv = k[0] * vector[0] + k[1] * vector[1] + k[2] * vector[2]
    ckv = cross(k, vector)
    return (
        vector[0] * cos_a + ckv[0] * sin_a + k[0] * dot_kv * (1 - cos_a),
        vector[1] * cos_a + ckv[1] * sin_a + k[1] * dot_kv * (1 - cos_a),
        vector[2] * cos_a + ckv[2] * sin_a + k[2] * dot_kv * (1 - cos_a),
    )


def generate_line_path(center, camera_start, length, num_points=20, mode="lr", roll=0.0, pitch=0.0, yaw=0.0):
    cx, cy, cz = center
    sx, sy, sz = camera_start
    cam_vec = (cx - sx, cy - sy, cz - sz)
    cam_vec_n = normalize(cam_vec)
    roll_rad, pitch_rad, yaw_rad = math.radians(roll), math.radians(pitch), math.radians(yaw)
    denom = max(num_points - 1, 1)

    if mode == "lr":
        proj_xy = normalize((cam_vec_n[0], cam_vec_n[1], 0.0))
        dir_vec = cross((0, 0, 1), proj_xy)
        dir_vec = normalize((dir_vec[0], dir_vec[1], 0.0))
        if yaw != 0:
            dir_vec = rotate_around_axis(dir_vec, (0, 0, 1), yaw_rad)
        if roll != 0:
            dir_vec = rotate_around_axis(dir_vec, cam_vec_n, roll_rad)
        half_len = length / 2.0
        ts = [(-half_len + (length * i / denom)) for i in range(num_points)]
        return [(sx + dir_vec[0] * t, sy + dir_vec[1] * t, sz + dir_vec[2] * t) for t in ts]

    if mode == "ud":
        proj_xy = normalize((cam_vec_n[0], cam_vec_n[1], 0.0))
        tangent_xy = cross(proj_xy, (0, 0, 1))
        dir_vec = normalize(cross(cam_vec_n, tangent_xy))
        if pitch != 0:
            horizontal_axis = cross(cam_vec_n, (0, 0, 1))
            if horizontal_axis == (0, 0, 0):
                horizontal_axis = (1, 0, 0)
            dir_vec = rotate_around_axis(dir_vec, normalize(horizontal_axis), pitch_rad)
        if roll != 0:
            dir_vec = rotate_around_axis(dir_vec, cam_vec_n, roll_rad)
        half_len = length / 2.0
        ts = [(-half_len + (length * i / denom)) for i in range(num_points)]
        return [(sx + dir_vec[0] * t, sy + dir_vec[1] * t, sz + dir_vec[2] * t) for t in ts]

    if mode == "fb":
        dir_vec = normalize((-cam_vec[0], -cam_vec[1], -cam_vec[2]))
        if pitch != 0:
            horizontal_axis = cross(cam_vec_n, (0, 0, 1))
            if horizontal_axis == (0, 0, 0):
                horizontal_axis = (1, 0, 0)
            dir_vec = rotate_around_axis(dir_vec, normalize(horizontal_axis), pitch_rad)
        if yaw != 0:
            dir_vec = rotate_around_axis(dir_vec, (0, 0, 1), yaw_rad)
        ts = [(length * i / denom) for i in range(num_points)]
        return [(sx + dir_vec[0] * (length - t), sy + dir_vec[1] * (length - t), sz + dir_vec[2] * (length - t))
                for t in ts]

    raise ValueError("mode must be 'lr', 'ud', or 'fb'")


def generate_ellipse_path(center, camera_position, k=1.5, theta_range=90, orientation="azi",
                          num_points=100, pitch=0.0, yaw=0.0, roll=0.0):
    cx, cy, cz = center
    px, py, pz = camera_position
    vx, vy, vz = px - cx, py - cy, pz - cz
    dist_total = math.sqrt(vx * vx + vy * vy + vz * vz)
    dist_horizontal = math.sqrt(vx * vx + vy * vy)
    elev_angle = math.atan2(vz, dist_horizontal) if dist_horizontal != 0 else 0.0
    denom = max(num_points - 1, 1)

    offset = math.pi / 2
    theta_half = math.radians(theta_range / 2.0)
    theta_vals = [-theta_half + (2 * theta_half) * i / denom for i in range(num_points)]

    if orientation == "azi":
        b = dist_total * math.cos(elev_angle)
        a = b * k
        ellipse_2d = [(a * math.cos(t + offset), b * math.sin(t + offset)) for t in theta_vals]
        cam_vec = normalize((vx, vy, 0.0))
        orig_short_axis = (0.0, 1.0)
        dot_val = orig_short_axis[0] * cam_vec[0] + orig_short_axis[1] * cam_vec[1]
        cross_z = orig_short_axis[0] * cam_vec[1] - orig_short_axis[1] * cam_vec[0]
        angle_rad = math.atan2(cross_z, dot_val)
        ellipse_rot = [(x * math.cos(angle_rad) - y * math.sin(angle_rad),
                        x * math.sin(angle_rad) + y * math.cos(angle_rad)) for x, y in ellipse_2d]
        points = [(x + cx, y + cy, pz) for x, y in ellipse_rot]

    elif orientation == "elv":
        b = dist_total
        a = b * k
        pitch_y = math.atan2(vx, vz)
        yaw_z = math.atan2(vy, vx)
        cos_z, sin_z = math.cos(-yaw_z), math.sin(-yaw_z)
        cam_after_z = (vx * cos_z - vy * sin_z, vx * sin_z + vy * cos_z, vz)
        cos_y, sin_y = math.cos(-pitch_y), math.sin(-pitch_y)
        cam_in_xz = (cam_after_z[0] * cos_y + cam_after_z[2] * sin_y, cam_after_z[1],
                     -cam_after_z[0] * sin_y + cam_after_z[2] * cos_y)
        t_cam = math.atan2(cam_in_xz[2] / b, cam_in_xz[0] / a)
        theta_vals = [t_cam - theta_half + (2 * theta_half) * i / denom for i in range(num_points)]
        ellipse_pts = [(a * math.cos(t), 0.0, b * math.sin(t)) for t in theta_vals]
        rotated = [rotate_around_axis(p, (0, 1, 0), pitch_y) for p in ellipse_pts]
        rotated = [rotate_around_axis(p, (0, 0, 1), yaw_z) for p in rotated]
        points = [(x + cx, y + cy, z + cz) for x, y, z in rotated]
    else:
        raise ValueError("orientation must be 'azi' or 'elv'")

    yaw_axis = (0, 0, 1)
    h_mag = math.sqrt(vx * vx + vy * vy)
    pitch_axis = normalize((-vy, vx, 0)) if h_mag > 1e-7 else (1, 0, 0)
    roll_axis = normalize((vx, vy, vz))
    yaw_rad, pitch_rad, roll_rad = math.radians(yaw), math.radians(pitch), math.radians(roll)

    out = []
    for p in points:
        rel = (p[0] - px, p[1] - py, p[2] - pz)
        if yaw != 0:
            rel = rotate_around_axis(rel, yaw_axis, yaw_rad)
        if pitch != 0:
            rel = rotate_around_axis(rel, pitch_axis, pitch_rad)
        if roll != 0:
            rel = rotate_around_axis(rel, roll_axis, roll_rad)
        out.append((rel[0] + px, rel[1] + py, rel[2] + pz))
    return out


# ----------------------------------------------------------------------------- noise
def smooth_camera_positions_simple(camera_position_list, window_size=6):
    if len(camera_position_list) <= window_size:
        return camera_position_list
    half = window_size // 2
    out = []
    n = len(camera_position_list)
    for i in range(n):
        s, e = max(0, i - half), min(n, i + half + 1)
        out.append([sum(camera_position_list[j][k] for j in range(s, e)) / (e - s) for k in range(3)])
    return out


def generate_smooth_noise_sequence(length, amplitude=1.0):
    interval = max(1, int(length * random.uniform(0.2, 0.3)))
    values = []
    current = random.uniform(-amplitude, amplitude)
    nxt = random.uniform(-amplitude, amplitude)
    for frame in range(length):
        if frame % interval == 0:
            current = nxt
            nxt = random.uniform(-amplitude, amplitude)
        alpha = (frame % interval) / float(interval)
        values.append((1 - alpha) * current + alpha * nxt)
    return values


def generate_high_frequency_noise(length, amplitude=0.1):
    return [random.uniform(-amplitude, amplitude) for _ in range(length)]


def add_position_noise(camera_location_list, delta_distance):
    n = len(camera_location_list)
    lf = [generate_smooth_noise_sequence(n, delta_distance), generate_smooth_noise_sequence(n, delta_distance),
          generate_smooth_noise_sequence(n, delta_distance * 0.1)]
    hf = [generate_high_frequency_noise(n, delta_distance * 0.1), generate_high_frequency_noise(n, delta_distance * 0.1),
          generate_high_frequency_noise(n, delta_distance * 0.01)]
    return [[loc[k] + lf[k][i] + hf[k][i] for k in range(3)] for i, loc in enumerate(camera_location_list)]


def add_rotation_noise(rotation_list, rot_amp=5.0):
    n = len(rotation_list)
    lf = [generate_smooth_noise_sequence(n, rot_amp * 0.1), generate_smooth_noise_sequence(n, rot_amp),
          generate_smooth_noise_sequence(n, rot_amp)]
    hf = [generate_high_frequency_noise(n, rot_amp * 0.01), generate_high_frequency_noise(n, rot_amp * 0.1),
          generate_high_frequency_noise(n, rot_amp * 0.1)]
    return [[rot[k] + lf[k][i] + hf[k][i] for k in range(3)] for i, rot in enumerate(rotation_list)]


def get_static_list(value, length):
    return [list(value) for _ in range(length)]


# ----------------------------------------------------------------------------- validity checks
def create_frustum_traces(camera_location, look_at_point, horizontal_fov, aspect_ratio=16.0 / 9.0):
    direction = Vector(look_at_point) - Vector(camera_location)
    distance = direction.length
    check_distance = distance * 0.3
    forward = direction / distance
    right = forward.cross(Vector((0, 0, 1)))
    if right.length < 1e-8:
        right = Vector((0, 1, 0))
    right.normalize()
    up = right.cross(forward).normalized()
    near_width = 2.0 * math.tan(math.radians(horizontal_fov * 0.5)) * check_distance
    near_height = near_width / aspect_ratio
    half_h, half_w = near_height * 0.5, near_width * 0.5
    center_point = Vector(camera_location) + forward * check_distance
    points = []
    for i in range(4):
        for j in range(4):
            h_factor = (i / 3 * 2.0) - 1.0
            v_factor = (j / 3 * 2.0) - 1.0
            points.append(center_point + right * (half_w * h_factor) + up * (half_h * v_factor))
    return points


def check_camera_single(bvh, global_bboxes_centers, fov, camera_position_list, image_ratio=(16, 9),
                        camera_extent_scale=0.05):
    distances = [(Vector(p) - c).length for p, c in zip(camera_position_list, global_bboxes_centers)]
    if min(distances) < MIN_CAMERA_DISTANCE:
        return False
    total_frames = len(global_bboxes_centers)
    check_frames = max(int(total_frames * 0.25), 1)
    interval = max(total_frames // check_frames, 1)
    aspect = image_ratio[0] / image_ratio[1]
    for i in range(0, total_frames, interval):
        obj_center = global_bboxes_centers[i]
        cam = Vector(camera_position_list[i])
        extent = camera_extent_scale * (obj_center - cam).length
        if bvh.segment_hit(cam, obj_center, extent):
            return False
        for point in create_frustum_traces(cam, obj_center, fov, aspect):
            if bvh.segment_hit(cam, point, extent):
                return False
    return True


# ----------------------------------------------------------------------------- placement
def place_camera_single(global_bboxes, camera_fov_range, camera_elevation_range,
                        camera_distance_scale_range=(1.0, 1.0), camera_position_mode="static",
                        camera_look_at_mode="static", camera_traj_sphere_angle_range=(45, 90),
                        camera_traj_translation_scale_range=(0.25, 0.5), camera_traj_plane_tile_range=(5, 10),
                        noise_scale_range=(0.01, 0.02), noise_angle_range=(0, 5), image_ratio=(16, 9)):
    if camera_look_at_mode == "follow_obj":
        bboxes = [obj[0] for obj in global_bboxes]
    else:
        bboxes = [b for obj in global_bboxes for b in obj]
    initial_obj_center = calculate_bboxes_center(bboxes)

    fov = random.uniform(*camera_fov_range)
    elevation_angle = random.uniform(*camera_elevation_range)
    azimuth_angle = random.uniform(0, 360)
    distance_scale = random.uniform(*camera_distance_scale_range)

    camera_direction = generate_camera_direction(elevation_angle, azimuth_angle)
    min_dist = calculate_min_camera_distance(fov, initial_obj_center, camera_direction, bboxes, image_ratio)
    camera_distance = min_dist * distance_scale

    total_frames = len(global_bboxes[0])
    center = [initial_obj_center.x, initial_obj_center.y, initial_obj_center.z]
    initial_camera_position = [
        center[0] + camera_distance * math.cos(math.radians(azimuth_angle)) * math.cos(math.radians(elevation_angle)),
        center[1] + camera_distance * math.sin(math.radians(azimuth_angle)) * math.cos(math.radians(elevation_angle)),
        center[2] + camera_distance * math.sin(math.radians(elevation_angle)),
    ]

    def frame_centers():
        return [calculate_bboxes_center([obj[i] for obj in global_bboxes]) for i in range(total_frames)]

    if camera_position_mode == "static":
        positions = get_static_list(initial_camera_position, total_frames)
    elif camera_position_mode == "follow_obj":
        oc = frame_centers()
        positions = [[initial_camera_position[k] + oc[i][k] - oc[0][k] for k in range(3)] for i in range(total_frames)]
        positions = smooth_camera_positions_simple(positions)
    elif camera_position_mode.startswith("sphere_"):
        angle = random.uniform(*camera_traj_sphere_angle_range)
        if camera_position_mode.endswith("elv"):
            angle *= 0.125
        tilt = random.uniform(*camera_traj_plane_tile_range)
        positions = [list(p) for p in generate_ellipse_path(
            center, initial_camera_position, k=random.uniform(1.0, 1.5), theta_range=angle,
            orientation=camera_position_mode.split("_")[1], num_points=total_frames,
            pitch=tilt, yaw=tilt, roll=tilt)]
    elif camera_position_mode.startswith("translation_"):
        length = random.uniform(*camera_traj_translation_scale_range) * camera_distance
        tilt = random.uniform(*camera_traj_plane_tile_range)
        positions = [list(p) for p in generate_line_path(
            center, initial_camera_position, length=length, num_points=total_frames,
            mode=camera_position_mode.split("_")[1], pitch=tilt, yaw=tilt, roll=tilt)]
    else:
        raise ValueError(f"Invalid camera_position_mode: {camera_position_mode}")

    if camera_look_at_mode == "static":
        look_ats = get_static_list(center, total_frames)
    elif camera_look_at_mode == "follow_obj":
        look_ats = smooth_camera_positions_simple([[c.x, c.y, c.z] for c in frame_centers()])
    elif camera_look_at_mode == "follow_position":
        half = calculate_bboxes_center([obj[total_frames // 2] for obj in global_bboxes])
        look_ats = [[half[k] + positions[i][k] - positions[total_frames // 2][k] for k in range(3)]
                    for i in range(total_frames)]
    else:
        raise ValueError(f"Invalid camera_look_at_mode: {camera_look_at_mode}")

    if random.random() < 0.5 and camera_position_mode != "follow_obj":
        positions = positions[::-1]
        if camera_look_at_mode == "follow_position":
            look_ats = look_ats[::-1]

    rotations = [calculate_rotation(positions[i], look_ats[i]) for i in range(total_frames)]

    pos_noise = random.uniform(*noise_scale_range) * camera_distance
    rot_noise = random.uniform(*noise_angle_range)
    positions = add_position_noise(positions, pos_noise)
    rotations = add_rotation_noise(rotations, rot_noise)
    return fov, positions, rotations, look_ats


def place_camera(bvh, global_bboxes, camera_fov_range=(60, 100), camera_elevation_range=(0, 30),
                 camera_distance_scale_range=(1.0, 1.0), camera_traj_sphere_angle_range=(45, 90),
                 camera_traj_translation_scale_range=(0.25, 0.5), camera_traj_plane_tile_range=(5, 10),
                 noise_scale_range=(0.01, 0.02), noise_angle_range=(0, 5), image_ratio=(16, 9),
                 camera_extent_scale=0.1, max_attempts_per_camera=17, min_valid_ratio=0.9,
                 combinations=None):
    """Place one camera per (position_mode, look_at_mode) combination (16 by default).
    Returns (fov_all, position_all, rotation_all, look_at_all, check_all) or None."""
    combinations = combinations or CAMERA_POS_LOOKAT_COMBINATION
    fov_all, pos_all, rot_all, look_all, check_all = [], [], [], [], []
    total_frames = len(global_bboxes[0])
    centers = [calculate_bboxes_center([obj[i] for obj in global_bboxes]) for i in range(total_frames)]

    for pos_mode, look_mode in combinations:
        attempt = 0
        ok = False
        while attempt < max_attempts_per_camera:
            fov, positions, rotations, look_ats = place_camera_single(
                global_bboxes, camera_fov_range, camera_elevation_range, camera_distance_scale_range,
                pos_mode, look_mode, camera_traj_sphere_angle_range, camera_traj_translation_scale_range,
                camera_traj_plane_tile_range, noise_scale_range, noise_angle_range, image_ratio)
            ok = check_camera_single(bvh, centers, fov, positions, image_ratio, camera_extent_scale)
            if ok:
                break
            attempt += 1
        fov_all.append(fov)
        pos_all.append(positions)
        rot_all.append(rotations)
        look_all.append(look_ats)
        check_all.append(bool(ok))

    if sum(check_all) < len(check_all) * min_valid_ratio:
        log(f"camera placement rejected: {sum(check_all)}/{len(check_all)} cameras valid")
        return None
    return fov_all, pos_all, rot_all, look_all, check_all
