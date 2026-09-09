"""Camera / coordinate conventions.

World: right-handed, Z-up, meters (Blender).
Camera orientation is stored as [roll, pitch, yaw] in degrees, same semantic as the UE version:
    yaw   : rotation about world +Z  (atan2(forward.y, forward.x))
    pitch : elevation of the forward vector (asin(forward.z))
    roll  : rotation about the forward axis
Blender camera object looks along local -Z with local +Y up.
OpenCV camera looks along +Z with +Y down; this is what the offline tools consume (c2w).
"""
import math

try:
    from mathutils import Vector, Matrix
except ImportError:  # offline tools never call the Matrix helpers, keep import optional
    Vector = Matrix = None


def look_at_rpy(position, target):
    d = Vector(target) - Vector(position)
    length = d.length
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    yaw = math.degrees(math.atan2(d.y, d.x))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, d.z / length))))
    return [0.0, pitch, yaw]


def rpy_to_basis(roll, pitch, yaw):
    """Returns (forward, right, up) unit vectors in world space."""
    p = math.radians(pitch)
    y = math.radians(yaw)
    forward = Vector((math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)))
    right = Vector((math.sin(y), -math.cos(y), 0.0))
    up = right.cross(forward)
    if abs(roll) > 1e-9:
        rot = Matrix.Rotation(math.radians(roll), 3, forward)
        right = rot @ right
        up = rot @ up
    return forward.normalized(), right.normalized(), up.normalized()


def camera_basis_from_forward(forward):
    """(forward, right, up) with zero roll for an arbitrary forward vector."""
    f = Vector(forward).normalized()
    r = f.cross(Vector((0.0, 0.0, 1.0)))
    if r.length < 1e-8:
        r = Vector((0.0, -1.0, 0.0))
    r.normalize()
    u = r.cross(f).normalized()
    return f, r, u


def blender_camera_matrix(position, rpy):
    """4x4 world matrix for a Blender camera object at `position` with orientation `rpy`."""
    f, r, u = rpy_to_basis(*rpy)
    m = Matrix.Identity(4)
    for i in range(3):
        m[i][0] = r[i]
        m[i][1] = u[i]
        m[i][2] = -f[i]
        m[i][3] = float(position[i])
    return m


def opencv_c2w(position, rpy):
    """4x4 camera-to-world (OpenCV convention: x right, y down, z forward) as nested lists."""
    f, r, u = rpy_to_basis(*rpy)
    rows = []
    for i in range(3):
        rows.append([float(r[i]), float(-u[i]), float(f[i]), float(position[i])])
    rows.append([0.0, 0.0, 0.0, 1.0])
    return rows


def intrinsics(hfov_deg, width, height):
    fx = width * 0.5 / math.tan(math.radians(hfov_deg) * 0.5)
    return [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]]
