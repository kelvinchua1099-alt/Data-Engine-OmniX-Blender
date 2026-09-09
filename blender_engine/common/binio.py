"""Binary vertex / face files.

Byte-compatible with the UE VertexTracker plugin so that the original
`compute_foreground_motion.py` / `check_multicam_world.py` readers keep working.

vertex file (one per frame):
    header : <IIII  magic("VTXD"=0x56545844), version, total_frames(=1), reserved
    frame  : <IIQ   frame_number, actor_count, data_size(bytes of all actor blocks)
    actor  : <II    name_length, vertex_count ; name(utf-8) ; vertex_count * <ddd (float64 x,y,z)

faces file (one per sequence):
    header : <IIII  magic("FACE"=0x45434146), version, actor_count, reserved
    actor  : <II    name_length, section_count ; name(utf-8)
    section: <II    section_index, face_count ; face_count * <III (uint32 i0,i1,i2)
"""
import os
import struct

VTXD_MAGIC = 0x56545844
FACE_MAGIC = 0x45434146
VERSION = 1


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def write_vertex_frame(path, frame_number, actors):
    """actors: list of (name:str, vertices: iterable of (x, y, z))."""
    body = bytearray()
    for name, verts in actors:
        name_b = name.encode("utf-8")
        flat = []
        for v in verts:
            flat.extend((float(v[0]), float(v[1]), float(v[2])))
        n = len(flat) // 3
        body += struct.pack("<II", len(name_b), n)
        body += name_b
        body += struct.pack(f"<{len(flat)}d", *flat)
    header = struct.pack("<IIII", VTXD_MAGIC, VERSION, 1, 0)
    frame_header = struct.pack("<IIQ", int(frame_number), len(actors), len(body))
    _ensure_dir(path)
    with open(path, "wb") as f:
        f.write(header)
        f.write(frame_header)
        f.write(bytes(body))


def write_faces(path, actors):
    """actors: list of (name:str, faces: iterable of (i0, i1, i2)). One section per actor."""
    out = bytearray()
    out += struct.pack("<IIII", FACE_MAGIC, VERSION, len(actors), 0)
    for name, faces in actors:
        name_b = name.encode("utf-8")
        faces = [tuple(int(i) for i in f) for f in faces]
        out += struct.pack("<II", len(name_b), 1)
        out += name_b
        out += struct.pack("<II", 0, len(faces))
        flat = [i for f in faces for i in f]
        if flat:
            out += struct.pack(f"<{len(flat)}I", *flat)
    _ensure_dir(path)
    with open(path, "wb") as f:
        f.write(bytes(out))


def read_vertex_frame(path):
    """Returns (frame_number, {actor_name: [(x,y,z), ...]})."""
    with open(path, "rb") as f:
        magic, version, total_frames, _ = struct.unpack("<IIII", f.read(16))
        if magic != VTXD_MAGIC:
            raise ValueError(f"bad VTXD magic {hex(magic)} in {path}")
        frame_number, actor_count, data_size = struct.unpack("<IIQ", f.read(16))
        actors = {}
        for _ in range(actor_count):
            name_len, n = struct.unpack("<II", f.read(8))
            name = f.read(name_len).decode("utf-8")
            data = struct.unpack(f"<{3 * n}d", f.read(24 * n))
            actors[name] = [tuple(data[3 * i:3 * i + 3]) for i in range(n)]
    return frame_number, actors


def read_faces(path):
    """Returns {actor_name: [(i0,i1,i2), ...]}."""
    with open(path, "rb") as f:
        magic, version, actor_count, _ = struct.unpack("<IIII", f.read(16))
        if magic != FACE_MAGIC:
            raise ValueError(f"bad FACE magic {hex(magic)} in {path}")
        actors = {}
        for _ in range(actor_count):
            name_len, section_count = struct.unpack("<II", f.read(8))
            name = f.read(name_len).decode("utf-8")
            faces = []
            for _ in range(section_count):
                _, face_count = struct.unpack("<II", f.read(8))
                data = struct.unpack(f"<{3 * face_count}I", f.read(12 * face_count))
                faces.extend(tuple(data[3 * i:3 * i + 3]) for i in range(face_count))
            actors[name] = faces
    return actors
