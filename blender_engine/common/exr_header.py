"""Pure-python OpenEXR header parsing (no EXR library needed) + channel selection for Blender output.

Blender 5.x writes the object-index pass as "<ViewLayer>.Object Index.X" (older versions: "IndexOB.X"),
depth as "<ViewLayer>.Depth.Z" (older: "Z"), and the beauty as "<ViewLayer>.Combined.R/G/B/A".
"""
import struct


def exr_channel_names(path):
    """Channel names of a single-part OpenEXR file (Blender's legacy interleaved layout)."""
    with open(path, "rb") as f:
        data = f.read(1 << 20)
    if data[:4] != b"\x76\x2f\x31\x01":
        raise ValueError(f"not an EXR file: {path}")
    version = struct.unpack("<i", data[4:8])[0]
    if version & 0x200:
        raise ValueError(f"multi-part EXR not supported by this header parser: {path}")
    pos = 8
    channels = []
    while True:
        end = data.index(b"\x00", pos)
        name = data[pos:end].decode("latin-1")
        pos = end + 1
        if name == "":
            break
        end = data.index(b"\x00", pos)
        typ = data[pos:end].decode("latin-1")
        pos = end + 1
        size = struct.unpack("<i", data[pos:pos + 4])[0]
        pos += 4
        value = data[pos:pos + size]
        pos += size
        if name == "channels" and typ == "chlist":
            p = 0
            while p < len(value):
                e = value.index(b"\x00", p)
                cname = value[p:e].decode("latin-1")
                if cname == "":
                    break
                channels.append(cname)
                p = e + 1 + 16
    return channels


def pick_blender_channels(available):
    """([R, G, B], depth, mask) channel names from a Blender multi-layer EXR channel list."""
    avail = list(available)

    def pick(*preds):
        for pred in preds:
            for c in avail:
                if pred(c):
                    return c
        return None

    r = pick(lambda c: c.endswith(".Combined.R"), lambda c: c == "R")
    g = pick(lambda c: c.endswith(".Combined.G"), lambda c: c == "G")
    b = pick(lambda c: c.endswith(".Combined.B"), lambda c: c == "B")
    depth = pick(lambda c: c.endswith(".Depth.Z"), lambda c: c.endswith(".Z.Z"), lambda c: c == "Z")
    mask = pick(lambda c: c.endswith(".Object Index.X"), lambda c: c.endswith(".IndexOB.X"),
                lambda c: c.endswith("IndexOB.X"), lambda c: ".Mask." in c, lambda c: c.endswith("Matte.V"))
    return [r, g, b], depth, mask
