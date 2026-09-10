"""Download the CC0 Poly Haven assets (and the three.js Mixamo soldier) used by scripts/build_forest_env.py.

    python3 assets/download_forest_assets.py [--res 1k] [--tex_res 2k] [--hdri_res 2k]

Files land in assets/polyhaven/<asset_id>/ (models keep their relative textures/ folder so the .blend
links resolve) and assets/soldier/. Re-running skips files that already exist.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.polyhaven.com/files/"

MODELS = ["fir_tree_01", "pine_tree_01", "fir_sapling_medium", "tree_small_02", "dead_tree_trunk_02",
          "rock_moss_set_01", "rock_moss_set_02", "boulder_01", "fern_02", "grass_medium_01"]
TEXTURES = ["forrest_ground_01"]
HDRIS = ["meadow_2"]
SOLDIER_URL = "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/Soldier.glb"


def _curl(url, dst=None):
    """curl is used instead of urllib: the python.org interpreter on macOS often lacks CA certificates."""
    cmd = ["curl", "-sSL", "--fail", "--max-time", "600", "--retry", "3", "--retry-delay", "2", url]
    if dst:
        cmd += ["-o", dst]
        subprocess.run(cmd, check=True)
        return None
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def fetch_json(url):
    return json.loads(_curl(url))


def download(url, dst, tries=4):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(tries):
        try:
            _curl(url, dst + ".part")
            os.replace(dst + ".part", dst)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  retry {i + 1} {os.path.basename(dst)}: {e}")
    raise RuntimeError(f"failed to download {url}")


def get_model(asset_id, res):
    files = fetch_json(API + asset_id)
    entry = files["blend"][res]["blend"]
    root = os.path.join(HERE, "polyhaven", asset_id)
    n = int(download(entry["url"], os.path.join(root, f"{asset_id}_{res}.blend")))
    for rel, inc in entry.get("include", {}).items():
        n += int(download(inc["url"], os.path.join(root, rel)))
    print(f"model {asset_id}: {n} new file(s)")


def get_texture(asset_id, res):
    files = fetch_json(API + asset_id)
    root = os.path.join(HERE, "polyhaven", asset_id)
    got = {}
    for map_name in ("Diffuse", "nor_gl", "Rough", "Displacement", "AO"):
        if map_name not in files or res not in files[map_name]:
            continue
        fmts = files[map_name][res]
        fmt = "jpg" if "jpg" in fmts else ("png" if "png" in fmts else next(iter(fmts)))
        url = fmts[fmt]["url"]
        dst = os.path.join(root, f"{asset_id}_{map_name.lower()}_{res}.{fmt}")
        download(url, dst)
        got[map_name] = dst
    with open(os.path.join(root, "maps.json"), "w") as f:
        json.dump(got, f, indent=2)
    print(f"texture {asset_id}: {list(got)}")


def get_hdri(asset_id, res):
    files = fetch_json(API + asset_id)
    fmts = files["hdri"][res]
    fmt = "hdr" if "hdr" in fmts else "exr"
    dst = os.path.join(HERE, "polyhaven", asset_id, f"{asset_id}_{res}.{fmt}")
    download(fmts[fmt]["url"], dst)
    print(f"hdri {asset_id}: {os.path.basename(dst)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", default="1k", help="model texture resolution")
    p.add_argument("--tex_res", default="2k")
    p.add_argument("--hdri_res", default="2k")
    a = p.parse_args()
    for m in MODELS:
        get_model(m, a.res)
    for t in TEXTURES:
        get_texture(t, a.tex_res)
    for h in HDRIS:
        get_hdri(h, a.hdri_res)
    download(SOLDIER_URL, os.path.join(HERE, "soldier", "Soldier.glb"))
    print("soldier: Soldier.glb")
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())
