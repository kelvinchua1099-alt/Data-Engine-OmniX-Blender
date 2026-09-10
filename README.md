# Data Engine for 4D Scenes — Blender port

Blender re-implementation of the UE5 data engine from **OmniX: Any-view and Any-time 4D Reconstruction via
Feed-forward Trajectory Fields** (`../Data-Engine-OmniX`). Same pipeline, same output contract, no C++ and no
Unreal project: everything is `bpy` Python driven headless with `blender -b`.

| UE5 original | Blender port |
|---|---|
| `Plugins/SceneSetup` (C++ automation test, navmesh) | `blender_engine/scene_analysis.py` + `scene_query.py` (BVH ray casts replace Recast) |
| `Plugins/BoundingBoxTool` (C++) | `blender_engine/bbox_tool.py` |
| `Plugins/VertexTracker` (C++ MRQ render pass) | `blender_engine/render.py` (per-frame evaluated mesh export) |
| `construct_scene/main.py` + `utils/*` | `blender_engine/construct_scene.py`, `trajectory.py`, `camera.py`, `common/geom.py` |
| Level Sequence / Sequencer | `blender_engine/assemble.py` (keyframes + NLA) |
| Movie Render Queue config | `blender_engine/render.py::configure_render` |
| `render_single_scene.sh` | `scripts/render_single_scene.sh` |
| `tools/*.py` | `tools/*.py` (read both UE and Blender output) |

Tested with Blender 5.2.1 LTS (macOS, Metal). Blender 4.2+ should work, the EXR settings branch on version.

## Conventions

* **Units / axes**: meters, right-handed Z-up (Blender). The UE version used centimeters and a left-handed frame.
* **Camera orientation** is stored as `[roll, pitch, yaw]` degrees like the UE annotation, but the sequence JSON
  also carries an explicit `c2w` (OpenCV: x right, y down, z forward) per camera per frame. The tools use `c2w`
  directly; no hand conversion is needed. Intrinsics follow from `fov` (horizontal) and the image size.
* **Frames**: `warmup_frames` (default 4) static frames `-0004 .. -0001` with dynamic objects hidden, then frames
  `0000 .. max_frame-1`. Blender frame `b = f + warmup_frames`.
* **Binary formats** `vertex_data/frame_NNNN.bin` (VTXD) and `faces.bin` (FACE) are byte-identical to the UE
  plugin's output, so the original readers keep working (`blender_engine/common/binio.py`).
* **Images**: one multi-layer EXR per camera per frame, `images/cam_XX/NNNN.exr`, channels
  `<ViewLayer>.Combined.RGBA`, `<ViewLayer>.Depth.Z` (planar camera-space depth, meters) and
  `<ViewLayer>.Object Index.X` (pass index 1 = dynamic foreground). The UE version packed all cameras into one
  EXR; `tools/exr_layout.py` detects either layout.
* **Renderer**: Cycles (needed for the object-index pass). EEVEE renders RGB + depth but no mask.

## Layout

```
Data-Engine-OmniX-Blender/
├── blender_engine/          # importable package (runs inside Blender's python)
│   ├── common/              # geom, convert (camera math), binio (VTXD/FACE), exr_header, log
│   ├── scene_query.py       # SceneBVH: ray casts, box overlap, walkable-surface sampling
│   ├── scene_analysis.py    # bounds, density map, masks, ground-point search  (SceneSetup)
│   ├── bbox_tool.py         # per-frame animation bounding boxes                (BoundingBoxTool)
│   ├── assets.py            # import .blend/.glb/.fbx, actions, NLA retiming
│   ├── trajectory.py        # bezier trajectories + collision checks
│   ├── camera.py            # 16 camera modes, min distance, occlusion checks
│   ├── construct_scene.py   # sampling loop -> SEQUENCE_xxxxxxxx/sequence_xxxxxxxx.json
│   ├── assemble.py          # spec -> Blender objects, keyframes, cameras
│   └── render.py            # multi-camera EXR render + vertex/face export      (MRQ + VertexTracker)
├── scripts/                 # headless entry points (blender -b ... -P script -- args)
├── tools/                   # offline numpy/OpenEXR tools (system python)
├── tests/smoke_test.py      # synthetic end-to-end test
└── config/                  # template.json, collected_object.example.json
```

## 1. Prepare assets

The UE `.uasset` skeletal meshes cannot be read by Blender. You need FBX / glTF / .blend characters with
animations (Mixamo, Sketchfab, Poly Haven ...), and an environment `.blend`.

Environment requirements: static geometry as mesh objects, ground and walkable surfaces facing up, sky domes
named with `sky`/`dome`/`background` (or simply huge) so they are ignored. Lights and cameras in the file feed
the light / camera masks, like PPV / reflection captures did in UE.

Per-animation bounding boxes (replaces the UE `object_data/collected_bbox_info` download):

```bash
BLENDER=/Applications/Blender.app/Contents/MacOS/Blender
$BLENDER -b -P scripts/get_anim_bbox.py -- --asset assets/dog.glb --action Run --out_dir object_data/collected_bbox_info
$BLENDER -b -P scripts/get_anim_bbox.py -- --asset assets/dog.glb --out_dir object_data/collected_bbox_info   # all actions
```

`object_data/collected_object.json` lists objects, each a list of animation variants
(see `config/collected_object.example.json`); relative `asset_path` is resolved against the json's folder.

## 2. Run the pipeline

```bash
export BLENDER=/opt/blender/blender ENV_BLEND=/data/env/city.blend SCENE_TYPE=outdoor FILE_MAP_INDEX=CITY01
export EXPECT_SEQUENCE_NUM=3 WIDTH=1280 HEIGHT=720 SAMPLES=64 DEVICE=GPU
bash scripts/render_single_scene.sh
```

or step by step:

```bash
$BLENDER -b env.blend -P scripts/run_scene_setup.py -- --config config/template.json --output logs/scene_info.json
$BLENDER -b env.blend -P scripts/run_construct.py -- --scene_info logs/scene_info.json \
    --object_file object_data/collected_object.json --bbox_folder object_data/collected_bbox_info \
    --anno_base render_output/DEBUG --expect_sequence_num 3 --group_index 0 --scene_type outdoor
$BLENDER -b env.blend -P scripts/run_render.py -- --sequence_dir render_output/DEBUG/SEQUENCE_00000000 \
    --width 1280 --height 720 --engine CYCLES --samples 64 --device GPU [--save_blend seq0.blend]
```

Output per sequence:

```
render_output/DEBUG/SEQUENCE_00000000/
├── sequence_00000000.json     # spec + annotation: fov, camera_position, rotation, c2w, objects, trajectories
├── images/cam_00/-0004.exr .. 0035.exr        (one folder per camera)
├── vertex_data/frame_0000.bin ..              (VTXD)
├── faces.bin                                  (FACE)
└── render_meta.json                           # resolution, engine, channel names, frames
```

`run_construct.py` accepts `--config overrides.json` to change any key of `construct_scene.DEFAULT_CONFIG`
(frame count, camera ranges, noise, number of cameras ...).

## 2b. Reproducing the forest case (photoreal example)

```bash
python3 assets/download_forest_assets.py                  # ~1 GB of CC0 Poly Haven assets + the Mixamo soldier
$BLENDER -b -P scripts/build_forest_env.py -- --out assets/env/forest_clearing.blend --preview /tmp/forest.png
$BLENDER -b -P scripts/make_asset.py -- --src assets/fox/Fox.glb --dst assets/fox/fox.blend --height 0.7 \
    --bbox_dir assets/collected_bbox_info --object_json assets/collected_object.json
$BLENDER -b -P scripts/make_asset.py -- --src assets/soldier/Soldier.glb --dst assets/soldier/soldier.blend --height 1.8 \
    --bbox_dir assets/collected_bbox_info --object_json assets/collected_object.json --actions Idle,Run,Walk
$BLENDER -b assets/env/forest_clearing.blend -P scripts/run_scene_setup.py -- --cell_count 12 --output assets/case_forest/logs/scene_info.json
$BLENDER -b assets/env/forest_clearing.blend -P scripts/run_construct.py -- --scene_info assets/case_forest/logs/scene_info.json \
    --object_file assets/collected_object.json --bbox_folder assets/collected_bbox_info \
    --anno_base assets/case_forest/render_output --expect_sequence_num 2 --config config/construct_forest.json
$BLENDER -b assets/env/forest_clearing.blend -P scripts/run_render.py -- --sequence_dir assets/case_forest/render_output/SEQUENCE_00000000 \
    --width 960 --height 540 --samples 48
```

`scripts/build_forest_env.py` assembles a 90 m forest clearing: displaced ground with a tiled PBR material,
HDRI sky, ~90 scanned trees (mid LODs) in a ring around a 7 m clearing, rocks and logs, ferns and a few
thousand grass clumps as hair particles (grass and ferns are tagged `omnix_no_collision`). `SceneBVH`
decimates every unique mesh to 20k triangles for the collision BVH, so the 2M-triangle scene is analysed in
seconds. `config/construct_forest.json` narrows the random object scale to 0.8-1.25 so characters keep
realistic sizes (the UE default is 0.2-5).

## 3. Tools (system python: numpy, OpenEXR, opencv-python, scikit-learn)

```bash
python tools/check_multicam_world.py --sequence_path render_output/DEBUG/SEQUENCE_00000000 --frame_id 0 --camera_ids 0-15 --out_dir /tmp/check
python tools/compute_foreground_motion.py --sequence_path render_output/DEBUG/SEQUENCE_00000000 --start_frame 0 --target_frame 10 --camera_id 0 --out_dir /tmp/fg
```

Both auto-detect UE vs Blender layout; `--flip_x` defaults to on for UE data and off for Blender data.

## 4. Smoke test

```bash
$BLENDER -b -P tests/smoke_test.py -- --out /tmp/omnix_smoke --cameras 3 --frames all --max_frame 12
```

Builds a synthetic environment and an animated character, runs every stage, renders with Cycles and checks
that the exported vertex bounding box matches the planned one. Prints a `SMOKE_REPORT` json line.

## Differences from the UE version worth knowing

* **No navmesh.** Walkable points come from vertical ray casts onto up-facing surfaces (`SceneBVH`). The
  three ray checks of `generate_initial_point` are kept; the "ground" check is active but lenient
  (`ground_check_slack`), the "surroundings" check is opt-in. In UE both were effectively no-ops.
* **No PPV / reflection-capture masks** (zeros); light and camera masks are derived from Blender objects.
* **Box traces** are approximated by a BVH overlap test (static boxes) and five parallel rays (sweeps).
* **Animation retiming** uses an NLA strip: `scale = max_frame / frame_number` (UE `play_rate` inverse),
  `action_frame_start = action start + start_frame`.
* **Depth of field** is off by default (`--use_dof` enables it with the UE-style manual focus distance).
* Rendering is one camera per pass; expect Cycles to be slower than UE's real-time renderer.
