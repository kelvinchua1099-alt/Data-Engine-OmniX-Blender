"""Blender port of the OmniX 4D data engine (originally UE5).

Pipeline (mirrors the UE version):
    1. scene_analysis   : analyse a static environment .blend  -> scene_info.json
    2. bbox_tool        : per-frame bounding boxes of an animated asset -> json
    3. construct_scene  : sample objects / trajectories / cameras  -> SEQUENCE_xxxxxxxx/sequence_xxxxxxxx.json
    4. assemble + render: rebuild the sequence in Blender, render multi-layer EXR per camera,
                          export per-frame world-space vertices (VTXD) and faces (FACE)
All internal units are meters, world is right-handed Z-up (Blender convention).
"""

__version__ = "0.1.0"
