# Third-party sources, versions, and licenses

Checked/downloaded on 2026-08-12. Package versions are captured by `doctor` and
the environment snapshot.

| Component | Source | Version used/status | License/source note |
| --- | --- | --- | --- |
| PyBullet/Bullet | https://github.com/bulletphysics/bullet3 | PyPI 3.2.7 | zlib license; bundled Panda/assets used |
| Franka Panda URDF | PyBullet `pybullet_data/franka_panda` | bundled with 3.2.7 | upstream asset notices apply |
| Open3D | https://github.com/isl-org/Open3D | PyPI 0.19.0 | MIT |
| Gradio | https://github.com/gradio-app/gradio | PyPI 5.50.0 | Apache-2.0; local interactive dashboard only |
| DeepSeek hosted API | https://api-docs.deepseek.com/ | optional external service; `deepseek-v4-flash` configured | Provider terms and usage pricing apply; no model weights vendored |
| Ultralytics YOLO-World | https://docs.ultralytics.com/models/yolo-world/ | Ultralytics 8.4.118; CPU validated | AGPL-3.0; deployment obligations must be reviewed |
| YOLOv8s-World-v2 weights | https://github.com/ultralytics/assets/releases/tag/v8.4.0 | `yolov8s-worldv2.pt`, SHA-256 `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792` | Official Ultralytics asset; model package license obligations apply |
| Ultralytics CLIP fork | https://github.com/ultralytics/CLIP | commit `488e81a6711eea7346872b46ea928b367da8889d`; source archive SHA-256 `e5d76b944483dd028cb1ed6bb339d6bd08d2fe653f2ed755dc91ed4c4a9afdd0` | AGPL-3.0 |
| YOLO-World research code | https://github.com/AILab-CVC/YOLO-World | researched, not vendored | GPL-3.0 repository license |
| GraspNet baseline | https://github.com/graspnet/graspnet-baseline | pinned commit `280c215129f759ed8649cb4e89fc5dfee55f4f80`; official CUDA inference verified on RTX 3050 with PyTorch 1.13.1/CUDA 11.7 | Custom GRASPNET-BASELINE Software License Agreement: academic/non-profit noncommercial research only; commercial use requires permission |
| GraspNet RealSense checkpoint | Official baseline README Google Drive file `1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk` | epoch 18; SHA-256 `60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868` | Covered by upstream custom noncommercial research license |
| NumPy/SciPy/PyYAML/Pillow/imageio/pytest | PyPI | see doctor/pip freeze | open-source; consult package metadata |

Stage 1 creates primitives through PyBullet APIs. Stage 2 additionally uses the
`objects/mug` mesh bundled in `pybullet_data`; no external YCB asset is vendored.
The round bowl mesh under `assets/procedural/` is generated entirely by
`scripts/generate_procedural_assets.py`; it contains no third-party geometry.
