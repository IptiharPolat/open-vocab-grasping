# Isolated official GraspNet service

Status: official neural inference is verified on this host's RTX 3050. The main
`ovg` environment stays independent and exchanges versioned NPZ files with the
dedicated `ovg-graspnet-cu117` Conda environment. A geometric protocol baseline
still exists for CPU transport tests, but is never reported as GraspNet.

Upstream is `graspnet/graspnet-baseline` pinned at commit
`280c215129f759ed8649cb4e89fc5dfee55f4f80`. The official repository originally
targets PyTorch 1.6 and custom PointNet2/KNN CUDA operators. RTX 3050 is `sm_86`,
so this project uses the closest verified compatibility bridge: official
PyTorch 1.13.1 with CUDA 11.7 and locally compiled upstream operators.

## Build the dedicated environment

```bash
cd /home/ubuntu/open-vocab-grasping
bash services/graspnet/bootstrap-cu117.sh
bash services/graspnet/build-extensions-cu117.sh
```

The environment file pins the compiler and individual CUDA development
libraries. This avoids a moving NVIDIA-channel metapackage resolving historical
CUDA 11.7 requirements to incompatible 13.x headers. It also pins NumPy 1.24,
Open3D 0.18, OpenCV 4.10 and setuptools 59.5 for the old extension build.

Verify that both extensions execute real GPU kernels:

```bash
conda run -n ovg-graspnet-cu117 \
  python services/graspnet/validate_extensions.py
```

Expected host identity includes PyTorch 1.13.1, CUDA 11.7, compute capability
`[8, 6]`, PointNet2 FPS indices and KNN indices.

## Checkpoint

The official RealSense checkpoint Google Drive file ID is
`1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk`. It is stored outside Git at
`weights/graspnet-checkpoint-rs.tar`; it is an epoch-18 checkpoint with SHA-256:

```text
60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868
```

## Reproduce the validation order

Prepare and run the official bundled RGB-D example first:

```bash
mkdir -p outputs/graspnet_official_validation
conda run -n ovg-graspnet-cu117 \
  python services/graspnet/validate_official_example.py \
  --request-output outputs/graspnet_official_validation/request.npz

conda run -n ovg-graspnet-cu117 python services/graspnet/infer.py \
  --request outputs/graspnet_official_validation/request.npz \
  --response outputs/graspnet_official_validation/response.npz \
  --checkpoint weights/graspnet-checkpoint-rs.tar \
  --baseline-root third_party/graspnet-baseline
```

The verified 20,000-point run produced 317 candidates, 109 collision-free.
Export rotatable, headless 3D artifacts with:

```bash
conda run -n ovg-graspnet-cu117 python services/graspnet/visualize_response.py \
  --request outputs/graspnet_official_validation/request.npz \
  --response outputs/graspnet_official_validation/response.npz \
  --output-dir outputs/graspnet_official_validation/visualization --top-k 30
```

This writes `scene.ply`, `grippers_topk.ply` and `visualization.json` without
opening an Open3D window.

Then export PyBullet data in the main environment and run the same service:

```bash
conda run -n ovg python -m open_vocab_grasping.cli export-graspnet \
  --config configs/default.yaml --seed 0

conda run -n ovg-graspnet-cu117 python services/graspnet/infer.py \
  --request <request.npz> --response <response.npz> \
  --checkpoint weights/graspnet-checkpoint-rs.tar \
  --baseline-root third_party/graspnet-baseline
```

The verified saved PyBullet request produced 362 candidates, 44 collision-free.
For the integrated real-YOLO + official-GraspNet pipeline:

```bash
conda run -n ovg python -m open_vocab_grasping.cli run \
  --target mug --seed 0 --config configs/graspnet.yaml
```

Every integrated run preserves `graspnet_request.npz`,
`graspnet_response.npz`, service logs, candidate decisions and video. See
`schema.md` for frames/dtypes and `PROGRESS.md` for actual successes/failures.
