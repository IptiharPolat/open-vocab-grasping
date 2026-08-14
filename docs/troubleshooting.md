# Troubleshooting

## Wrong Python or dependency conflicts

The repository requires Python 3.10 or 3.11. The Miniforge base Python 3.13 is
intentionally not used. Confirm the project-specific environment before
debugging package errors:

```bash
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate ovg
which python
python -m open_vocab_grasping.cli doctor
python -m pip check
```

Do not install legacy GraspNet packages into `ovg`; use `ovg-graspnet`.

## No PyBullet window

Batch and tests intentionally use `DIRECT` mode with `ER_TINY_RENDERER`. Set
`simulation.gui: true` only in a graphical Ubuntu session, or run the committed
VS Code configuration / `bash scripts/run_gui_demo.sh`. Check `echo "$DISPLAY"`.
SSH, WSL and containers may additionally need X11/Wayland forwarding. When no
display server is available, use the saved RGB, overlays, PLY, GIF and MP4; this
does not reduce numerical reproducibility.

## CUDA or GPU is unavailable

GPU visibility has four separate layers: physical NVIDIA hardware, a loaded host
driver, device forwarding into the current container/runtime, and the CUDA
toolkit/libraries expected by PyTorch and custom extensions. A working GPU in
another desktop application does not prove this process can see `/dev/nvidia*`.

Check in order:

```bash
nvidia-smi
ls -l /dev/nvidia*
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

On this audited runtime all four process-visible checks fail or are absent, so
reinstalling the Python package cannot fix the boundary. Viable alternatives are:

1. run the complete `open-vocab-simple` geometric baseline on CPU, which is what
   the verified results use;
2. run YOLO-World on CPU and send only the versioned NPZ request to a local or
   remote CUDA machine running `ovg-graspnet`;
3. start an NVIDIA Container Toolkit container with GPU device forwarding and a
   CUDA image compatible with the GraspNet extensions;
4. port/replace the old PointNet2 backend with a maintained CPU/CUDA grasp model,
   but record it as a new method rather than calling it official GraspNet.

If using a remote service, transfer files over an authenticated channel, record
the model/checkpoint hash and return the unchanged schema-v1.0 response. No paid
cloud or API is a core dependency.

## YOLO-World detects nothing

A failed `detect` command never switches to simulator truth. Inspect
`raw_predictions.json`, the prompt, confidence threshold and rendered RGB. The
current procedural bottle/bowl failures are known domain-gap observations. Try a
prompt ensemble or licensed photorealistic assets as a separately configured
experiment; do not lower thresholds only for selected successes. Oracle mode is
for diagnostics and upper-bound comparison only.

If model loading fails, verify the exact weight and CLIP dependency recorded in
`docs/third_party_licenses.md`. Proxy or DNS failures affect downloading, not
already-installed inference.

## GraspNet build or inference fails

Read the full error inside `services/graspnet`; leave the main environment
unchanged. `CUDA_HOME environment variable is not set` means a CUDA development
toolkit is unavailable to the compiler even if a runtime library exists. First
validate the official RGB-D example, then compile PointNet2/KNN, then run
official inference, and only afterward consume PyBullet requests. The service
refuses to emit a result when CUDA is absent so a mock cannot be mislabeled.

## A grasp candidate is unexpectedly rejected

Open `candidates.json` or the episode failure folder. Rejections are cumulative
and may include grasp score, width, workspace, table penetration, approach
direction, point-cloud clearance, pregrasp/grasp IK, external/self collision or
trajectory collision. Change the corresponding YAML value and rerun the same
seed; never patch a hidden constant for one episode.

## A command parsed but the target is not in the scene

`parse` validates syntax and action safety, not scene existence. During capture,
the target name must resolve uniquely against the randomized scene labels. For
example, `red bottle` resolves to the red bottle, while a color that conflicts
with the actual scene is rejected instead of silently grasping another object.
