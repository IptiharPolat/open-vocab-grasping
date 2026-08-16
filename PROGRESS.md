# Progress

Last updated: 2026-08-14

## Stage 0 - environment audit

Completed:

- Ubuntu 22.04.5 LTS, kernel 6.8.0-124-generic, x86_64.
- AMD Ryzen 7 6800H, 8 cores / 16 threads; about 15 GiB RAM.
- `/home/ubuntu` filesystem: 128 GiB total, 66 GiB available at audit time.
- GCC/G++ 11.4.0, CMake 3.22.1, Git 2.34.1.
- Miniforge 26.3.2. Base Python 3.13.13 is unsuitable for the intended stack;
  project `.venv` was created from the existing Python 3.10.20 environment.
- PCI inventory sees NVIDIA GeForce RTX 3050 Mobile. Kernel module reports
  580.159.03, but this restricted process sees no `/dev/nvidia*` and `nvidia-smi`
  cannot communicate with the driver. Treat GPU/CUDA as unverified here.
- Installed isolated CPU dependencies including PyBullet 3.2.7 and Open3D 0.19.0.
- Reviewed official YOLO-World docs, GraspNet baseline repository, PyBullet quick
  start material, and Open3D pinhole projection docs.

Known issues:

- CUDA toolkit (`nvcc`) is not visible. Host GPU status requires an unrestricted
  host check. No GPU claim is made.
- GraspNet upstream declares PyTorch 1.6 and custom PointNet2/KNN CUDA extensions;
  likely incompatibilities include modern compiler/CUDA/PyTorch APIs and binary ABI.

## Stage 1 - CPU simulation loop (verified)

Completed:

- Deterministic PyBullet DIRECT-mode world with Franka Panda, fixed RGB-D camera,
  table, and three-to-four labelled coloured primitives (`mug`, `bottle`, `bowl`,
  `box`) with randomized pose under a fixed seed.
- RGB, metric depth, instance truth, intrinsic matrix, `T_world_camera`, coloured
  point cloud, workspace crop, PLY export, and headless preview.
- Explicit world / OpenCV-camera / Panda-base / Panda-tool conventions and tested
  projection, backprojection and rigid transforms.
- A clearly labelled geometric top-down generator, candidate records, YAML rank
  weights, width and IK reachability filters, interpolated joint control, gripper
  actuation, lift, and automatic success measurement.
- Full state trace from `RESET` through `EVALUATE` to `DONE/FAILED`, structured
  JSON artifacts, and an actual GIF generated from simulation camera frames.

Verification on 2026-08-12:

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q`: **11 passed**
  in 0.78 s. Tests cover depth, projection/backprojection, transforms,
  association, depth consistency, ranking/unreachable penalty, success metric,
  seeded reproducibility, point-cloud truth agreement, and unreachable IK.
- `capture --config configs/default.yaml`: 11,341 cropped points; simulated table
  surface median z = 0.00203 m; object visible-surface median to body-center errors
  = 0.0208 to 0.0295 m (within the 0.07 m test threshold).
- `smoke --config configs/cpu_smoke.yaml`, seed 7: **success**; 4 generated and
  accepted candidates; target box lift = 0.117416 m; final tool/object distance =
  0.0108725 m; elapsed = 1.271 s including GIF capture.
- Repeating seed 7 before and after GIF instrumentation produced identical lift
  and distance metrics, demonstrating deterministic physics for this case.
- `run --target mug --seed 0 --config configs/default.yaml`: **success**; mug
  proxy lift = 0.116920 m; tool/object distance = 0.0114529 m; elapsed = 1.418 s.

Artifacts:

- Capture: `outputs/20260812_133334_318336_capture_seed0/`
- Successful demo: `outputs/20260812_133437_870097_smoke_seed7/`
- Successful mug run: `outputs/20260812_133553_462414_smoke_seed0/`
- First failed smoke is intentionally retained at
  `outputs/20260812_133338_055103_smoke_seed7/`; root cause was double-counting
  Panda's 0.105 m hand-to-`panda_grasptarget` TCP offset. The correction is
  documented and did not alter the success threshold.

Known stage-1 limitations:

- Perception is explicitly oracle and proposals are explicitly geometric.
- Trajectory collision checking and richer object meshes are implemented/planned
  for later integration stages; stage 1 validates IK and deterministic execution.
- YOLO-World and official GraspNet results are not claimed.

## Stage 2 - YOLO-World open-vocabulary detection (verified on CPU)

Environment and model:

- Migrated active project execution to the user-requested dedicated Conda
  environment `/home/ubuntu/miniforge3/envs/ovg` (`conda activate ovg`), Python
  3.10.20. Existing `base` and `lerobot` environments were not modified.
- Installed PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, Ultralytics 8.4.118, and
  Ultralytics CLIP 1.0 at commit `488e81a6711eea7346872b46ea928b367da8889d`.
- Downloaded official `yolov8s-worldv2.pt` from Ultralytics assets release v8.4.0.
  Model SHA-256 is
  `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`.
- `pip check`: no broken requirements. The full CPU regression suite in `ovg`
  passes: **13 passed in 0.77 s**.

Implementation:

- Dynamic text classes through `YOLOWorld.set_classes`, configurable confidence,
  NMS, image size, device, maximum detections, and evaluation IoU.
- RGB-safe PIL input prevents an ndarray BGR/RGB color swap.
- Real predictions are saved separately from `oracle_truth.json`, with overlay,
  resolved config, environment snapshot, model checksum, model/device identity,
  wall timing and Ultralytics preprocess/inference/postprocess timing.
- Truth is computed from instance segmentation only after inference. Detection
  success and target-selection correctness use configurable box IoU; truth is
  never supplied to YOLO-World.
- Compound color targets are truth-resolved only when the named color matches the
  scene label; e.g. `red bottle` maps to the red bottle while `red bowl` is rejected.
- Upgraded the default perception scene to 640x480, a closer 45-degree-FOV camera,
  and PyBullet's bundled handle-bearing mug mesh. The CPU smoke config retains
  small deterministic primitives and its original camera.
- The mug uses its licensed mesh for rendered perception and a documented
  cylindrical collision proxy for stable manipulation. A point-cloud-only RANSAC
  cylinder-center estimate rejects handle outliers; it does not read body pose.

Measured results on 2026-08-12, seed 0, CPU:

- `mug`: **1 prediction**, confidence 0.205413, truth IoU **0.967390**,
  detection success true, target selection true. Model inference 75.30 ms;
  detector wall time 98.45 ms.
- Revalidated after the perception/manipulation asset integration: `mug` remained
  detected with IoU 0.960534, and the oracle-perception geometric executor lifted
  the mesh-rendered mug by 0.104213 m without reading its body pose for planning.
- Earlier low-resolution primitive-mug baseline: **0 predictions**. This failure
  motivated a documented perception-scene correction rather than an oracle fallback.
- `red bottle`: **0 predictions**; measured failure caused by the current simple
  compound cylinder appearance/domain gap.
- `bowl`: **0 predictions**; measured failure caused by the current simple short
  cylinder appearance/domain gap.

Artifacts:

- Successful real mug detection:
  `outputs/20260812_142139_666676_detect_seed0/`
- Successful mesh-rendered mug execution:
  `outputs/20260812_142357_834280_smoke_seed0/`
- Initial low-resolution failure:
  `outputs/20260812_141344_509914_detect_seed0/`
- Red-bottle failure: `outputs/20260812_141700_589321_detect_seed0/`
- Bowl failure: `outputs/20260812_141708_279292_detect_seed0/`

Known stage-2 limitations:

- At the original stage-2 checkpoint only mug was detected. The 2026-08-14
  prompt-ensemble update below adds verified bottle detection; bowl/box remain.
- Confidence 0.205 is above the configured 0.10 simulation-domain threshold but
  modest; this must be re-evaluated with richer textures and randomized lighting.
- This line described the original restricted audit. The host-side GraspNet GPU
  validation in stage 3 supersedes it.

Prompt-ensemble update verified on 2026-08-14:

- Moved per-target dynamic prompt sets into YAML and lowered the simulation
  threshold from 0.10 to 0.05 after inspecting raw predictions. The detector
  still runs the official model and does not receive an instance mask.
- `mug/cup/coffee mug` raises the seed-0 mug confidence to 0.6567 in the prompt
  probe. `bottle/water bottle/red bottle` yields one real bottle box at 0.0871
  confidence with truth IoU **0.907828** and correct target selection.
- Verified CLI artifact: `outputs/20260814_153547_192896_detect_seed0/`.
- Bowl and box still have no correct prediction with the present primitive
  appearance; these remain the next perception-asset/domain-gap work item.

## Stage 3 - official GraspNet GPU inference and PyBullet integration (verified)

Completed and verified on 2026-08-14:

- Re-audited outside the restricted process: the host has an NVIDIA GeForce RTX
  3050 Laptop GPU (4 GiB), driver 580.159.03 and compute capability 8.6. The main
  `ovg` environment intentionally remains CPU-only; neural grasp inference runs
  in the dedicated `ovg-graspnet-cu117` Conda environment.
- Built a reproducible Python 3.10 / official PyTorch 1.13.1 + CUDA 11.7 stack.
  Exact CUDA 11.7 compiler, CUDART, CCCL/Thrust, cuBLAS, cuSPARSE and cuSOLVER
  development packages are pinned in `services/graspnet/environment-cu117.yml`.
- Compiled the official PointNet2 and KNN CUDA extensions for `sm_86`. A real GPU
  validation executed farthest-point sampling and KNN, returning FPS indices
  `[0, 3]` and one-based nearest-neighbour index `[1]`.
- Ran the official repository's example data first, with the official epoch-18
  RealSense checkpoint. A 20,000-point forward pass produced 317 6-DoF
  candidates; 109 passed official model-free collision detection. No mock or
  geometric result is counted as this validation.
- Exported headless Open3D visualizations for both official and PyBullet runs:
  metric `scene.ply` plus the top 30 collision-free official gripper meshes in
  `grippers_topk.ply`, directly viewable in VS Code/Open3D.
- Replaced the input with saved PyBullet RGB-D. The official model produced 362
  candidates; 44 passed model-free collision detection. The schema response was
  parsed by the main `ovg` environment.
- Integrated the isolated process into `run_open_vocab_grasp` through
  `configs/graspnet.yaml`. Each run preserves the exact NPZ request/response and
  service stdout/stderr. CPU defaults still select the clearly labelled
  geometric baseline.
- Derived and tested the full GraspNet-to-Panda adapter. GraspNet +x approach
  maps to Panda tool +z, GraspNet +y closing maps to Panda tool +y (confirmed from
  Panda URDF finger axes), and tool +x maps to -GraspNet +z. The contact TCP also
  applies the official per-candidate `depth` translation.
- Added official collision rejection plus an RGB-D-only semantic target-center
  estimate. Candidate centers are transformed into the target grasp frame and
  rejected when the object lies outside the official 2 cm gripper-height slice.
  Simulator truth remains excluded from semantic selection.
- Full CPU regression suite after integration: **37 passed in 0.94 s**.

Actual end-to-end status:

- Seed-0 bottle is a verified real-YOLO + official-GraspNet success. A configured
  prompt ensemble detected it with IoU 0.907828. GraspNet produced 1,003
  candidates; 16 passed box/depth association and one passed official collision,
  table, predicted-width, semantic-height, point-cloud, IK and trajectory checks.
  Panda executed an oblique side grasp and reached `DONE`: lift 0.117675 m and
  final tool/object distance 0.024640 m. No instance truth entered selection.
- Seed-0 mug remains a useful measured failure. Real YOLO detected it (IoU
  0.960534), GraspNet generated/filtered candidates and Panda executed through
  close/lift, but the selected pose closed on the rim/one wall and did not lift.
  It remains `object_not_lifted`, not reported as success.
- The RGB-D estimated object center `[0.46795, -0.19053, 0.07380] m` was within
  about 1.3 cm of truth, demonstrating that the remaining failure is grasp pose
  quality/contact geometry rather than a gross camera/world transform error.
- The initial bottle attempt did stop at `detection_failed`. A prompt-ensemble
  regression (`bottle`, `water bottle`, `red bottle`) fixed perception without
  oracle fallback; the successful rerun above supersedes that initial failure.

Artifacts:

- Official example: `outputs/graspnet_official_validation/`.
- PyBullet service validation: `outputs/graspnet_pybullet_validation/`.
- Full execution/failure video and all candidate reasons:
  `outputs/20260814_152123_017523_run_seed0/`.
- Successful real-YOLO + official-GraspNet bottle run:
  `outputs/20260814_153729_431593_run_seed0/`.
- Reproducible environment/build scripts and schema: `services/graspnet/`.

## Stage 4 - semantic/geometric association (verified on CPU)

Completed on 2026-08-13:

- Added scene-level geometric proposal generation from the full cropped RGB-D
  cloud: table-height removal, Open3D DBSCAN clustering, geometric top-down
  candidates per cluster. It does not read PyBullet instance IDs.
- Added explicit world-to-camera transformation, pinhole projection, inclusive
  box test, shrunken-box median depth reference, configurable depth tolerance,
  normalized box-center score and structured rejection reasons.
- Added immutable association records containing camera-frame center, projected
  pixel, candidate/reference depth, depth delta, acceptance and rejection reasons.
- Added `associate` CLI and YAML settings for clustering, box shrinkage, depth
  tolerance and truth-only center accuracy.
- Added 2D overlay and 3D top-down visualizations: green candidates are retained;
  orange candidates are rejected. Candidate indices match JSON records.
- Regression suite: **16 passed in 0.80 s** in the final verification.

Actual seed-0 mug experiment:

- Method: `real_yolo_world + geometric_scene_baseline`.
- YOLO detections: 1; scene clusters: 4; candidates: 16; retained: 4.
- Association succeeded. The top-ranked retained center was 0.013217 m from the
  mug truth center and passed the configured 0.08 m truth threshold.
- Selection used no truth IDs (`truth_used_for_selection=false`); segmentation/
  AABB truth was queried only after selection to report the metric.
- Final repeat: YOLO wall time 0.123485 s; full association pipeline 6.122723 s
  on CPU, with identical counts and selected-center error.
- This is not a GraspNet result; generator is explicitly recorded as
  `geometric_scene_baseline_not_graspnet`.

Artifacts:

- `outputs/20260812_235957_378687_associate_seed0/association_2d.png`
- `outputs/20260812_235957_378687_associate_seed0/association_3d_topdown.png`
- `outputs/20260812_235957_378687_associate_seed0/association_records.json`
- `outputs/20260812_235957_378687_associate_seed0/candidates.json`
- Final repeat: `outputs/20260813_000153_197595_associate_seed0/`

## Developer visualization usability

- Added `configs/gui_demo.yaml` for interactive PyBullet rather than automated
  DIRECT mode. It uses real-time pacing, scene labels, grasp-frame axes and a
  ten-second final-state hold.
- Added VS Code Run and Debug entries for the GUI grasp demo, real YOLO/3D
  association, and the CPU regression suite. All select the dedicated `ovg`
  Conda interpreter instead of base Python.
- Added `scripts/run_gui_demo.sh` as an equivalent ordinary desktop-terminal
  entrypoint. GUI display remains intentionally separate from headless CI and
  reproducible batch evaluation.

## Stage 5 - planning filters and text-target execution (verified on CPU)

Completed on 2026-08-13:

- Added configurable filters for minimum grasp score, Panda width range, workspace
  bounds, table-center clearance, top-down approach alignment and non-target
  point-cloud clearance.
- Added pregrasp and final-grasp IK checks, joint-space motion cost, interpolated
  home-to-pregrasp and pregrasp-to-grasp collision checks, external-body and
  non-adjacent self-collision checks.
- PyBullet IDs are used only by the physics collision engine and final success
  metric. YOLO plus RGB-D geometry still determine semantic selection; result
  records explicitly state `truth_used_for_semantic_selection=false`.
- Promoted `run` from the old stage-1 guard to the real-YOLO `open-vocab-simple`
  pipeline. `smoke` remains the separately labelled oracle CPU regression path.
- The VS Code GUI launch and `scripts/run_gui_demo.sh` now execute this complete
  real-YOLO pipeline and draw the selected grasp frame.
- Added analytic table/point-cloud collision tests and an actual PyBullet
  external-obstacle trajectory collision test. Final suite: **19 passed in 0.95 s**.

Actual seed-0 mug run:

- YOLO detections 1; scene clusters 4; generated candidates 16; associated 4;
  geometry 4; point-cloud collision 4; IK/trajectory 4.
- Selected center error before execution: 0.013217 m.
- State machine reached `DONE`: pregrasp, approach, close, lift and evaluation.
- Target lift: **0.108188 m**; final target/tool distance: **0.035911 m**;
  end-to-end success true.
- Final video-producing repeat: detector wall time 0.111251 s; total CPU wall
  time 8.279421 s.
- Generator remains explicitly `geometric_scene_baseline_not_graspnet`.

Artifacts:

- `outputs/20260813_113306_340633_run_seed0/demo.gif`
- `outputs/20260813_113306_340633_run_seed0/demo.mp4` (H.264, 640 x 480,
  12 FPS, 3.0 s; decoded successfully with imageio)
- `outputs/20260813_113306_340633_run_seed0/result.json`
- `outputs/20260813_113306_340633_run_seed0/candidates.json`
- `outputs/20260813_113306_340633_run_seed0/filtered_candidates_2d.png`
- `outputs/20260813_113306_340633_run_seed0/filtered_candidates_3d.png`

## Stage 6 - fixed-seed batch evaluation (verified on CPU)

Completed on 2026-08-13:

- Implemented `evaluate` for paired target/seed runs. `--episodes N` is N
  episodes per target per enabled mode and is recorded in the summary.
- Implemented flat `episodes.csv`, aggregate `summary.json`, human-readable
  `summary.md`, environment/config snapshots, videos, detections, point-cloud
  artifacts and per-episode failure-case directories.
- Added rates for text target selection, detection, candidate generation, IK
  reachability, grasp and end-to-end success; added detector, grasp generation,
  planning, execution and total wall times plus failure distributions.
- Ensured a fair oracle comparison: oracle and open-vocabulary modes use the same
  table, point-cloud, IK and trajectory filters; only target perception is
  replaced by instance truth in the explicitly labelled oracle mode.
- The original 2026-08-13 CPU evaluation recorded GPU modes as unavailable at
  that time. The 2026-08-14 extension below supersedes that status for
  `open-vocab-graspnet`; `graspnet-only` still receives no invented values.
- Added summary regression tests. Final CPU suite: **20 passed in 0.90 s** after
  the fair-filter change.

Actual final experiment:

- Command: `evaluate --targets mug,bottle,bowl --episodes 3` with modes
  `oracle-perception,open-vocab-simple` and seed start 100.
- Planned and actual episodes: 18; CSV data rows: 18; failure directories: 10.
- `oracle-perception`: detection/selection 9/9, candidate generation 9/9,
  IK reachable 6/9, grasp and end-to-end success **5/9 (55.6%)**.
- `open-vocab-simple`: detection/selection 3/9, candidate generation 9/9,
  IK reachable 3/9, grasp and end-to-end success **3/9 (33.3%)**.
- Open-vocabulary mug: 3/3 success. Bottle and bowl: 0/3 each due to honest
  YOLO detection failures on the current procedural assets.
- Failure distribution: six `detection_failed`, three `no_accepted_candidate`
  (all bowl candidates record `table_penetration`), one `object_not_lifted`.
- Mean total time: 1.869 s oracle, 3.496 s open-vocabulary on CPU.

Final artifacts:

- `outputs/20260813_115220_286118_evaluation/episodes.csv`
- `outputs/20260813_115220_286118_evaluation/summary.json`
- `outputs/20260813_115220_286118_evaluation/summary.md`
- `outputs/20260813_115220_286118_evaluation/videos/`
- `outputs/20260813_115220_286118_evaluation/failure_cases/`
- `docs/failure_analysis.md`

GPU evaluation extension verified on 2026-08-14:

- Removed the obsolete hard-coded `open-vocab-graspnet unavailable` status.
  The evaluator now selects the official isolated backend per mode and records
  `graspnet_inference_s` in CSV/JSON/Markdown; `graspnet-only` remains explicitly
  unimplemented with no synthetic score.
- Actual command: `evaluate --targets mug --episodes 1 --modes
  open-vocab-graspnet --config configs/evaluation_graspnet.yaml`.
- Seed 100: real detection, target selection and candidate generation were 1/1;
  IK/end-to-end were 0/1 with `no_accepted_candidate`. Official inference took
  6.039 s and total time was 12.759 s.
- Artifact: `outputs/20260814_152848_741177_evaluation/`.

Required 10-seed-per-target GPU evaluation completed on 2026-08-14:

- Command: `evaluate --targets mug,bottle --episodes 10 --modes
  open-vocab-graspnet --config configs/evaluation_graspnet.yaml`.
- Exactly 20 CSV rows: mug 10, bottle 10. Candidate generation was 20/20;
  detection/target selection 17/20; IK reachable 13/20; end-to-end success
  **6/20 (30%)**.
- Bottle: detection 7/10, IK 6/10, grasp/end-to-end **6/10 (60%)**. Successful
  seeds were 110, 111, 113, 115, 118 and 119.
- Mug: detection 10/10, IK 7/10, end-to-end **0/10**. Seven executed poses failed
  contact/lift and three had no accepted candidate.
- Failure distribution: 3 `detection_failed`, 4 `no_accepted_candidate`, 7
  `object_not_lifted`. Mean official GraspNet inference 6.002 s; mean total
  episode time 11.155 s.
- Data consistency check reproduced target counts, success seeds and failures
  directly from CSV; 14 failure directories and 40 GIF/MP4 files exist.
- Artifact: `outputs/20260814_154643_981832_evaluation/`.

## Stage 7 - portfolio and reproducibility handoff (verified on CPU)

Completed on 2026-08-13:

- Promoted the deterministic English command parser into the CLI. Both
  `parse --instruction "pick the red mug"` and
  `run --instruction "pick the mug"` now emit/consume the same validated
  `{action, target, destination}` contract.
- Added a Draft 2020-12 JSON Schema, an explicit `pick` action whitelist,
  rejection of extra fields/unapproved actions, and tests. No natural-language
  or external-model output is evaluated as Python code.
- Rebuilt the README around an honest capability matrix, Mermaid architecture,
  dedicated Conda install, CPU/GPU commands, CLI reference, VS Code GUI,
  coordinate/evaluation links, actual results, limitations and FAQ.
- Expanded architecture and troubleshooting documentation. CUDA troubleshooting
  now separates physical GPU, host driver, device forwarding, toolkit and
  PyTorch visibility and lists CPU, remote NPZ service, GPU container and
  alternative-backend options.
- Added portfolio-ready Chinese resume and English CV descriptions, five
  implementation-based interview questions, and a staged migration plan for a
  calibrated real RGB-D camera and SO-ARM101 with controller/safety boundaries.
- Extended `doctor` to report the actual Python prefix/environment, shell Conda
  state, package managers, disk, device nodes, package versions and CUDA state.
  This distinguishes direct use of the `ovg` interpreter from the caller shell's
  active environment.

Final verification:

- CPU regression suite: **28 passed in 1.11 s**.
- `pip check`: no broken requirements.
- All configuration YAML and committed VS Code JSON parsed successfully.
- All shell entrypoints passed `bash -n`; all local Markdown links resolved.
- Natural-language end-to-end command actually executed:
  `run --instruction "pick the mug" --seed 0 --config configs/default.yaml`.
  The state machine reached `DONE`, lifted the mug **0.108188 m**, retained four
  IK/trajectory-valid candidates and recorded
  `truth_used_for_semantic_selection=false`.
- The run remains explicitly `real_yolo_world +
  geometric_scene_baseline_not_graspnet`; no unavailable GraspNet claim or
  number was introduced.

Final natural-language run artifacts:

- `outputs/20260813_125234_542676_run_seed0/result.json`
- `outputs/20260813_125234_542676_run_seed0/demo.gif`
- `outputs/20260813_125234_542676_run_seed0/demo.mp4`

The original CPU stages remain reproducible. Official GraspNet neural inference
has since been validated on the host RTX 3050 and integrated/evaluated above.
Remaining work inside the current Panda-simulation scope is domain-gap and
contact-reliability improvement for the mug/bowl/box assets. The earlier
SO-ARM101 migration note is retained only as historical design material; the
user has assigned that hardware to a separate project, so it is not an
unfinished deliverable here.

## Interactive visual dashboard - verified on CPU

Completed on 2026-08-13:

- Added a local Gradio control room connected directly to the existing pipeline,
  not a separate or mocked demonstration path. Users can enter a schema-validated
  instruction and select RGB-D capture, YOLO detection, semantic/3D association,
  or complete grasp execution.
- The UI renders RGB, 2D detection/association overlays, 3D top-down plots, a
  rotatable PLY point cloud, candidate-level scores/rejection reasons, state
  history, structured truth-boundary metrics and the actual execution MP4.
- Restricted the dispatcher to four explicit actions, serialized execution at
  concurrency one, bound the service to `127.0.0.1`, disabled public sharing and
  analytics, and exposed only the project output directory.
- Added the optional `ui` dependency, `dashboard` CLI command,
  `scripts/run_dashboard.sh`, and a VS Code Run and Debug entry. The launcher
  safely removes the desktop's invalid `socks://` proxy variables only for this
  local process.
- Added dashboard artifact/rendering and action-whitelist tests. Full CPU suite:
  **30 passed in 1.08 s** in the final regression run.

Actual verification:

- Instantiated the complete 30-component UI under Gradio 5.50.0.
- Started the server on `127.0.0.1:7861`; both `/` and `/config` returned HTTP
  200, the latter reporting analytics disabled.
- Ran the full dashboard action with `pick the mug`, seed 0. It used real
  YOLO-World, generated 16 geometric candidates, retained four, traversed the
  complete state machine to `DONE`, and exposed the generated point cloud/video.
- Verified output:
  `outputs/20260813_141744_650541_run_seed0/`. Truth remained excluded from
  semantic selection; the candidate source remained explicitly
  `geometric_scene_baseline_not_graspnet`.
- Hardened the dashboard launcher after a real nested-environment failure: it
  now invokes the dedicated `ovg/bin/python` directly and removes inherited
  `.venv`, `PYTHONHOME`, ROS `PYTHONPATH` and invalid proxy variables. Import
  failures now retain the underlying exception instead of reporting every cause
  as an absent package.

## DeepSeek terminal Agent - real API end-to-end verified

Implemented on 2026-08-14:

- Added an interactive VS Code terminal Agent with `/help`, `/mode`, `/seed`,
  `/status`, `/last` and `/quit`, plus a reproducible one-shot CLI path.
- Added a real DeepSeek HTTP client configured for the current official
  `deepseek-v4-flash` Chat Completions endpoint and JSON Output. The client reads
  `DEEPSEEK_API_KEY` only at request time and never serializes headers or secrets.
- Added a bilingual task-planning Prompt, Draft 2020-12 grasp-plan Schema, exact
  action/step constraints and current-scene target whitelist. Model-written
  Python, extra fields, unknown targets and reordered steps are rejected before
  any simulator action.
- Added three separated modes: real `deepseek`, labelled offline `mock`, and the
  original English `deterministic` parser.
- Added per-run `agent_request.json`, `agent_response.json`, `agent_plan.json`
  and `agent_result.json` audit files alongside perception, candidate and video
  artifacts.
- Added missing-key, unsafe-field, invalid-step, unknown-target, Chinese Mock,
  response parsing and secret-redaction regression tests. Full suite:
  **35 passed in 1.13 s** in the final post-API regression run.

Actual Mock end-to-end verification:

- Command: `agent --mode mock --instruction '请帮我抓取桌面上的杯子' --seed 0`.
- The explicitly labelled Mock produced target `mug`; the local validator
  accepted the canonical plan; real YOLO-World generated the semantic result;
  16 geometric candidates were generated and four retained.
- Panda traversed the full execution state machine to `DONE`; target lift was
  0.108188 m and the run reported success.
- Artifacts: `outputs/20260814_115147_209693_run_seed0/`.
- This is not a real DeepSeek result and is not a GraspNet result.

Real DeepSeek end-to-end verification on 2026-08-14:

- Command: `agent --mode deepseek --instruction '请帮我抓取桌面上的杯子' --seed 0`.
- The hosted API returned HTTP 200. The actual response model was
  `deepseek-v4-flash`; the request used JSON Output and the locally validated
  plan selected target `mug` with the canonical six high-level steps.
- LLM latency was 0.964554 s. Usage was 231 prompt tokens, 73 completion tokens
  and 304 total tokens, including 128 cached prompt tokens.
- The downstream real YOLO-World path detected the mug, generated 16 geometric
  candidates, retained four after association/filtering, passed IK and reached
  `DONE`. Lift was 0.108188 m; tool-object distance was 0.035911 m.
- Semantic selection did not use simulator truth. The grasp backend remained
  explicitly `geometric_scene_baseline_not_graspnet`.
- Verified all four Agent audit files and searched them for `Authorization`,
  `Bearer`, `DEEPSEEK_API_KEY` and key-like strings; no credential material was
  found.
- Artifacts: `outputs/20260814_122735_073119_run_seed0/`.

Agent live visualization verification:

- Added `configs/agent_gui.yaml` and `scripts/run_agent_gui.sh`. The interactive
  terminal remains the command source; after plan validation, PyBullet opens a
  separate native window with real-time simulation pacing, labels and grasp axes.
- The launcher checks `DISPLAY`, uses the dedicated `ovg` interpreter and
  preserves `DEEPSEEK_API_KEY`. The terminal prints the generated MP4 path after
  every completed run so it can be opened directly from VS Code.
- Actually launched the GUI on `DISPLAY=:0`. PyBullet created an NVIDIA OpenGL
  3.3 context on a GeForce RTX 3050 Laptop GPU, displayed the complete Panda
  grasp, held the final state, and exited cleanly.
- GUI validation used the explicitly labelled Mock planner to avoid an extra API
  charge; downstream YOLO, point cloud, filters, IK and execution were real. It
  reached `DONE` with lift 0.108188 m.
- GUI artifact: `outputs/20260814_123228_950497_run_seed0/demo.mp4`.
- Added GUI config inheritance regression coverage. Final suite: **36 passed in
  1.10 s**.

Validated Python planning DSL added on 2026-08-14:

- Every schema-valid LLM plan is compiled locally to `agent_generated_plan.py`,
  containing only the six allowlisted `controller` calls. The model never gets
  an arbitrary Python execution channel.
- The generated AST must be one undecorated function with one argument and the
  exact canonical calls/target. Imports, builtins, arbitrary attributes,
  reordered calls, extra statements and target changes are rejected.
- The validated code runs with an empty `__builtins__` mapping against a plan
  recorder. Its trace must exactly match the JSON plan before the normal robot
  pipeline is dispatched. The trace is saved as
  `agent_generated_plan_trace.json`.
- Regression tests cover valid compilation/execution plus import, arbitrary call
  and target-tampering rejection. Full suite: **41 passed in 0.98 s**.
- Actual no-cost Mock-planner validation used real downstream YOLO/geometric
  execution and reached `DONE`, lifting 0.108188 m. Mock is not reported as a
  real API result. Artifact: `outputs/20260814_153224_136681_run_seed0/`.
- Added `configs/agent_graspnet.yaml` and trusted-config execution-mode binding.
  A labelled Mock high-level plan then exercised the complete real YOLO +
  official GraspNet backend in the same audited run and reached `DONE`: bottle
  lift 0.118767 m, one accepted candidate, and plan/result mode both
  `open-vocab-graspnet`. Artifact:
  `outputs/20260814_155614_596265_run_seed0/`. This proves integration but is not
  claimed as a real DeepSeek API call.

## Final verification snapshot - 2026-08-14

- Main regression suite: **42 passed in 0.96 s**.
- Main `ovg` environment: `pip check` reports no broken requirements; all YAML
  configs and VS Code launch JSON parse successfully.
- Isolated GPU regression reran successfully on NVIDIA GeForce RTX 3050 Laptop,
  compute capability 8.6, PyTorch 1.13.1/CUDA 11.7. PointNet2 FPS and KNN both
  executed CUDA kernels with the expected test indices.
- Final `doctor` sees driver 580.159.03, 4,096 MiB GPU, checkpoint and compiled
  extensions. Main PyTorch remains CPU-only by design; GraspNet is isolated.
- Disk after environments, weights and experiments: 25.48 GiB free. Main env is
  3.7 GiB, GraspNet env 8.2 GiB, weights 37 MiB and the 20-episode GPU evaluation
  36 MiB.
- The agent process used for this final audit does not inherit the user's shell
  `DEEPSEEK_API_KEY`; therefore the final combined DeepSeek+GraspNet API call is
  intentionally left for the user's configured terminal. A real DeepSeek call
  and a separate real GraspNet bottle success are both already evidenced; the
  combined labelled-Mock integration run is also verified and must not be
  misreported as a hosted API result.

## Grasp reliability hardening - 2026-08-14

- Added an RGB-D-only parallel-jaw contact-volume metric. It records visible
  bilateral balance, enclosure, target point count and an explicit evidence
  mode for every geometry-valid candidate; no simulator pose or instance ID is
  used for this score.
- Added bounded GraspNet center refinement. Network rotation, depth, score and
  collision flag remain unchanged; candidate translation is accepted only
  under semantic-center/contact constraints and the shift is logged.
- Added a single-view occlusion fallback limited to nearly top-down candidates
  within 0.012 m of the semantic RGB-D center and with at least 100 visible
  target points. It is recorded as `centered_topdown_single_view_fallback`, not
  as visible bilateral evidence.
- Added the physically equivalent 180-degree parallel-jaw orientation as an IK
  fallback. Usage is logged in `used_parallel_jaw_symmetry`.
- Closed a planning gap by checking lift IK and the grasp-to-lift collision path
  before execution. The earlier candidate that failed at runtime now receives
  `lift_ik_unreachable` during selection.
- Corrected the mesh mug's visual/collision mismatch. The PyBullet mesh body is
  0.082 m wide at unit scale; `scene.mug_mesh_scale: 0.90` produces a 0.0738 m
  body that fits Panda's 0.080 m opening, and the collision cylinder now uses
  the same body scale. Explicit object and rubber-like finger friction are
  configured; the success threshold remains unchanged.
- Increased only the GraspNet *predicted-width tolerance* from 0.090 to 0.095 m
  to admit small sim-domain bias. Panda's commanded physical opening remains
  clamped to 0.080 m.
- Real YOLO-World still detects the corrected mug at seed 0: IoU 0.940576,
  artifact `outputs/20260814_171530_582568_detect_seed0/`.
- **New real-YOLO + official-GraspNet mug success:** 1,008 candidates, one final
  accepted candidate after lift-IK filtering, lift 0.117182 m, state `DONE`.
  Artifact: `outputs/20260814_171731_976396_run_seed0/`.
- Bottle regression remains successful: two final candidates, lift 0.116492 m,
  state `DONE`. Artifact: `outputs/20260814_171805_457738_run_seed0/`.
- Historical failed intermediate runs were retained, including the 0.056032 m
  near miss. No threshold was lowered and no failed run was relabelled.
- Main regression suite after these changes: **48 passed in 0.99 s**. The prior
  20-episode report remains the historical pre-improvement baseline and must be
  rerun before claiming a new aggregate success rate.

## Low-confidence prompt consensus and four-class seed-0 closure - 2026-08-14

- Diagnosed YOLO-World at confidence 0.001 without simulator truth in the
  detector. The real bowl produced spatially agreeing low-score boxes for
  `bowl`, `green bowl` and `ceramic bowl`; the old yellow primitive box produced
  no box-aligned response and was frequently confused with the mug.
- Added a retry that runs only after the normal 0.05-confidence pass returns no
  boxes. It accepts a low-score box only when at least two distinct prompts
  overlap at IoU 0.55. Consensus confidence uses geometric-mean prompt support
  times vote count, reducing single-prompt outliers.
- Added fixed-tabletop fallback geometry checks: boxes occupying more than 8%
  of the image or touching its border are rejected only in the low-confidence
  retry. This removed table/robot-wide false positives without oracle data.
- Replaced the untextured yellow cube visual with a project-owned procedural
  cardboard parcel assembled from primitive shapes, including tape and a light
  label. Its physics remains a simple 0.080 x 0.060 x 0.060 m box. No external
  texture or unlicensed asset was added.
- Seed-0 real detection results after the change: bowl IoU 0.935826 and box IoU
  0.901423, both with correct target selection. Artifacts:
  `outputs/20260814_172807_250356_detect_seed0/` and
  `outputs/20260814_172751_812855_detect_seed0/`.
- Added adaptive pregrasp IK checks constrained to the specified 0.10, 0.09 and
  0.08 m distances. The selected distance is logged and reused by execution.
- **Four-class seed-0 real closure is now verified:**
  - mug: lift 0.117182 m, `outputs/20260814_171731_976396_run_seed0/`;
  - bottle: lift 0.116492 m, `outputs/20260814_171805_457738_run_seed0/`;
  - bowl: lift 0.089291 m, `outputs/20260814_172858_796852_run_seed0/`;
  - box: lift 0.108922 m, `outputs/20260814_173204_467222_run_seed0/`.
- All four runs used real YOLO-World, the official GraspNet checkpoint and no
  simulation truth for semantic selection. Aggregate post-change reliability
  is still pending a fixed-seed batch rerun.
- Regression suite after prompt consensus and adaptive planning: **53 passed in
  0.96 s**.

## Executable semantic-free GraspNet baseline - 2026-08-14

- Implemented `graspnet-only` as an executable evaluation mode. It skips YOLO
  entirely and associates each official GraspNet proposal only with its nearest
  RGB-D geometry cluster. All tabletop object bodies are allowed solely for
  physics contact checks; no target identity is provided to selection.
- The requested target is consulted only after ranking/execution for the success
  metric. Detection and target-selection rates are emitted as `n/a`, not 0%,
  100% or a fabricated score.
- Actual seed-0 bottle baseline: 1,003 candidates, two accepted, IK reachable,
  but the text-free selector chose a grasp 0.381566 m from the requested bottle
  and the bottle was not lifted. This honest failure demonstrates why semantic
  association matters. Evaluation artifact:
  `outputs/20260814_173624_100867_evaluation/`; source run:
  `outputs/20260814_173624_101325_run_seed0/`.

## Batch geometry accuracy and executable filter ablation - 2026-08-14

- Added `geometry-eval`, which uses PyBullet segmentation only as explicitly
  labelled evaluation truth. It measures pixel projection/backprojection,
  camera/world round trips, table height and visible/robust object point-cloud
  centers over fixed random seeds.
- Actual 10-episode run (seeds 200-209, 40 object samples) passed: maximum pixel
  round-trip error 1.271057e-13 px, maximum camera/world round-trip error
  3.333559e-16 m, maximum table error 0.000610 m, robust object-center mean
  0.008773 m and maximum 0.023755 m. Artifact:
  `outputs/20260814_174005_700158_geometry_benchmark/`.
- Added `ablate`, which reruns the same deterministic RGB-D/GraspNet scene for
  full filtering, no depth consistency, no contact support, no scene-cloud
  clearance and no-text GraspNet. These are actual simulator executions, not
  inferred counts from incomplete rejection logs.
- Actual mug seed-0 ablation ran five variants and all five lifted the mug. This
  easy seed therefore does not establish that every filter improves success;
  it does show depth consistency reduced associated candidates from 37 to 32.
  The no-text variant happened to select the mug here, unlike the bottle-target
  baseline where it selected another object. Artifact:
  `outputs/20260814_174152_205931_filter_ablation/`.
- Regression suite after the implemented comparison/accuracy commands:
  **54 passed in 1.00 s**.

## Small-object GraspNet ranking and current four-class benchmark - 2026-08-15

- Diagnosed the scaled bowl failure from retained candidate logs. The official
  network produced 287 candidates and 62 initially passed planning, but ranking
  selected a high-score pose 0.0192 m from the RGB-D target center and the bowl
  did not lift. No simulator pose was used in this diagnosis or the fix.
- Added a YAML-weighted semantic-center-distance penalty and a conservative
  `maximum_topdown_refined_center_distance_m: 0.015` gate for candidates whose
  network approach axis is replaced by the explicit tabletop top-down prior.
  Rejection reason `topdown_refinement_too_far_from_semantic_center` remains in
  every candidate record.
- Made the GraspNet table-plane input mask target-specific. Excluding the plane
  is useful for shallow `bowl` proposals but reduced mug candidates from about
  1,000 to 197 and caused association failure; `exclude_table_plane_targets:
  [bowl]` preserves both behaviors explicitly.
- Bowl seed 20 then retained five candidates and succeeded with 0.114774 m lift.
  Artifact: `outputs/20260814_180932_740374_run_seed20/`.
- Current four-class seed-0 regressions all used real YOLO-World and the official
  checkpoint and reached `DONE`: mug 0.098600 m, bottle 0.117293 m, box
  0.115541 m and bowl 0.113518 m. Artifacts:
  `outputs/20260815_001209_554506_run_seed0/`,
  `outputs/20260815_001236_753600_run_seed0/`,
  `outputs/20260815_001300_038326_run_seed0/`, and
  `outputs/20260815_001324_210827_run_seed0/`.
- Added `configs/evaluation_open_vocab_graspnet.yaml` so the headline benchmark
  cannot accidentally inherit oracle or geometric modes.
- Actual command: `evaluate --targets mug,bottle,bowl,box --episodes 10 --config
  configs/evaluation_open_vocab_graspnet.yaml`. It ran exactly 40 episodes.
- Current aggregate: detection 38/40 (95%), target selection 37/40 (92.5%),
  candidate generation 40/40, IK reachable 35/40 (87.5%) and end-to-end success
  **26/40 (65%)**. Per target: bottle 9/10, mug 6/10, box 6/10 and bowl 5/10.
- The 14 failures remain in the denominator: one `detection_failed`, four
  `no_accepted_candidate`, and nine `object_not_lifted`. Mean official inference
  was 5.845 s; mean total time was 11.519 s. Artifact:
  `outputs/20260815_001417_040250_evaluation/`.
- The historical 6/20 benchmark remains preserved as the pre-improvement result;
  it is not substituted or deleted.

## DeepSeek/GraspNet audit consistency - 2026-08-15

- Added explicit `agent.execution_mode` to Agent YAML. The DeepSeek system Prompt
  now requests `open-vocab-graspnet` directly for GraspNet configurations, so
  provider response, validated plan, generated Python DSL and robot backend can
  be audited without a post-response mode mismatch.
- Added regression coverage for Prompt mode binding and config inheritance.
  The current CPU suite is **59 passed in 1.04 s**.
- Re-ran the complete no-cost integration with an explicitly labelled Mock
  planner and real downstream YOLO-World, official GraspNet and Panda physics.
  The generated six-call Python plan passed AST validation; the bottle lifted
  0.117293 m and all modes record `open-vocab-graspnet`. Artifact:
  `outputs/20260815_002539_969124_run_seed0/`.
- This run is not claimed as a hosted DeepSeek call. The Codex process correctly
  has no access to the user's shell-only `DEEPSEEK_API_KEY`; the final real API +
  official GraspNet joint command must be launched from that configured terminal.

Final verification on 2026-08-15:

- Main suite: **59 passed in 1.02 s**; `pip check` reports no broken requirements.
- All 12 YAML configs and both VS Code JSON files parse; all shell launchers pass
  `bash -n`; Python sources/tests pass `compileall`.
- Outside the managed sandbox, `doctor` sees the RTX 3050 Laptop GPU (4,096 MiB),
  driver 580.159.03, checkpoint and compiled extensions. The isolated environment
  re-executed PointNet2 FPS and KNN CUDA kernels with PyTorch 1.13.1/CUDA 11.7.
- Free disk is 24.29 GiB after retaining the 40-episode evidence. Required CSV,
  summary, video and generated-plan artifacts were checked for existence.
- Initialized the repository on branch `main`; `.gitignore` excludes environments,
  weights, third-party source and all generated outputs except `outputs/.gitkeep`.
  The staged tree was checked for large ignored paths and credential patterns.

## Fully combined DeepSeek V4 + YOLO-World + official GraspNet run - 2026-08-15

- The user launched the final command from the project-specific `ovg` Conda
  environment so the shell-only `DEEPSEEK_API_KEY` remained outside source,
  YAML and run artifacts.
- Real `deepseek-v4-flash` returned HTTP 200 and a schema-valid plan for the
  Chinese instruction `请帮我抓取桌面上的瓶子`. The plan explicitly selected
  `open-vocab-graspnet`; the locally generated six-call Python DSL passed AST
  validation and its recorded trace exactly matched the plan.
- In that same audited run, real YOLO-World detected the bottle (IoU 0.913144),
  the official RealSense GraspNet checkpoint generated 1,005 proposals and five
  survived association, geometry, contact, collision, IK and trajectory checks.
- Panda completed `RESET` through `DONE`, lifted the bottle 0.117293 m and ended
  0.018208 m from it. Semantic selection used RGB-D scene geometry rather than
  simulator truth. GraspNet inference took 8.829 s, DeepSeek planning 1.358 s,
  and the total task took 26.308 s.
- The run used 310 API tokens and retained its sanitized request/response,
  validated plan, generated Python, exact trace, detections, candidate records,
  point cloud and video under
  `outputs/20260815_131627_786978_run_seed0/`.
- This single successful demonstration is integration evidence, not a new
  success-rate estimate. The fixed-seed official-GraspNet benchmark remains
  **26/40 (65%)**, with all failures retained.

## Generated Python synchronized robot-stage execution - 2026-08-16

- Replaced the generated-plan recorder-only dispatch with `SafeRobotController`.
  The AST-validated Python now runs in a restricted worker with no builtins; each
  of its six allowlisted calls blocks until the real pipeline begins and completes
  the matching observation, detection, grasp generation, selection, physical
  execution or evaluation stage.
- The robot pipeline cannot advance without the exact one-shot stage capability.
  Missing, reordered, extra or target-changing calls fail closed. The completed
  audit trace records authorization, real stage start and real stage completion
  timestamps instead of only replaying method names.
- Added `generated_plan_execution` to the structured robot result and print the
  generated `.py` path directly in the terminal, so the generated program is
  visible rather than represented only by `generated_python_validated: true`.
- Actual CPU integration run used the explicitly labelled Mock planner with real
  YOLO-World and the geometric baseline. All six generated-code gates completed,
  Panda lifted the mug 0.078137 m and reached `DONE`. Artifact:
  `outputs/20260816_125925_863684_run_seed0/`.
- The Mock result is not a hosted DeepSeek claim and the geometric proposals are
  not GraspNet. Its purpose is to verify the new code-to-real-pipeline execution
  boundary without API cost or CUDA dependence.
- A second actual regression used the same explicitly labelled Mock planner but
  real YOLO-World and the official GraspNet checkpoint. The six synchronized
  generated-code stages completed, Panda lifted the bottle 0.117293 m and reached
  `DONE`. Artifact: `outputs/20260816_130458_513676_run_seed0/`. This validates
  controller/GPU-backend compatibility, not hosted-LLM accuracy.

## Multilingual Agent reliability evaluation infrastructure - 2026-08-16

- Added a fixed 20-case bilingual suite with five paraphrases each for mug,
  bottle, bowl and box. Added `agent-evaluate` and a launcher that record schema
  validity, target accuracy, Python validation, latency, token usage and optional
  full robot execution without making a second planner call.
- Reports contain `cases.csv`, `summary.json`, `summary.md`, sanitized per-case
  audits and an environment snapshot. Planner-only and robot-executed rates are
  separate, and failures remain in the denominator.
- Actual no-cost planner-only infrastructure run with the explicitly labelled
  Mock planner produced 18/20 valid/correct plans. Its fixed alias table rejected
  “方块” and “纸盒”; this limitation was retained. Artifact:
  `outputs/20260816_130402_157039_agent_evaluation/`.
- Actual Mock-planner plus real-YOLO/geometric-Panda evaluation requested four
  robot episodes, executed all four and succeeded in three. Bowl seed 2 failed
  with `no_accepted_candidate`. Artifact:
  `outputs/20260816_130414_805882_agent_evaluation/`.
- Neither run is claimed as DeepSeek or GraspNet reliability. The real 20-case
  DeepSeek run and its four official-GraspNet robot cases require execution from
  the user's terminal where `DEEPSEEK_API_KEY` is set; the Codex process cannot
  access that shell-only credential.
- Regression suite after the synchronized controller and Agent evaluator:
  **63 passed in 3.58 s**. The launcher passes `bash -n`, all 20 suite cases load
  with valid scene targets, Python sources compile, and `git diff --check` passes.

## Real multilingual DeepSeek and bounded full-chain reliability run - 2026-08-16

- The user executed `scripts/run_agent_evaluation.sh --mode deepseek
  --robot-cases-per-target 1 --seed-start 0` from the `ovg` terminal containing
  the shell-only key. All 20 hosted requests returned HTTP 200 from
  `deepseek-v4-flash`; no credential-like value appears in the evaluation or its
  four source-run artifacts.
- Across the fixed 12 Chinese and eight English cases, schema-valid plan rate,
  labelled target accuracy and generated-Python validation were each **20/20
  (100%)**. Mean API planning latency was 0.876628 s and total reported usage was
  6,135 tokens. There were no planning failures.
- Four cases requested full execution, one per class. All four used real
  YOLO-World, `official_graspnet_checkpoint_rs`, `SafeRobotController` stage
  gating and no simulation truth for semantic selection; all four reached
  `DONE`. Lifts: mug seed 0 0.098600 m, bottle seed 1 0.117607 m, bowl seed 2
  0.114215 m and box seed 3 0.116835 m.
- Evaluation artifact: `outputs/20260816_133053_957394_agent_evaluation/`.
  Source runs: `outputs/20260816_133055_044631_run_seed0/`,
  `outputs/20260816_133116_668199_run_seed1/`,
  `outputs/20260816_133134_176548_run_seed2/`, and
  `outputs/20260816_133150_905697_run_seed3/`.
- Statistical boundary: the 4/4 robot result is a bounded integration sample
  with one seed per class, not a replacement for the independently measured
  official-GraspNet reliability of **26/40 (65%)** over ten seeds per class.

## True 40-case Agent benchmark support - 2026-08-16

- Added `--full-episodes-per-target N`. Unlike the earlier 20-plan/four-robot
  protocol, it expands every target to N complete cases, makes one new planner
  request per scene and executes one robot episode after validation.
- Seed pairing matches the existing downstream benchmark exactly. For N=10 and
  seed start 0: mug uses 0-9, bottle 10-19, bowl 20-29 and box 30-39. Each target's
  five fixed bilingual instructions is used twice.
- Added an explicit full-chain rate whose denominator is every requested case.
  Planner exceptions, target mismatches, generated-code validation failures and
  robot runtime exceptions cannot disappear merely because motion did not start.
- Actual no-cost full-mode regression with the labelled Mock planner used one
  episode per target and seeds 0-3. All four plans/code paths executed; the real
  YOLO/geometric Panda backend succeeded 3/4 and retained bowl's
  `no_accepted_candidate`. Artifact:
  `outputs/20260816_135857_061732_agent_evaluation/`.
- Regression suite: **64 passed in 3.58 s**. The true 40-case hosted run remains
  to be launched from the user's key-bearing terminal with
  `--full-episodes-per-target 10`.
