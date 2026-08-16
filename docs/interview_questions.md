# Technical interview questions and implementation-based answers

## 1. How did you prevent simulator ground truth from leaking into perception?

The open-vocabulary path generates candidates from the complete cropped scene
point cloud and associates them with a real YOLO-World box plus measured depth.
Instance IDs are unavailable to that selection code. They are read afterward
for IoU/target-error metrics, by PyBullet for physical collision queries, and by
the final success evaluator. A separate mode named `oracle-perception` replaces
only semantic selection so its result is clearly an upper-bound baseline.

## 2. How did you verify the camera coordinate transformations?

The project uses OpenCV optical coordinates: x right, y down, z forward. A
transform named `T_destination_source` maps homogeneous column vectors from the
source into the destination. Tests cover OpenGL buffer-to-metric depth,
3D-to-pixel-to-3D round trips, camera/world/base composition, table height and
object-cloud center error. The camera basis is constructed analytically, so no
unexplained axis swap or sign flip is present.

## 3. Why use a 2D box and depth check instead of cropping every point inside the box?

A 2D box contains background, table and sometimes another object. Each candidate
center is projected into the box, but it is accepted only when its camera z is
consistent with the robust median depth in a shrunken box region. Scene-wide
clustering and non-target point-cloud clearance add geometric evidence. The
remaining weakness is that a box is not an instance mask; a segmentation-capable
open-vocabulary model is a logical next experiment.

## 4. What made the experiment reproducible and diagnosable?

Every run resolves YAML configuration and a fixed seed, records raw detector
predictions and candidate rejection reasons, and writes timing and success fields
to a single episode row. Batch reports are computed from `episodes.csv`. The
18-episode comparison reuses identical target/seed pairs in oracle and real-YOLO
modes. Failures remain in the denominator and are copied to dedicated folders.

## 5. Why is GraspNet separated, and what is actually verified?

The official baseline targets old PyTorch/CUDA versions and custom PointNet2/KNN
extensions. Keeping it in `ovg-graspnet-cu117` avoids destabilizing the modern
PyBullet/YOLO environment. On the RTX 3050 I compiled both CUDA extensions for
`sm_86`, ran the official example and checkpoint, then ran the same service on
PyBullet RGB-D through a versioned NPZ boundary. The current four-class benchmark
ran 40 fixed seeds and reached 38/40 detections, 35/40 IK-reachable episodes and
26/40 end-to-end successes. Nine executed grasps still failed to lift, exposing a
real contact/domain gap rather than substituting the geometric baseline.

## 6. Did the LLM really generate and execute robot-planning Python?

DeepSeek produces a schema-constrained semantic plan rather than arbitrary code.
The system compiles that validated plan into a six-call Python DSL, rejects any
AST containing imports, builtins, extra calls, reordered stages or a changed
target, and executes it with no builtins against `SafeRobotController`. Each call
blocks until the matching real perception/planning/execution stage completes, so
the Python is an auditable stage program rather than a decorative file. In a
fixed 20-instruction bilingual evaluation, all plans, targets and programs were
valid; four bounded real-YOLO/official-GraspNet robot trials succeeded. I keep
that 4/4 sample separate from the broader 26/40 downstream grasp benchmark.
