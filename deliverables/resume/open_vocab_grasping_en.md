# Open-Vocabulary Perception-Guided Robotic Grasping (PyBullet Simulation)

**Stack:** Python, PyBullet, YOLO-World, GraspNet, Open3D, PyTorch, DeepSeek API

## English CV Description

- Built a text-to-grasp system for a Franka Panda **in PyBullet simulation**: real YOLO-World dynamically localizes text-specified objects, RGB-D is reconstructed into a workspace point cloud, and an isolated official GraspNet service proposes scene-level 6-DoF parallel-jaw grasps before pregrasp, approach, closure, and lift execution. Across 40 fixed-seed episodes over four tabletop classes, the end-to-end task result was **26/40 (65.0%)**.
- Implemented metric OpenGL-depth conversion, projection/back-projection, camera-to-world/base and GraspNet-to-Panda tool transforms; combined 2D-box projection, depth consistency, gripper width, table/point-cloud collision, workspace, contact-support, IK, and interpolated joint-trajectory checks for candidate selection. In a retained real bottle episode, **1,009** official GraspNet candidates were reduced to **4** executable candidates and the object lifted **0.115870 m**.
- Designed paired fixed-seed evaluation and failure attribution: detection was **38/40 (95.0%)**, target selection **37/40 (92.5%)**, and IK reachability **35/40 (87.5%)**, with all 14 failures retained. Added DeepSeek as a high-level structured planner: 40 real API plans, target resolutions, and locally compiled DSL validations were **40/40**, with **0.861 s** mean planning latency; JSON Schema, fixed stage order, AST allowlisting, and `SafeRobotController` gated execution, while full-chain success remained **26/40 (65.0%)**.

## Evidence for Quantified Claims

| Claim | Evidence |
| --- | --- |
| 40 episodes, four classes, 38/40 detection, 37/40 target selection, 35/40 IK, 26/40 end-to-end | [`episodes.csv`](../../outputs/20260815_001417_040250_evaluation/episodes.csv), [`summary.json`](../../outputs/20260815_001417_040250_evaluation/summary.json) |
| 1,009 candidates, four accepted, 0.115870 m lift | [`agent_result.json`](../../outputs/20260816_140448_414159_run_seed10/agent_result.json) |
| 40 DeepSeek cases, 0.861097 s mean latency, 26/40 full-chain | [`cases.csv`](../../outputs/20260816_140225_904008_agent_evaluation/cases.csv), [`summary.json`](../../outputs/20260816_140225_904008_agent_evaluation/summary.json) |

Use only for simulated robotic grasping. Do not represent the work as a physical-robot deployment or as arbitrary LLM-generated-code execution.
