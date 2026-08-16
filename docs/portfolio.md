# Portfolio material

The wording below distinguishes the verified official GraspNet GPU path from
the CPU geometric baseline and reports both successes and failures.

## 中文简历描述

**开放词汇感知驱动的机械臂抓取规划｜Python、PyBullet、YOLO-World、Open3D、PyTorch**

- 从零搭建文本指令到机械臂执行的模块化抓取系统：使用真实 YOLO-World 动态
  文本提示词检测，从 PyBullet RGB-D 反投影三维点云，通过独立 CUDA 11.7 Conda
  服务运行官方 GraspNet 生成 6-DoF 候选，并结合二维框、深度、夹爪截面、官方
  碰撞、工作空间、IK 和插值轨迹筛选后控制 Franka Panda 完成抓取抬升。
- 建立统一的 OpenCV 相机坐标约定和齐次变换链，为深度缓冲米制转换、投影/
  反投影、相机到世界/基座变换编写 CPU 自动测试；候选级日志保留所有拒绝原因，
  实验输出包含 CSV、JSON、失败案例和 GIF/MP4。
- 在 RTX 3050 上编译并验证 PointNet2/KNN CUDA 扩展：官方示例输出 317 个候选，
  PyBullet RGB-D 输出可解析的真实候选；进一步运行 40 个固定种子四类别实验，
  检测成功率 95%、目标选择准确率 92.5%、IK 可达率 87.5%、端到端成功率
  65%（26/40；瓶子 9/10、杯子 6/10、盒子 6/10、碗 5/10），并将 14 次检测、
  候选和接触失败全部保留在统计中。
- 接入真实 DeepSeek `deepseek-v4-flash` API 作为高层任务规划器，将中文指令转换为
  受 JSON Schema、动作顺序和场景目标白名单约束的抓取计划；在 20 条固定中英文
  指令上实现计划有效率、目标解析准确率和 Python 校验率均为 100%，平均规划延迟
  0.877 秒、总计 6,135 tokens。计划被自动编译为受 AST 白名单约束的六调用 Python
  DSL，并通过无 builtins 的 `SafeRobotController` 与真实流水线逐阶段同步；每类一次
  的 DeepSeek + YOLO-World + 官方 GraspNet + Panda 联合实验达到 4/4 成功。该小样本
  与独立 40 次抓取基准的 65% 成功率分开报告。

## English CV description

**Open-Vocabulary Perception-Guided Robotic Grasping | Python, PyBullet,
YOLO-World, Open3D, PyTorch**

- Built a modular text-to-grasp pipeline that runs real YOLO-World open-vocabulary
  detection, reconstructs metric RGB-D point clouds, associates 2D semantics with
  scene-level 6-DoF candidates, filters table/scene collisions and Panda IK paths,
  and executes a pregrasp-approach-close-lift state machine.
- Defined and regression-tested the OpenCV camera convention, OpenGL depth
  conversion, projection/back-projection and camera/world/base transforms; added
  candidate-level rejection traces and reproducible CSV/JSON/video reports.
- Compiled and validated the official PointNet2/KNN CUDA extensions on an RTX
  3050. The official example emitted 317 candidates; an end-to-end seed-0 bottle
  path generated parseable neural candidates from PyBullet RGB-D. Across 40
  fixed-seed, four-class episodes, detection reached 95%, target selection 92.5%,
  IK reachability 87.5%, and end-to-end success 65% (26/40), with every failure retained.
- Integrated the real DeepSeek `deepseek-v4-flash` API as a high-level planner,
  validating Chinese-command plans against a JSON Schema, fixed action sequence,
  and scene-target allowlist before dispatch. Across 20 fixed Chinese/English
  instructions, plan validity, target accuracy and generated-Python validity were
  all 100%, with 0.877-second mean latency and 6,135 total tokens. Compiled each
  plan to an AST-whitelisted six-call Python DSL whose `SafeRobotController`
  synchronizes every real pipeline stage; four bounded DeepSeek + YOLO-World +
  official-GraspNet + Panda trials succeeded 4/4. This small integration sample
  is reported separately from the 26/40 fixed-seed grasp benchmark.

## Two-minute portfolio walkthrough

1. Open the architecture diagram in `README.md` and state the truth boundary.
2. Play `outputs/20260815_131627_786978_run_seed0/demo.mp4` (real DeepSeek,
   YOLO-World and official GraspNet bottle grasp).
3. Compare `detections.png`, `filtered_candidates_2d.png` and
   `candidates.json` to show perception, association and rejection traces.
4. Open the 40-row GraspNet `episodes.csv`, then explain 26 successes and the
   14 retained failures using `docs/failure_analysis.md`.
5. Show `agent_generated_plan.py`, its synchronized stage trace, and the 20-case
   DeepSeek `cases.csv`; finish by distinguishing 4/4 bounded integration from
   the 26/40 fixed-seed grasp benchmark.
