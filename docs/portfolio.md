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
  受 JSON Schema、动作顺序和场景目标白名单约束的抓取计划；真实端到端实验使用
  304 tokens、0.965 秒规划后驱动现有感知与控制闭环抓取杯子成功，并保存脱敏的
  请求、响应、计划和执行审计记录。计划进一步编译为受 AST 白名单约束的 Python
  DSL，仅允许六个控制器动作，在无 builtins 记录器上验证后才调度机器人。

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
  and scene-target allowlist before dispatch. A verified 304-token, 0.965-second
  planning call drove the existing perception/control pipeline to a successful
  mug baseline grasp with redacted audit artifacts. Compiled each validated plan
  to an AST-whitelisted six-call Python DSL before dispatch; raw model Python is
  never executed.

## Two-minute portfolio walkthrough

1. Open the architecture diagram in `README.md` and state the truth boundary.
2. Play `outputs/20260814_153729_431593_run_seed0/demo.mp4` (real GraspNet bottle).
3. Compare `detections.png`, `filtered_candidates_2d.png` and
   `candidates.json` to show perception, association and rejection traces.
4. Open the 40-row GraspNet `episodes.csv`, then explain 26 successes and the
   14 retained failures using `docs/failure_analysis.md`.
5. Show `agent_generated_plan.py` and its AST trace, then finish with the concrete
   RGB-D/SO-ARM101 calibration and controller migration plan.
