# 求职材料证据审计

审计日期：2026-08-23。范围仅限当前本地工作树、现存正式输出和代码；未调用
DeepSeek API，未重新运行 YOLO-World 或 GraspNet，未重新训练模型。所有百分比都由
下列 CSV 在本轮独立重算，并与对应 JSON 汇总交叉核对。

## Verified from code

- 文本到抓取的主流水线在
  [`src/open_vocab_grasping/pipeline.py`](../src/open_vocab_grasping/pipeline.py)：从
  PyBullet RGB-D、YOLO-World 检测、候选生成、二维框/深度关联、过滤、排序到 Panda
  执行。`grasp.generator: graspnet` 会走隔离的官方 GraspNet 服务；
  `geometric_baseline` 被结果字段显式标记为
  `geometric_scene_baseline_not_graspnet`，不能混称。
- YOLO-World 适配器在
  [`perception/yolo_world.py`](../src/open_vocab_grasping/perception/yolo_world.py)：
  按运行时文本类调用 `set_classes`，以 RGB PIL 图输入；正常检测为空时才可使用
  多提示词共识重试。代码没有将实例分割结果作为 YOLO 输入的静默回退。
- 米制深度、投影和变换可由
  [`perception/depth.py`](../src/open_vocab_grasping/perception/depth.py)、
  [`geometry/projection.py`](../src/open_vocab_grasping/geometry/projection.py) 和
  [`geometry/transforms.py`](../src/open_vocab_grasping/geometry/transforms.py) 验证。
  坐标约定和 GraspNet-to-Panda 固定适配见
  [`coordinate_frames.md`](coordinate_frames.md)。
- 官方 GraspNet 的输入/输出是版本化 NPZ 契约，隔离调用逻辑在
  [`grasping/graspnet_client.py`](../src/open_vocab_grasping/grasping/graspnet_client.py)。
  `configs/graspnet.yaml` 固定了 `ovg-graspnet-cu117`、官方 checkpoint 和
  `graspnet` generator；主环境不直接混入旧版 CUDA 扩展。
- 语义关联、深度一致性和候选拒绝原因分别实现在
  [`grasping/association.py`](../src/open_vocab_grasping/grasping/association.py) 与
  [`planning/filtering.py`](../src/open_vocab_grasping/planning/filtering.py)。过滤包含
  GraspNet 碰撞标记、宽度、桌面、工作空间、接触支持、点云间隙、IK 与插值轨迹。
- Panda 执行状态机是 `RESET → OBSERVE → DETECT → GENERATE_GRASPS → SELECT_GRASP
  → MOVE_PREGRASP → APPROACH → CLOSE_GRIPPER → LIFT → EVALUATE → DONE/FAILED`，实现于
  [`planning/executor.py`](../src/open_vocab_grasping/planning/executor.py)。成功需要
  目标抬升达到配置阈值且末端距离合格，计算见
  [`evaluation/metrics.py`](../src/open_vocab_grasping/evaluation/metrics.py)。
- DeepSeek 只产生结构化高层计划。Schema、目标白名单和固定六步顺序在
  [`agent/schemas.py`](../src/open_vocab_grasping/agent/schemas.py)；本地模板化 DSL、
  AST 精确白名单和空 `__builtins__` 执行在
  [`agent/codegen.py`](../src/open_vocab_grasping/agent/codegen.py)；实际阶段阻塞门控
  在 [`agent/safe_controller.py`](../src/open_vocab_grasping/agent/safe_controller.py)。
  原始模型输出的 Python 从未被执行。
- 固定目标/seed 扩展与全链路分母逻辑在
  [`agent/evaluation.py`](../src/open_vocab_grasping/agent/evaluation.py)。本轮无模型
  回归测试命令为 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ... python -m pytest -q
  -p no:cacheprovider`，结果为 **64 passed in 5.22 s**；测试覆盖列表在
  [`tests/`](../tests)。

## Verified from experiment artifacts

### A. 40 回合 open-vocab-graspnet 基准

原始行级证据是
[`episodes.csv`](../outputs/20260815_001417_040250_evaluation/episodes.csv)，汇总交叉
证据是 [`summary.json`](../outputs/20260815_001417_040250_evaluation/summary.json)。
二者均标记 `actual_run: true`，模式为 `open-vocab-graspnet`，共 40 行：每类 10 个
固定 seed，mug 为 0–9、bottle 为 10–19、bowl 为 20–29、box 为 30–39。

| 指标 | 独立重算 | 原始字段/文件 |
| --- | ---: | --- |
| 检测成功 | **38/40 (95.0%)** | `detection_success` / `episodes.csv` |
| YOLO 目标选择正确 | **37/40 (92.5%)** | `target_selection_correct` / `episodes.csv` |
| 候选生成成功 | **40/40 (100.0%)** | `candidate_generation_success` / `episodes.csv` |
| IK 可达 | **35/40 (87.5%)** | `ik_reachable` / `episodes.csv` |
| 端到端成功 | **26/40 (65.0%)** | `end_to_end_success` / `episodes.csv` |
| 平均官方 GraspNet 推理 | **5.844949 s** | `graspnet_inference_s` / `episodes.csv` |
| 平均单回合总时长 | **11.519495 s** | `total_s` / `episodes.csv` |

| 类别 | 检测 | 目标选择 | IK | 端到端 |
| --- | ---: | ---: | ---: | ---: |
| mug | 10/10 | 10/10 | 10/10 | **6/10 (60.0%)** |
| bottle | 9/10 | 9/10 | 9/10 | **9/10 (90.0%)** |
| bowl | 10/10 | 9/10 | 9/10 | **5/10 (50.0%)** |
| box | 9/10 | 9/10 | 7/10 | **6/10 (60.0%)** |

失败行独立计数为：`detection_failed` 1、`no_accepted_candidate` 4、
`object_not_lifted` 9；总计 14，且所有失败仍在分母中。

### B. 40 次真实 DeepSeek 完整系统基准

原始行级证据是
[`cases.csv`](../outputs/20260816_140225_904008_agent_evaluation/cases.csv)，汇总交叉
证据是 [`summary.json`](../outputs/20260816_140225_904008_agent_evaluation/summary.json)。
所有 40 行的 `provider=deepseek`、`model=deepseek-v4-flash`，并都请求和执行了机器人。

| 指标 | 独立重算 | 原始字段/文件 |
| --- | ---: | --- |
| 成功的 DeepSeek 响应 | **40/40** | `provider` / `cases.csv` |
| Schema 有效计划 | **40/40 (100.0%)** | `plan_valid` / `cases.csv` |
| 目标解析正确 | **40/40 (100.0%)** | `target_correct` / `cases.csv` |
| 本地生成 DSL 的 Python 校验通过 | **40/40 (100.0%)** | `python_valid` / `cases.csv` |
| 实际机器人执行 | **40/40** | `robot_executed` / `cases.csv` |
| Agent 完整系统成功 | **26/40 (65.0%)** | `robot_success` / `cases.csv` |
| 平均规划延迟 | **0.861097 s** | `planning_latency_s` / `cases.csv` |
| 报告 token 总数 | **12,267** | `total_tokens` / `cases.csv` |

按 seed 将这 40 行与上节 `episodes.csv` 对齐后，成功标记和失败原因的差异为
**0/40**。这支持的表述是：在这组固定场景中，结构化 Agent 没有观察到额外失败；
不支持将其泛化为任意语言、任意场景或真实机器人上的因果结论。

### C. 可视化的单个真实完整 episode

目录
[`outputs/20260816_140448_414159_run_seed10`](../outputs/20260816_140448_414159_run_seed10)
包含公开保留的一次成功 bottle episode。其
[`agent_result.json`](../outputs/20260816_140448_414159_run_seed10/agent_result.json)
记录：`provider=deepseek`、`deepseek-v4-flash`、seed 10、真实 YOLO-World、
`official_graspnet_checkpoint_rs`、1 个检测、IoU **0.918234**、1,009 个候选、4 个
最终接受候选、目标抬升 **0.115870 m**、末端距离 **0.026456 m**、总耗时
**11.426946 s**，且 `truth_used_for_semantic_selection=false`。

同一目录的
[`agent_response.json`](../outputs/20260816_140448_414159_run_seed10/agent_response.json)、
[`agent_plan.json`](../outputs/20260816_140448_414159_run_seed10/agent_plan.json)、
[`agent_generated_plan.py`](../outputs/20260816_140448_414159_run_seed10/agent_generated_plan.py)
和 [`agent_generated_plan_trace.json`](../outputs/20260816_140448_414159_run_seed10/agent_generated_plan_trace.json)
共同证明：模型返回 JSON 计划；系统本地编译六个白名单调用；六个真实阶段均完成。

## Not verified / must not claim

- 没有真实 RGB-D 相机、ROS、真实 Franka/Panda、SO-101 或其他物理机械臂部署证据；
  只能写“在 PyBullet 仿真中验证”。
- 没有当前保留的 PLY、深度图、原始 RGB、40 个逐回合视频、失败案例视频或全量候选
  JSON/NPZ。它们曾作为本地忽略输出生成，但已清理，不能在本轮作品集里当作现存素材。
- 不得把 `oracle-perception` 写成开放词汇感知结果；它只是实例分割真值的调试/上界模式。
- 不得把 CPU `geometric_baseline` 写成 GraspNet，或把 `mock`/`deterministic` planner
  写成 DeepSeek。
- 不得声称 DeepSeek 自由生成并执行了任意抓取规划 Python。准确表述是：DeepSeek
  输出受 Schema 约束的 JSON；系统把已验证计划编译为固定六调用 DSL，并以 AST
  白名单和 `SafeRobotController` 逐阶段门控执行。
- 不得将 40/40 的计划正确率写成 100% 抓取成功率；完整物理任务是 26/40 (65.0%)。
- 不得将中文 19/24 与英文 7/16 的机器人成功率解释为语言优劣；语言与固定 seed
  混杂，未做配对因果实验。
- 不得声称训练、微调或从零训练了 YOLO-World/GraspNet，或声称跨真实域泛化、商业
  可部署。官方 GraspNet checkpoint 的许可为非商业研究用途，YOLO-World 相关许可
  义务见 [`third_party_licenses.md`](third_party_licenses.md)。
