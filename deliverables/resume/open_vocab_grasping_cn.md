# 开放词汇感知驱动的机械臂抓取规划（PyBullet 仿真）

**技术栈：** Python、PyBullet、YOLO-World、GraspNet、Open3D、PyTorch、DeepSeek API

## 中文简历项目描述（3 条）

- 在 **PyBullet 仿真**中搭建中文/英文文本指令到 Franka Panda 抓取的端到端系统：使用 YOLO-World 动态文本类别定位目标，从固定 RGB-D 相机重建工作空间点云，通过隔离的官方 GraspNet 生成场景级 6-DoF 平行夹爪候选，并执行预抓取、接近、闭合和抬升；在 4 类桌面物体、40 个固定 seed 的正式实验中，端到端完成 **26/40（65.0%）** 任务。
- 建立 OpenGL 深度缓冲到米制深度、像素投影/反投影、相机—世界—机器人基座及 GraspNet—Panda 工具坐标变换；将二维检测框与候选中心投影、区域深度一致性、夹爪宽度、桌面/点云碰撞、工作空间、接触支持、IK 与关节插值轨迹串联为候选筛选链路。保留的真实 bottle episode 从 **1,009** 个官方 GraspNet 候选中筛至 **4** 个可执行候选，最终抬升 **0.115870 m**。
- 设计按类别配对的固定 seed 评估与失败归因流程，正式结果为检测 **38/40（95.0%）**、目标选择 **37/40（92.5%）**、IK 可达 **35/40（87.5%）**；14 次失败均保留在统计中。额外接入 DeepSeek 作为高层结构化任务规划器：40 次真实 API 计划、目标解析和本地 DSL 校验均为 **40/40**，平均规划延迟 **0.861 s**；计划经 JSON Schema、固定动作顺序和 AST 白名单验证后，由 `SafeRobotController` 分阶段门控执行，完整系统成功率仍如实报告为 **26/40（65.0%）**。

## 数字证据索引

| 简历数字 | 原始证据 |
| --- | --- |
| 4 类、40 个固定 seed；38/40 检测；37/40 目标选择；35/40 IK；26/40 端到端；14 次失败 | [`outputs/20260815_001417_040250_evaluation/episodes.csv`](../../outputs/20260815_001417_040250_evaluation/episodes.csv)；[`summary.json`](../../outputs/20260815_001417_040250_evaluation/summary.json)；核对说明见 [`docs/job_material_evidence.md`](../../docs/job_material_evidence.md) |
| 1,009 候选、4 个接受候选、0.115870 m 抬升 | [`agent_result.json`](../../outputs/20260816_140448_414159_run_seed10/agent_result.json) |
| 40/40 DeepSeek 真实响应、计划/目标/DSL 校验；0.861097 s；12,267 tokens；26/40 完整系统 | [`outputs/20260816_140225_904008_agent_evaluation/cases.csv`](../../outputs/20260816_140225_904008_agent_evaluation/cases.csv)；[`summary.json`](../../outputs/20260816_140225_904008_agent_evaluation/summary.json) |

## 使用边界

- 写“PyBullet 仿真中验证”，不要写成真实机械臂、ROS 或真实相机部署。
- 写“官方 GraspNet”，仅指 `open-vocab-graspnet` 正式实验；不要混入 oracle、Mock 或 CPU geometric baseline 数字。
- 写“DeepSeek 结构化计划经本地安全验证后执行”，不要写成“DeepSeek 任意生成并执行 Python”。
