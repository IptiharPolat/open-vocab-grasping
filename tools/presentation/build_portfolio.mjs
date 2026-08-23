import fs from 'node:fs';
import path from 'node:path';
import PptxGenJS from 'pptxgenjs';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', '..');
const OUT = path.join(ROOT, 'deliverables', 'portfolio');
const REL = (...parts) => path.join(ROOT, ...parts);
const img = (...parts) => REL(...parts);

const C = {
  navy: '071522', ink: '0D1B2A', panel: '102535', slate: '2B4252', muted: 'AFC2CD',
  white: 'F3F7F8', cyan: '39D5C5', teal: '159E99', blue: '4CA8FF', gold: 'F5C256',
  red: 'F06F6F', green: '66D49E', grid: '284151', pale: 'DDEAF0', black: '000000',
};
const FONT = 'Noto Sans CJK SC';
const W = 13.333;
const H = 7.5;

function addBg(slide, dark = true) {
  slide.background = { color: dark ? C.navy : 'F5F8FA' };
  if (dark) {
    slide.addShape(PPT.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.08, line: { color: C.cyan }, fill: { color: C.cyan } });
    slide.addShape(PPT.ShapeType.arc, { x: 10.7, y: 5.55, w: 2.6, h: 2.6, adjustPoint: 0.25, rotate: 20, line: { color: C.slate, transparency: 25, width: 1 }, fill: { color: C.navy, transparency: 100 } });
  }
}

function addTitle(slide, title, kicker, dark = true) {
  slide.addText(kicker.toUpperCase(), { x: 0.55, y: 0.28, w: 3.5, h: 0.26, fontFace: FONT, fontSize: 10, bold: true, color: dark ? C.cyan : C.teal, charSpacing: 1.3, margin: 0 });
  slide.addText(title, { x: 0.55, y: 0.59, w: 12.1, h: 0.55, fontFace: FONT, fontSize: 36, bold: true, color: dark ? C.white : C.ink, margin: 0, breakLine: false, fit: 'shrink' });
}

function addFooter(slide, page, text, dark = true) {
  slide.addShape(PPT.ShapeType.line, { x: 0.55, y: 7.08, w: 12.2, h: 0, line: { color: dark ? C.slate : 'C6D4DB', width: 0.5 } });
  slide.addText(`OPEN-VOCAB GRASPING  |  PYBULLET SIMULATION`, { x: 0.55, y: 7.16, w: 4.8, h: 0.16, fontFace: FONT, fontSize: 7.5, color: dark ? C.muted : '607684', margin: 0, charSpacing: 0.4 });
  slide.addText(text, { x: 5.15, y: 7.16, w: 6.8, h: 0.16, fontFace: FONT, fontSize: 7.2, color: dark ? C.muted : '607684', align: 'right', margin: 0, fit: 'shrink' });
  slide.addText(String(page).padStart(2, '0'), { x: 12.1, y: 7.1, w: 0.55, h: 0.22, fontFace: 'Arial', fontSize: 9, bold: true, color: C.cyan, align: 'right', margin: 0 });
}

function note(slide, body, sources) {
  slide.addNotes(`${body}\n\n[Sources]\n${sources.map((item) => `- ${item}`).join('\n')}`);
}

function rect(slide, x, y, w, h, fill, radius = true, line = C.slate) {
  slide.addShape(radius ? PPT.ShapeType.roundRect : PPT.ShapeType.rect, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: line, width: 0.8 } });
}

function label(slide, text, x, y, w, color = C.cyan) {
  slide.addText(text, { x, y, w, h: 0.26, fontFace: FONT, fontSize: 10, bold: true, color, margin: 0, fit: 'shrink' });
}

function text(slide, value, x, y, w, h, opts = {}) {
  slide.addText(value, { x, y, w, h, fontFace: FONT, fontSize: opts.fontSize ?? 17, bold: opts.bold ?? false, color: opts.color ?? C.white, margin: opts.margin ?? 0, breakLine: false, valign: opts.valign ?? 'mid', align: opts.align ?? 'left', fit: opts.fit ?? 'shrink', bullet: opts.bullet });
}

function image(slide, file, x, y, w, h, caption) {
  slide.addImage({ path: file, x, y, w, h });
  if (caption) {
    slide.addShape(PPT.ShapeType.rect, { x, y: y + h - 0.34, w, h: 0.34, fill: { color: C.black, transparency: 20 }, line: { color: C.black, transparency: 100 } });
    text(slide, caption, x + 0.12, y + h - 0.29, w - 0.24, 0.18, { fontSize: 9, color: C.white, bold: true });
  }
}

function sourceBadge(slide, value, x, y, w) {
  rect(slide, x, y, w, 0.34, C.panel, true, C.teal);
  text(slide, value, x + 0.12, y + 0.06, w - 0.24, 0.2, { fontSize: 9.5, color: C.cyan, bold: true });
}

function csvRows(file) {
  const [header, ...rows] = fs.readFileSync(file, 'utf8').trim().split(/\r?\n/);
  const fields = header.split(',');
  return rows.map((row) => Object.fromEntries(row.split(',').map((value, index) => [fields[index], value])));
}

const episodes = csvRows(REL('outputs/20260815_001417_040250_evaluation/episodes.csv'));
const agent = csvRows(REL('outputs/20260816_140225_904008_agent_evaluation/cases.csv'));
const truth = (value) => value === 'True';
const count = (rows, key) => rows.filter((row) => truth(row[key])).length;
const expected = { detection: 38, selection: 37, ik: 35, success: 26, agentPlans: 40 };
if (episodes.length !== 40 || count(episodes, 'detection_success') !== expected.detection || count(episodes, 'target_selection_correct') !== expected.selection || count(episodes, 'ik_reachable') !== expected.ik || count(episodes, 'end_to_end_success') !== expected.success || agent.length !== 40 || count(agent, 'plan_valid') !== expected.agentPlans) {
  throw new Error('Formal CSV validation failed; portfolio values must not be built from inconsistent evidence.');
}

const PPT = new PptxGenJS();
PPT.layout = 'LAYOUT_WIDE';
PPT.author = 'Iptihar Polat';
PPT.company = 'Open-Vocabulary Perception-Guided Robotic Grasping';
PPT.subject = 'Robotics research portfolio; evidence-linked PyBullet simulation';
PPT.title = '开放词汇感知驱动的机械臂抓取规划';
PPT.lang = 'zh-CN';
PPT.theme = { headFontFace: FONT, bodyFontFace: FONT, lang: 'zh-CN' };

// 01 Cover
{
  const slide = PPT.addSlide(); addBg(slide, true);
  text(slide, '具身智能 · 机器人算法 · 抓取规划作品集', 0.62, 0.62, 5.8, 0.3, { fontSize: 13, color: C.cyan, bold: true });
  text(slide, '开放词汇感知驱动的\n机械臂抓取规划', 0.62, 1.05, 6.35, 1.72, { fontSize: 50, color: C.white, bold: true, valign: 'top' });
  text(slide, '文本指令 → 语义感知 → 6-DoF 抓取 → Panda 执行', 0.65, 2.93, 6.1, 0.35, { fontSize: 18, color: C.pale });
  sourceBadge(slide, 'PYBULLET SIMULATION', 0.65, 3.50, 2.15);
  text(slide, 'Python · PyBullet · YOLO-World · GraspNet · Open3D · PyTorch', 0.65, 4.02, 6.25, 0.25, { fontSize: 12, color: C.muted });
  text(slide, 'github.com/IptiharPolat/open-vocab-grasping', 0.65, 4.38, 5.8, 0.22, { fontSize: 11.5, color: C.cyan });
  image(slide, img('deliverables/portfolio/assets/seed10_lift.png'), 7.43, 0.64, 5.2, 3.9, '真实保留 seed-10：DeepSeek + YOLO-World + 官方 GraspNet 成功抬升');
  rect(slide, 7.43, 4.83, 5.2, 1.05, C.panel);
  text(slide, '一次真实成功 episode', 7.75, 5.07, 1.85, 0.22, { fontSize: 12, color: C.cyan, bold: true });
  text(slide, 'bottle  |  1,009 → 4 候选  |  抬升 0.115870 m', 7.75, 5.40, 4.35, 0.25, { fontSize: 16, color: C.white, bold: true });
  addFooter(slide, 1, '真实项目素材：seed-10 bottle demo.gif / agent_result.json');
  note(slide, '开场：这是一个在 PyBullet 仿真中验证的 text-to-grasp 系统。先展示最终闭环，再解释每层如何避免把仿真真值当作感知结果。', [
    'outputs/20260816_140448_414159_run_seed10/demo.gif',
    'outputs/20260816_140448_414159_run_seed10/agent_result.json',
    'https://github.com/IptiharPolat/open-vocab-grasping',
  ]);
}

// 02 Problem
{
  const slide = PPT.addSlide(); addBg(slide, false); addTitle(slide, '从“说出目标”到“安全抓起目标”', 'Problem & scope', false);
  rect(slide, 0.65, 1.38, 3.7, 4.95, 'FFFFFF', true, 'C8D8DF');
  label(slide, '固定类别感知的局限', 0.95, 1.75, 2.5, C.teal);
  text(slide, '传统抓取系统常把\n类别、目标选择和几何规划\n写死在任务中。', 0.95, 2.15, 2.85, 1.05, { fontSize: 22, color: C.ink, bold: true, valign: 'top' });
  text(slide, '面对“抓取红色瓶子 / pick the mug”这类自然语言目标时，需要把二维语义与三维可执行抓取对应起来。', 0.95, 3.62, 2.9, 1.35, { fontSize: 16, color: '36505E', valign: 'top' });
  rect(slide, 4.72, 1.38, 7.95, 4.95, C.navy, true, C.navy);
  label(slide, '本项目的输入 / 输出', 5.12, 1.75, 2.7, C.cyan);
  rect(slide, 5.12, 2.30, 2.35, 0.83, C.panel, true, C.teal); text(slide, '“请抓取桌面上的瓶子”', 5.35, 2.51, 1.86, 0.28, { fontSize: 14, color: C.white, bold: true, align: 'center' });
  slide.addShape(PPT.ShapeType.chevron, { x: 7.75, y: 2.49, w: 0.55, h: 0.38, fill: { color: C.cyan }, line: { color: C.cyan } });
  rect(slide, 8.55, 2.30, 3.50, 0.83, '123A4B', true, C.cyan); text(slide, 'Panda 预抓取 → 接近 → 闭合 → 抬升', 8.80, 2.51, 3.0, 0.28, { fontSize: 14, color: C.white, bold: true, align: 'center' });
  const bullets = ['中文/英文文本指定目标', '开放词汇检测与 RGB-D 三维关联', '6-DoF 候选、碰撞与 IK 约束', '自动成功判定与失败归因'];
  bullets.forEach((b, i) => { slide.addShape(PPT.ShapeType.ellipse, { x: 5.18, y: 3.65 + i * 0.55, w: 0.16, h: 0.16, fill: { color: C.cyan }, line: { color: C.cyan } }); text(slide, b, 5.48, 3.58 + i * 0.55, 5.6, 0.3, { fontSize: 17, color: C.white }); });
  text(slide, '验证范围：固定相机 + Franka Panda + 桌面场景的 PyBullet 仿真；不代表真机、ROS 或真实相机性能。', 5.12, 5.82, 6.85, 0.28, { fontSize: 11.5, color: C.gold, bold: true });
  addFooter(slide, 2, '范围声明：PyBullet 仿真，不是物理机器人部署', false);
  note(slide, '问题页：强调工程难点不是“让机械臂动起来”，而是把文本语义、RGB-D 几何和可达运动统一到一个可审计闭环中。', ['README.md', 'docs/architecture.md', 'docs/experiment_protocol.md']);
}

// 03 Architecture
{
  const slide = PPT.addSlide(); addBg(slide, true); addTitle(slide, '完整闭环：语义决定“抓谁”，几何决定“怎么抓”', 'System architecture', true);
  const nodes = [
    ['中文 / 英文\n任务指令', 0.65, '0E4051'], ['DeepSeek\n结构化计划', 2.45, '0E4051'], ['YOLO-World\n开放词汇检测', 4.25, '123A4B'], ['RGB-D\n点云重建', 6.05, '123A4B'], ['官方 GraspNet\n6-DoF 候选', 7.85, '173A4B'], ['语义+安全\n筛选', 9.65, '173A4B'], ['Panda\n执行', 11.45, '0E4051'],
  ];
  nodes.forEach(([value, x, color], i) => { rect(slide, x, 2.27, 1.40, 0.95, color, true, C.teal); text(slide, value, x + 0.10, 2.48, 1.2, 0.42, { fontSize: 13, color: C.white, bold: true, align: 'center' }); if (i < nodes.length - 1) slide.addShape(PPT.ShapeType.chevron, { x: x + 1.47, y: 2.57, w: 0.25, h: 0.30, fill: { color: C.cyan }, line: { color: C.cyan } }); });
  rect(slide, 1.30, 3.96, 10.72, 1.34, C.panel, true, C.slate);
  text(slide, '筛选链路', 1.60, 4.23, 1.15, 0.24, { fontSize: 14, color: C.cyan, bold: true });
  const filters = ['检测框投影', '深度一致性', '官方碰撞标记', '工作空间/夹爪宽度', 'Panda IK', '轨迹插值'];
  filters.forEach((value, i) => { const x = 2.75 + i * 1.42; slide.addShape(PPT.ShapeType.roundRect, { x, y: 4.18, w: 1.18, h: 0.48, rectRadius: 0.06, fill: { color: i % 2 ? '164252' : '173544' }, line: { color: C.teal, width: 0.5 } }); text(slide, value, x + 0.05, 4.30, 1.08, 0.17, { fontSize: 9.5, color: C.white, bold: true, align: 'center' }); });
  text(slide, '真值边界：PyBullet instance ID 只用于 oracle 上界、事后指标与物理/成功判定；开放词汇语义选择不读取实例 ID。', 1.30, 5.73, 10.7, 0.30, { fontSize: 14, color: C.gold, bold: true, align: 'center' });
  addFooter(slide, 3, '架构源：docs/architecture.md；LLM 安全源：docs/agent_architecture.md');
  note(slide, '这一页强调职责分工：DeepSeek 只规划受约束阶段；YOLO-World 给出语义证据；GraspNet 提供候选；最后由明确物理和运动学规则筛选并执行。', ['docs/architecture.md', 'docs/agent_architecture.md', 'docs/coordinate_frames.md', 'Ultralytics YOLO-World documentation', 'graspnet/graspnet-baseline official repository']);
}

// 04 Perception
{
  const slide = PPT.addSlide(); addBg(slide, false); addTitle(slide, '开放词汇检测与三维关联：不以实例 ID 替代感知', 'Open-vocabulary perception', false);
  image(slide, img('outputs/20260816_140448_414159_run_seed10/detections.png'), 0.62, 1.42, 4.0, 3.0, '真实 YOLO-World：bottle 检测框');
  image(slide, img('outputs/20260816_140448_414159_run_seed10/filtered_candidates_2d.png'), 4.85, 1.42, 4.0, 3.0, '候选中心投影到目标框并检查深度');
  image(slide, img('outputs/20260816_140448_414159_run_seed10/filtered_candidates_3d.png'), 9.08, 1.42, 3.0, 3.0, '场景级候选的 3D 过滤可视化');
  const callouts = [
    ['1', '动态文本类别：由指令给定目标词'], ['2', 'OpenGL 深度缓冲 → 米制 z，反投影为点云'], ['3', '候选中心需同时满足框内、深度一致与几何间隙'],
  ];
  callouts.forEach(([n, value], i) => { const x = 0.72 + i * 4.20; slide.addShape(PPT.ShapeType.ellipse, { x, y: 5.16, w: 0.35, h: 0.35, fill: { color: C.teal }, line: { color: C.teal } }); text(slide, n, x, 5.22, 0.35, 0.13, { fontSize: 10, color: C.white, bold: true, align: 'center' }); text(slide, value, x + 0.48, 5.07, 3.36, 0.47, { fontSize: 14, color: C.ink, bold: true, valign: 'top' }); });
  text(slide, '实例真值仅在推理后计算 IoU 和目标选择指标；没有作为 YOLO-World 输入或候选语义选择输入。', 0.72, 6.18, 11.75, 0.26, { fontSize: 13, color: C.teal, bold: true, align: 'center' });
  addFooter(slide, 4, '真实 seed-10 图像：detections.png / filtered_candidates_2d.png / filtered_candidates_3d.png', false);
  note(slide, '展示真实单次完整 episode 的三个阶段。强调检测框本身不足以识别抓取归属，因此将 3D 抓取中心投影回图像并加上深度一致性。', ['outputs/20260816_140448_414159_run_seed10/detections.png', 'outputs/20260816_140448_414159_run_seed10/filtered_candidates_2d.png', 'outputs/20260816_140448_414159_run_seed10/filtered_candidates_3d.png', 'docs/coordinate_frames.md', 'docs/experiment_protocol.md']);
}

// 05 Grasping
{
  const slide = PPT.addSlide(); addBg(slide, true); addTitle(slide, '6-DoF 候选不是直接执行：必须经过物理与运动学筛选', 'Grasp planning & execution', true);
  image(slide, img('outputs/20260816_140448_414159_run_seed10/filtered_candidates_3d.png'), 0.68, 1.43, 3.35, 3.35, '真实官方 GraspNet 候选筛选可视化');
  rect(slide, 4.38, 1.43, 3.64, 3.35, C.panel, true, C.slate);
  label(slide, '坐标链与筛选', 4.72, 1.75, 2.0, C.cyan);
  text(slide, 'T_base_grasp =\nT_base_world · T_world_camera ·\nT_camera_grasp · T_grasp_tool', 4.72, 2.18, 2.90, 0.92, { fontSize: 17, color: C.white, bold: true, valign: 'top' });
  ['抓取分数 / 宽度', '桌面与点云碰撞', '工作空间 / 接触支持', '预抓取 + 抓取 IK', '关节插值轨迹'].forEach((v, i) => { slide.addShape(PPT.ShapeType.ellipse, { x: 4.75, y: 3.42 + i * 0.25, w: 0.10, h: 0.10, fill: { color: C.cyan }, line: { color: C.cyan } }); text(slide, v, 4.98, 3.35 + i * 0.25, 2.55, 0.19, { fontSize: 10.5, color: C.pale }); });
  image(slide, img('deliverables/portfolio/assets/seed10_approach.png'), 8.35, 1.43, 2.0, 1.50, '预抓取 / 接近');
  image(slide, img('deliverables/portfolio/assets/seed10_lift.png'), 10.55, 1.43, 2.0, 1.50, '闭合 / 抬升');
  const states = ['RESET', 'OBSERVE', 'DETECT', 'GENERATE', 'SELECT', 'PREGRASP', 'APPROACH', 'CLOSE', 'LIFT', 'EVALUATE'];
  states.forEach((value, i) => { const x = 8.38 + (i % 5) * 0.83; const y = 3.42 + Math.floor(i / 5) * 0.50; rect(slide, x, y, 0.71, 0.30, i >= 5 ? '164252' : '113344', true, C.teal); text(slide, value, x + 0.03, y + 0.09, 0.65, 0.12, { fontSize: 7.2, color: C.white, bold: true, align: 'center' }); });
  rect(slide, 0.70, 5.34, 11.88, 0.87, '123A4B', true, C.teal);
  text(slide, '保留的真实 bottle 结果：官方 GraspNet 1,009 个候选 → 4 个可执行候选 → 抬升 0.115870 m；整个过程由 Panda 状态机执行。', 1.05, 5.63, 11.15, 0.25, { fontSize: 17, color: C.white, bold: true, align: 'center' });
  addFooter(slide, 5, '真实 seed-10：agent_result.json；坐标定义：docs/coordinate_frames.md');
  note(slide, '抓取候选是神经网络的输出，但不能直接给机器人。这里展示从相机抓取位姿到 Panda TCP 的变换，以及碰撞、IK、轨迹等可解释安全门。', ['outputs/20260816_140448_414159_run_seed10/filtered_candidates_3d.png', 'outputs/20260816_140448_414159_run_seed10/agent_result.json', 'docs/coordinate_frames.md', 'docs/architecture.md']);
}

// 06 DeepSeek
{
  const slide = PPT.addSlide(); addBg(slide, false); addTitle(slide, 'DeepSeek 只负责受约束任务规划，不自由执行代码', 'Safe LLM planning', false);
  const x = [0.62, 3.20, 5.76, 8.32, 10.88];
  const boxes = [
    ['中文/英文指令', '请抓取桌面上的瓶子'], ['JSON 计划', 'action=pick\ntarget=bottle'], ['规则校验', 'Schema\n目标白名单\n六步顺序'], ['受限 DSL', "controller.detect\n('bottle')\n… 共六调用"], ['安全执行', 'SafeRobot\nController\n逐阶段门控'],
  ];
  boxes.forEach(([title, body], i) => { rect(slide, x[i], 1.82, 1.90, 2.85, i === 2 ? '154550' : 'FFFFFF', true, i === 2 ? C.teal : 'C9D8DE'); text(slide, title, x[i] + 0.15, 2.14, 1.60, 0.24, { fontSize: 15, color: i === 2 ? C.cyan : C.ink, bold: true, align: 'center' }); text(slide, body, x[i] + 0.15, 2.75, 1.60, 0.95, { fontSize: i === 3 ? 10.5 : 13, color: i === 2 ? C.white : '36505E', bold: i === 1, align: 'center', valign: 'top' }); if (i < boxes.length - 1) slide.addShape(PPT.ShapeType.chevron, { x: x[i] + 2.08, y: 3.05, w: 0.28, h: 0.32, fill: { color: C.teal }, line: { color: C.teal } }); });
  rect(slide, 0.85, 5.26, 11.65, 0.77, 'E5F5F4', true, 'B4DCD7');
  text(slide, '40 次真实 DeepSeek API 规划：计划有效 / 目标正确 / 本地 DSL 校验均为 40/40；平均规划延迟 0.861 s。', 1.08, 5.52, 11.15, 0.25, { fontSize: 17, color: C.ink, bold: true, align: 'center' });
  text(slide, '关键边界：模型返回 JSON；系统本地编译并 AST 验证固定六调用 DSL；导入、builtins、额外调用、顺序变化和目标变化均拒绝。', 0.90, 6.35, 11.55, 0.26, { fontSize: 12.5, color: C.teal, bold: true, align: 'center' });
  addFooter(slide, 6, '真实 Agent 40 回合：cases.csv / summary.json；单例：agent_plan.json / trace.json', false);
  note(slide, '不要说“DeepSeek 生成并执行任意 Python”。正确描述是：模型只产生结构化计划；系统编译为固定 DSL 并做 AST 白名单检查；控制器逐阶段同步真实流水线。', ['outputs/20260816_140225_904008_agent_evaluation/cases.csv', 'outputs/20260816_140225_904008_agent_evaluation/summary.json', 'outputs/20260816_140448_414159_run_seed10/agent_plan.json', 'outputs/20260816_140448_414159_run_seed10/agent_generated_plan.py', 'outputs/20260816_140448_414159_run_seed10/agent_generated_plan_trace.json', 'docs/agent_architecture.md']);
}

// 07 Results
{
  const slide = PPT.addSlide(); addBg(slide, true); addTitle(slide, '正式 40 回合实验：语义链路稳定，物理接触仍是主要瓶颈', 'Formal evaluation', true);
  label(slide, '阶段通过率（40 个固定 seed，四类目标）', 0.66, 1.38, 4.5, C.cyan);
  const stages = [['检测', 38, C.blue], ['目标选择', 37, C.cyan], ['IK 可达', 35, C.gold], ['端到端', 26, C.green]];
  stages.forEach(([name, value, color], i) => { const y = 1.89 + i * 0.66; text(slide, name, 0.68, y + 0.11, 1.10, 0.2, { fontSize: 14, color: C.white, bold: true }); slide.addShape(PPT.ShapeType.rect, { x: 1.83, y, w: 3.55, h: 0.42, fill: { color: C.slate }, line: { color: C.slate } }); slide.addShape(PPT.ShapeType.rect, { x: 1.83, y, w: 3.55 * value / 40, h: 0.42, fill: { color }, line: { color } }); text(slide, `${value}/40  (${(value / 40 * 100).toFixed(1)}%)`, 5.53, y + 0.08, 1.35, 0.22, { fontSize: 14, color: C.white, bold: true }); });
  label(slide, '按类别端到端成功率（每类 10 回合）', 7.06, 1.38, 4.6, C.cyan);
  const byTarget = [['mug', 6, C.blue], ['bottle', 9, C.green], ['bowl', 5, C.gold], ['box', 6, C.cyan]];
  byTarget.forEach(([name, value, color], i) => { const x = 7.33 + i * 1.23; const barH = 2.50 * value / 10; slide.addShape(PPT.ShapeType.rect, { x, y: 4.46 - barH, w: 0.70, h: barH, fill: { color }, line: { color } }); text(slide, `${value}/10`, x - 0.08, 4.03 - barH, 0.86, 0.20, { fontSize: 12, color: C.white, bold: true, align: 'center' }); text(slide, name, x - 0.16, 4.62, 1.0, 0.21, { fontSize: 12, color: C.pale, bold: true, align: 'center' }); });
  rect(slide, 0.66, 5.46, 11.95, 0.72, C.panel, true, C.teal);
  text(slide, '官方 GraspNet 平均推理 5.845 s  ·  单回合平均 11.519 s  ·  DeepSeek Agent 完整系统：40/40 计划校验、26/40 物理任务成功', 0.95, 5.71, 11.35, 0.20, { fontSize: 14, color: C.white, bold: true, align: 'center' });
  addFooter(slide, 7, '数据：outputs/.../episodes.csv、summary.json、Agent cases.csv、summary.json');
  note(slide, '这是正式固定种子实验，所有失败均计入分母。端到端 26/40 不能被 40/40 的计划层指标掩盖；瓶子最好，碗最低。', ['outputs/20260815_001417_040250_evaluation/episodes.csv', 'outputs/20260815_001417_040250_evaluation/summary.json', 'outputs/20260816_140225_904008_agent_evaluation/cases.csv', 'outputs/20260816_140225_904008_agent_evaluation/summary.json', 'docs/experiment_protocol.md']);
}

// 08 Failure / next
{
  const slide = PPT.addSlide(); addBg(slide, false); addTitle(slide, '把 14 次失败保留在统计中：下一步从“能抓”走向“稳定抓”', 'Failure analysis & next', false);
  label(slide, '失败分布（40 回合）', 0.72, 1.42, 2.6, C.teal);
  const failures = [['检测失败', 1, C.red], ['无可接受候选', 4, C.gold], ['未稳定抬升', 9, C.red]];
  failures.forEach(([name, value, color], i) => { const y = 1.96 + i * 0.73; text(slide, name, 0.75, y + 0.09, 1.75, 0.22, { fontSize: 15, color: C.ink, bold: true }); slide.addShape(PPT.ShapeType.rect, { x: 2.55, y, w: 3.2, h: 0.40, fill: { color: 'DCE7EC' }, line: { color: 'DCE7EC' } }); slide.addShape(PPT.ShapeType.rect, { x: 2.55, y, w: 3.2 * value / 9, h: 0.40, fill: { color }, line: { color } }); text(slide, `${value}/40`, 5.93, y + 0.08, 0.7, 0.22, { fontSize: 15, color: C.ink, bold: true }); });
  rect(slide, 0.72, 4.53, 5.95, 1.15, 'F1F5F6', true, 'C8D8DF');
  text(slide, '当前限制', 1.02, 4.82, 1.1, 0.21, { fontSize: 15, color: C.teal, bold: true });
  text(slide, '仅 PyBullet 仿真；无真实 RGB-D、ROS 或真机。\n合成 RGB-D 与 GraspNet 训练域的差异仍影响候选和接触稳定性。', 2.08, 4.68, 4.05, 0.52, { fontSize: 13, color: '36505E', valign: 'top' });
  label(slide, '下一步：闭环与真机迁移前的四项改进', 7.12, 1.42, 4.5, C.teal);
  const next = [['开放词汇分割', '减少 2D 框内背景混入'], ['接触/力稳定性评分', '闭合后再决定是否抬升'], ['视觉反馈与重抓', '将失败检测转为闭环恢复'], ['真机标定与安全控制', '相机外参、碰撞模型、急停']];
  next.forEach(([head, body], i) => { const y = 1.95 + i * 0.90; slide.addShape(PPT.ShapeType.ellipse, { x: 7.14, y: y + 0.02, w: 0.30, h: 0.30, fill: { color: C.teal }, line: { color: C.teal } }); text(slide, String(i + 1), 7.14, y + 0.10, 0.30, 0.10, { fontSize: 8, color: C.white, bold: true, align: 'center' }); text(slide, head, 7.67, y, 2.2, 0.22, { fontSize: 16, color: C.ink, bold: true }); text(slide, body, 9.90, y + 0.02, 2.25, 0.20, { fontSize: 12, color: '56707D' }); });
  rect(slide, 7.12, 5.73, 5.1, 0.48, C.navy, true, C.navy); text(slide, '结论：语言规划已被安全约束，下一阶段的主攻点是物理闭环可靠性。', 7.37, 5.88, 4.6, 0.18, { fontSize: 12.5, color: C.white, bold: true, align: 'center' });
  addFooter(slide, 8, '失败来源：docs/failure_analysis.md；演示视频：outputs/.../demo.mp4', false);
  note(slide, '收尾：这不是把 65% 包装成完成，而是保留失败并将下一步聚焦在接触稳定、开放词汇分割、视觉闭环和真机安全迁移。', ['docs/failure_analysis.md', 'outputs/20260815_001417_040250_evaluation/episodes.csv', 'docs/job_material_evidence.md', 'outputs/20260816_140448_414159_run_seed10/demo.mp4']);
}

fs.mkdirSync(OUT, { recursive: true });
const target = path.join(OUT, 'open_vocab_grasping_portfolio_cn.pptx');
await PPT.writeFile({ fileName: target });
console.log(`Wrote ${target}`);
