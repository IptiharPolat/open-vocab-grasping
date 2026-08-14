from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from open_vocab_grasping.config import load_config
from open_vocab_grasping.nlp import parse_command
from open_vocab_grasping.pipeline import (
    associate_scene,
    capture_artifacts,
    detect_scene,
    run_open_vocab_grasp,
)


ACTION_CAPTURE = "采集 RGB-D 与点云"
ACTION_DETECT = "YOLO-World 目标检测"
ACTION_ASSOCIATE = "语义与三维抓取关联"
ACTION_RUN = "完整抓取执行"
ACTIONS = (ACTION_RUN, ACTION_DETECT, ACTION_ASSOCIATE, ACTION_CAPTURE)


@dataclass(frozen=True)
class DashboardView:
    status: str
    command: dict[str, Any]
    rgb_path: str | None
    overlay_path: str | None
    topdown_path: str | None
    pointcloud_path: str | None
    video_path: str | None
    timeline: str
    candidate_rows: list[list[Any]]
    metrics: dict[str, Any]
    output_path: str

    def outputs(self) -> tuple[Any, ...]:
        return (
            self.status,
            self.command,
            self.rgb_path,
            self.overlay_path,
            self.topdown_path,
            self.pointcloud_path,
            self.video_path,
            self.timeline,
            self.candidate_rows,
            self.metrics,
            self.output_path,
        )


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(output: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        path = output / name
        if path.is_file():
            return str(path.resolve())
    return None


def _candidate_rows(output: Path) -> list[list[Any]]:
    candidates = _read_json(output / "candidates.json", [])
    rows: list[list[Any]] = []
    for index, candidate in enumerate(candidates):
        center = candidate.get("center", [None, None, None])
        reasons = ", ".join(candidate.get("rejection_reasons", [])) or "-"
        rows.append(
            [
                index,
                bool(candidate.get("accepted", False)),
                round(float(candidate.get("final_score", 0.0)), 4),
                *[round(float(value), 4) for value in center[:3]],
                round(float(candidate.get("width", 0.0)), 4),
                round(float(candidate.get("detection_confidence", 0.0)), 4),
                round(float(candidate.get("clearance", 0.0)), 4),
                round(float(candidate.get("motion_cost", 0.0)), 4),
                reasons,
            ]
        )
    return rows


def _timeline(states: list[str]) -> str:
    if not states:
        return "尚未执行机械臂状态机。"
    return " → ".join(f"`{state}`" for state in states)


def build_dashboard_view(
    output: str | Path,
    action: str,
    command: dict[str, Any],
    payload: dict[str, Any],
) -> DashboardView:
    """Turn actual pipeline artifacts into UI values without altering results."""
    output_path = Path(output).resolve()
    result = _read_json(output_path / "result.json", payload)
    metrics = _read_json(output_path / "metrics.json", payload)
    display_payload = result if (output_path / "result.json").is_file() else metrics
    success = result.get("success") if action == ACTION_RUN else None
    if success is True:
        status = "✅ 抓取成功：系统已完成检测、规划、夹取、抬升和自动判定。"
    elif success is False:
        reason = result.get("failure_reason", "unknown")
        status = f"❌ 抓取失败：`{reason}`。请查看候选拒绝原因和结构化指标。"
    else:
        status = f"✅ `{action}` 已完成，以下内容来自本次真实运行产物。"
    truth_boundary = {
        "method_note": display_payload.get("candidate_generator", display_payload.get("mode")),
        "truth_used_for_semantic_selection": display_payload.get(
            "truth_used_for_semantic_selection",
            display_payload.get("truth_used_for_selection"),
        ),
        "output": str(output_path),
        "result": display_payload,
    }
    return DashboardView(
        status=status,
        command=command,
        rgb_path=_first_existing(output_path, ("rgb.png",)),
        overlay_path=_first_existing(
            output_path,
            ("filtered_candidates_2d.png", "association_2d.png", "detections.png"),
        ),
        topdown_path=_first_existing(
            output_path,
            (
                "filtered_candidates_3d.png",
                "association_3d_topdown.png",
                "pointcloud_topdown.png",
                "depth_preview.png",
            ),
        ),
        pointcloud_path=_first_existing(
            output_path,
            ("scene_pointcloud.ply", "pointcloud.ply", "pointcloud_camera.ply"),
        ),
        video_path=_first_existing(output_path, ("demo.mp4", "demo.gif")),
        timeline=_timeline(list(result.get("state_history", []))),
        candidate_rows=_candidate_rows(output_path),
        metrics=truth_boundary,
        output_path=str(output_path),
    )


def execute_dashboard_action(
    action: str,
    instruction: str,
    seed: int | float,
    config_path: str,
) -> DashboardView:
    """Execute one whitelisted existing pipeline action for the local UI."""
    if action not in ACTIONS:
        raise ValueError(f"Unsupported dashboard action: {action!r}")
    config = load_config(config_path)
    seed_value = int(seed)
    if action == ACTION_CAPTURE:
        command: dict[str, Any] = {
            "action": "observe",
            "target": None,
            "destination": None,
        }
        output, payload = capture_artifacts(config, seed_value)
    else:
        command = parse_command(instruction)
        target = str(command["target"])
        runners: dict[str, Callable[..., tuple[Path, dict[str, Any]]]] = {
            ACTION_DETECT: detect_scene,
            ACTION_ASSOCIATE: associate_scene,
            ACTION_RUN: run_open_vocab_grasp,
        }
        output, payload = runners[action](config, target, seed_value)
    return build_dashboard_view(output, action, command, payload)


def _prepare_gradio_import() -> Any:
    # httpx rejects the non-standard `socks://` scheme found in some desktop
    # proxy setups. The dashboard is local-only, so discard only those invalid
    # values for this process; valid HTTP/SOCKS5 proxies remain untouched.
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        if os.environ.get(name, "").lower().startswith("socks://"):
            os.environ.pop(name, None)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Dashboard dependency import failed: "
            f"{type(exc).__name__}: {exc}. "
            "Install or repair with the ovg interpreter: "
            "/home/ubuntu/miniforge3/envs/ovg/bin/python -m pip install -e '.[ui]'"
        ) from exc
    return gr


def build_dashboard(config_path: str = "configs/default.yaml") -> Any:
    gr = _prepare_gradio_import()
    resolved_config = str(Path(config_path).expanduser().resolve())
    candidate_headers = [
        "ID",
        "保留",
        "最终分数",
        "X (m)",
        "Y (m)",
        "Z (m)",
        "夹爪宽度",
        "检测置信度",
        "间隙",
        "运动代价",
        "拒绝原因",
    ]
    css = """
    .ovg-title {text-align:center; margin-bottom:0.2rem}
    .ovg-subtitle {text-align:center; color:#64748b; margin-bottom:1rem}
    .ovg-status {border-left:5px solid #0f766e; padding-left:0.8rem}
    """
    with gr.Blocks(title="开放词汇机械臂抓取控制台", css=css) as app:
        gr.Markdown("# 开放词汇机械臂抓取控制台", elem_classes="ovg-title")
        gr.Markdown(
            "输入指令并观察真实 YOLO-World、RGB-D 点云、候选筛选和 Panda 执行结果。"
            "当前候选生成器是明确标注的几何基线，不冒充 GraspNet。",
            elem_classes="ovg-subtitle",
        )
        with gr.Row():
            with gr.Column(scale=1):
                action = gr.Radio(ACTIONS, value=ACTION_RUN, label="运行模式")
                instruction = gr.Textbox(
                    value="pick the mug",
                    label="自然语言指令",
                    placeholder="pick the mug / pick the red bottle",
                )
                seed = gr.Number(value=0, precision=0, label="随机种子")
                config = gr.Textbox(value=resolved_config, label="配置文件", interactive=False)
                run_button = gr.Button("运行当前流程", variant="primary")
                gr.Markdown(
                    "提示：完整抓取通常需要约 10 秒。页面只绑定本机，不提供公网分享。"
                )
            with gr.Column(scale=2):
                status = gr.Markdown("等待运行。", elem_classes="ovg-status")
                command = gr.JSON(label="验证后的动作指令")
                timeline = gr.Markdown("尚未执行机械臂状态机。", label="状态机")

        with gr.Tab("二维感知"):
            with gr.Row():
                rgb = gr.Image(label="仿真 RGB 相机", type="filepath")
                overlay = gr.Image(label="检测 / 语义关联", type="filepath")
                topdown = gr.Image(label="三维候选俯视图 / 深度", type="filepath")
        with gr.Tab("三维点云"):
            pointcloud = gr.Model3D(
                label="可旋转、缩放的场景点云",
                display_mode="point_cloud",
                height=560,
            )
        with gr.Tab("机械臂执行"):
            video = gr.Video(label="预抓取 → 接近 → 闭合 → 抬升", autoplay=True)
        with gr.Tab("候选与指标"):
            candidates = gr.Dataframe(
                headers=candidate_headers,
                datatype=["number", "bool"] + ["number"] * 8 + ["str"],
                interactive=False,
                wrap=True,
                label="抓取候选筛选记录",
            )
            metrics = gr.JSON(label="真实运行结果与真值使用边界")
            output_path = gr.Textbox(label="本次输出目录", interactive=False)

        output_components = [
            status,
            command,
            rgb,
            overlay,
            topdown,
            pointcloud,
            video,
            timeline,
            candidates,
            metrics,
            output_path,
        ]

        def run_from_ui(selected_action: str, text: str, value: float, path: str) -> tuple[Any, ...]:
            try:
                return execute_dashboard_action(selected_action, text, value, path).outputs()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                return (
                    f"❌ 运行错误：`{error}`",
                    {"error": error},
                    None,
                    None,
                    None,
                    None,
                    None,
                    "流程未完成。",
                    [],
                    {"error": error},
                    "",
                )

        run_button.click(
            run_from_ui,
            inputs=[action, instruction, seed, config],
            outputs=output_components,
            concurrency_limit=1,
        )
        instruction.submit(
            run_from_ui,
            inputs=[action, instruction, seed, config],
            outputs=output_components,
            concurrency_limit=1,
        )
    return app.queue(default_concurrency_limit=1)


def launch_dashboard(
    config_path: str = "configs/default.yaml",
    host: str = "127.0.0.1",
    port: int = 7860,
    in_browser: bool = False,
) -> None:
    app = build_dashboard(config_path)
    output_root = Path(config_path).expanduser().resolve().parent.parent / "outputs"
    app.launch(
        server_name=host,
        server_port=port,
        inbrowser=in_browser,
        share=False,
        show_api=False,
        allowed_paths=[str(output_root)],
        max_file_size="5mb",
    )
