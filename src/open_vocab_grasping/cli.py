from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from open_vocab_grasping.config import load_config
from open_vocab_grasping.evaluation.runner import run_evaluation
from open_vocab_grasping.evaluation.geometry_benchmark import run_geometry_benchmark
from open_vocab_grasping.evaluation.ablation import run_filter_ablation
from open_vocab_grasping.logging_utils import configure_logging
from open_vocab_grasping.nlp import parse_command
from open_vocab_grasping.pipeline import (
    associate_scene,
    capture_artifacts,
    detect_scene,
    export_graspnet_request,
    run_cpu_smoke,
    run_open_vocab_grasp,
)


def _command_output(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if not executable:
        return None
    try:
        return subprocess.run(
            [executable, *command[1:]], capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def doctor_report() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    distributions = {
        "numpy": "numpy",
        "pybullet": "pybullet",
        "open3d": "open3d",
        "yaml": "PyYAML",
        "torch": "torch",
        "ultralytics": "ultralytics",
        "jsonschema": "jsonschema",
        "gradio": "gradio",
    }
    for name, distribution in distributions.items():
        try:
            packages[name] = version(distribution)
        except PackageNotFoundError:
            packages[name] = None
    nvidia = _command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    torch_cuda: dict[str, Any] | None = None
    try:
        import torch

        torch_cuda = {
            "torch_version": torch.__version__,
            "built_cuda": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        pass
    disk = shutil.disk_usage(Path.cwd())
    return {
        "os": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_environment": Path(sys.prefix).name,
        "shell_conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "package_managers": {
            "conda": _command_output(["conda", "--version"]),
            "uv": _command_output(["uv", "--version"]),
            "pip": _command_output([sys.executable, "-m", "pip", "--version"]),
        },
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "disk_current_directory_gib": {
            "total": round(disk.total / 2**30, 2),
            "free": round(disk.free / 2**30, 2),
        },
        "compiler": _command_output(["gcc", "--version"]),
        "cmake": _command_output(["cmake", "--version"]),
        "nvidia_smi": nvidia,
        "nvidia_device_nodes": sorted(str(path) for path in Path("/dev").glob("nvidia*")),
        "torch_cuda": torch_cuda,
        "packages": packages,
        "deepseek_api_key_set": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "graspnet_compatibility": {
            "isolation": "Conda environment ovg-graspnet-cu117",
            "verified_stack": "PyTorch 1.13.1 + CUDA 11.7 + PointNet2/KNN sm_86",
            "checkpoint_present": (Path.cwd() / "weights/graspnet-checkpoint-rs.tar").is_file(),
            "extensions_present": bool(
                list((Path.cwd() / "third_party/graspnet-baseline/pointnet2").glob("pointnet2/_ext*.so"))
            ),
            "note": (
                "The main ovg torch build is CPU-only by design. Official GraspNet GPU "
                "inference runs in the isolated service; see services/graspnet/README.md."
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-vocab-grasping")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="Audit runtime and optional accelerators")
    for name in ("smoke", "capture"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--seed", type=int)
        if name == "smoke":
            command.add_argument("--target")
    run = subcommands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--seed", type=int)
    run_target = run.add_mutually_exclusive_group()
    run_target.add_argument("--target")
    run_target.add_argument(
        "--instruction", help='Validated natural-language command, e.g. "pick the red mug"'
    )
    parse = subcommands.add_parser("parse", help="Parse and validate a whitelisted command")
    parse.add_argument("--instruction", required=True)
    detect = subcommands.add_parser("detect")
    detect.add_argument("--target", required=True)
    detect.add_argument("--config", required=True)
    detect.add_argument("--seed", type=int)
    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--targets", required=True)
    evaluate.add_argument("--episodes", type=int, required=True)
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--modes", help="Comma-separated executable modes; defaults to config")
    evaluate.add_argument("--seed-start", type=int, help="Override the first deterministic seed")
    geometry = subcommands.add_parser("geometry-eval", help="Run fixed-seed coordinate accuracy checks")
    geometry.add_argument("--episodes", type=int, required=True)
    geometry.add_argument("--config", required=True)
    geometry.add_argument("--seed", type=int)
    ablation = subcommands.add_parser("ablate", help="Execute fixed-seed grasp filter ablations")
    ablation.add_argument("--target", required=True)
    ablation.add_argument("--config", required=True)
    ablation.add_argument("--seed", type=int)
    export = subcommands.add_parser("export-graspnet")
    export.add_argument("--config", required=True)
    export.add_argument("--seed", type=int)
    associate = subcommands.add_parser("associate")
    associate.add_argument("--target", required=True)
    associate.add_argument("--config", required=True)
    associate.add_argument("--seed", type=int)
    dashboard = subcommands.add_parser("dashboard", help="Launch the local interactive dashboard")
    dashboard.add_argument("--config", default="configs/default.yaml")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=7860)
    dashboard.add_argument("--in-browser", action="store_true")
    agent = subcommands.add_parser("agent", help="Run the validated terminal task-planning agent")
    agent.add_argument("--config", default="configs/agent.yaml")
    agent.add_argument("--seed", type=int)
    agent.add_argument("--mode", choices=("deepseek", "mock", "deterministic"), default="deepseek")
    agent.add_argument("--instruction", help="Run one instruction and exit; omit for interactive mode")
    agent_evaluate = subcommands.add_parser(
        "agent-evaluate", help="Evaluate multilingual planning and optional full robot runs"
    )
    agent_evaluate.add_argument("--config", default="configs/agent_graspnet.yaml")
    agent_evaluate.add_argument("--suite", default="configs/agent_instruction_suite.yaml")
    agent_evaluate.add_argument(
        "--mode", choices=("deepseek", "mock", "deterministic"), default="deepseek"
    )
    agent_evaluate.add_argument("--robot-cases-per-target", type=int, default=0)
    agent_evaluate.add_argument("--seed-start", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor_report(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "parse":
        print(json.dumps(parse_command(args.instruction), indent=2, ensure_ascii=False))
        return 0
    if args.command == "dashboard":
        from open_vocab_grasping.dashboard import launch_dashboard

        launch_dashboard(
            config_path=args.config,
            host=args.host,
            port=args.port,
            in_browser=args.in_browser,
        )
        return 0
    if args.command == "agent":
        from open_vocab_grasping.agent.terminal import run_agent_once, run_agent_repl

        agent_config = load_config(args.config)
        agent_seed = int(args.seed if args.seed is not None else agent_config.get("seed", 0))
        if args.instruction:
            execution = run_agent_once(args.config, args.instruction, agent_seed, args.mode)
            print(json.dumps(execution.to_dict(), ensure_ascii=False, indent=2))
            return 0 if execution.success else 2
        return run_agent_repl(args.config, agent_seed, args.mode)
    if args.command == "agent-evaluate":
        from open_vocab_grasping.agent.evaluation import run_agent_evaluation

        agent_config = load_config(args.config)
        output, summary = run_agent_evaluation(
            agent_config,
            args.suite,
            args.mode,
            args.robot_cases_per_target,
            args.seed_start,
        )
        print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False, indent=2))
        return 0
    config = load_config(args.config)
    seed = int(args.seed if getattr(args, "seed", None) is not None else config.get("seed", 0))
    if args.command == "capture":
        output, metrics = capture_artifacts(config, seed)
        print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))
        return 0
    if args.command == "export-graspnet":
        output, metrics = export_graspnet_request(config, seed)
        print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))
        return 0
    if args.command == "associate":
        output, metrics = associate_scene(config, args.target, seed)
        print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))
        return 0 if metrics["association_success"] else 2
    if args.command == "smoke":
        output, result = run_cpu_smoke(config, args.target, seed)
        print(json.dumps({"output": str(output), **result}, indent=2))
        return 0 if result["success"] else 2
    if args.command == "run":
        target = parse_command(args.instruction)["target"] if args.instruction else args.target
        output, result = run_open_vocab_grasp(config, target, seed)
        print(json.dumps({"output": str(output), **result}, indent=2))
        return 0 if result["success"] else 2
    if args.command == "detect":
        output, metrics = detect_scene(config, args.target, seed)
        print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))
        return 0
    if args.command == "evaluate":
        targets = [target.strip() for target in args.targets.split(",") if target.strip()]
        if not targets:
            raise ValueError("At least one target is required")
        configured_modes = config.get("evaluation", {}).get(
            "modes", ["oracle-perception", "open-vocab-simple"]
        )
        modes = (
            [mode.strip() for mode in args.modes.split(",") if mode.strip()]
            if args.modes else [str(mode) for mode in configured_modes]
        )
        if args.seed_start is not None:
            config.setdefault("evaluation", {})["seed_start"] = int(args.seed_start)
        output, summary = run_evaluation(config, targets, args.episodes, modes)
        print(json.dumps({"output": str(output), "summary": summary}, indent=2))
        return 0
    if args.command == "geometry-eval":
        output, summary = run_geometry_benchmark(config, args.episodes, seed)
        print(json.dumps({"output": str(output), "summary": summary}, indent=2))
        return 0 if summary["passed"] else 2
    if args.command == "ablate":
        output, summary = run_filter_ablation(config, args.target, seed)
        print(json.dumps({"output": str(output), "summary": summary}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
