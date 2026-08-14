from __future__ import annotations

import json
from typing import Any

from open_vocab_grasping.agent.runtime import AgentExecution, run_agent_instruction
from open_vocab_grasping.config import load_config


HELP_TEXT = """Commands:
  /help                    Show this help
  /mode deepseek           Use the real DeepSeek API
  /mode mock               Use the labelled offline mock planner
  /mode deterministic      Use the English deterministic parser
  /seed N                  Set the PyBullet random seed
  /status                  Show session settings
  /last                    Show the previous structured result
  /quit                    Exit

Natural-language example: 请帮我抓取桌面上的杯子
"""


def _event_printer(event: str, fields: dict[str, Any]) -> None:
    labels = {
        "planning_started": "大模型规划",
        "plan_validated": "计划已验证",
        "robot_pipeline_started": "机器人流水线启动",
        "robot_pipeline_finished": "机器人流水线结束",
    }
    print(f"[{labels.get(event, event)}] {json.dumps(fields, ensure_ascii=False)}")
    if event == "robot_pipeline_finished" and fields.get("video"):
        print(f"[演示视频] {fields['video']}")


def run_agent_once(config_path: str, instruction: str, seed: int, mode: str) -> AgentExecution:
    config = load_config(config_path)
    return run_agent_instruction(config, instruction, seed, mode, _event_printer)


def run_agent_repl(config_path: str, seed: int, mode: str) -> int:
    config = load_config(config_path)
    last: AgentExecution | None = None
    print("Open-Vocabulary Grasping Agent")
    print(f"mode={mode} seed={seed} config={config_path}")
    print("输入 /help 查看帮助；输入 /quit 退出。")
    while True:
        try:
            text = input("\novg> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent stopped.")
            return 0
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            return 0
        if text == "/help":
            print(HELP_TEXT)
            continue
        if text == "/status":
            print(json.dumps({"mode": mode, "seed": seed, "config": config_path}, ensure_ascii=False, indent=2))
            continue
        if text == "/last":
            print(json.dumps(last.to_dict() if last else {"message": "no previous run"}, ensure_ascii=False, indent=2))
            continue
        if text.startswith("/mode "):
            requested = text.split(maxsplit=1)[1].strip()
            if requested not in {"deepseek", "mock", "deterministic"}:
                print("mode must be deepseek, mock, or deterministic")
            else:
                mode = requested
                print(f"mode={mode}")
            continue
        if text.startswith("/seed "):
            try:
                seed = int(text.split(maxsplit=1)[1])
                print(f"seed={seed}")
            except ValueError:
                print("seed must be an integer")
            continue
        if text.startswith("/"):
            print("Unknown command. Use /help.")
            continue
        try:
            last = run_agent_instruction(config, text, seed, mode, _event_printer)
            print(json.dumps(last.to_dict(), ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"[错误] {type(exc).__name__}: {exc}")
