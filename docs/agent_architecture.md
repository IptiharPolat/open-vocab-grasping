# DeepSeek terminal Agent architecture

## Scope

The Agent performs high-level task interpretation. It does not replace metric
geometry, collision checking, inverse kinematics, trajectory generation or motor
control. Those remain deterministic, tested robot tools.

```text
terminal instruction
  -> DeepSeek JSON plan (or labelled mock/deterministic planner)
  -> Draft 2020-12 JSON Schema
  -> scene target whitelist
  -> locally generated six-call Python DSL
  -> strict AST allowlist + no-builtins SafeRobotController execution
  -> existing YOLO/RGB-D/grasp/filter/IK/execution pipeline
  -> structured result and audit artifacts
```

## Provider boundary

`configs/agent.yaml` selects the provider, current model, API URL, timeout and
the *name* of the environment variable that stores the credential. The secret
value is read only at request time and is used only in the HTTP Authorization
header. The HTTP client ignores inherited proxy variables because this desktop
historically exports an invalid `socks://` proxy.

The current official hosted model configured here is `deepseek-v4-flash`.
`deepseek-v2` is not presented as the model actually called.

The fully combined hosted path was end-to-end verified on 2026-08-15 using the
Chinese instruction `请帮我抓取桌面上的瓶子`. The returned target was `bottle`
and the execution mode was `open-vocab-graspnet`; local schema/Python validation
passed, real YOLO-World and the official checkpoint ran, and the downstream
robot state machine reached `DONE` with a 0.117293 m lift. The selection path
did not use simulator truth.

## Validated plan

For the first safe release, the only accepted plan is:

```json
{
  "action": "pick",
  "target": "mug",
  "steps": [
    "observe",
    "detect",
    "generate_grasps",
    "select_grasp",
    "execute",
    "evaluate"
  ],
  "execution_mode": "open-vocab-simple",
  "explanation": "The user asked to pick the mug."
}
```

The target must be one exact name from the configured PyBullet scene. Extra
fields, reordered or missing steps, other actions, Python code and unknown
targets are rejected before the robot pipeline starts.

## Modes and truth boundary

- `deepseek`: real network request; requires `DEEPSEEK_API_KEY`.
- `mock`: deterministic bilingual test double, always labelled `mock`.
- `deterministic`: original English grammar, no external model.

The planner mode does not change the grasp-candidate truth boundary. With
`configs/agent.yaml` the backend is the CPU geometric baseline; with
`configs/agent_graspnet.yaml` it is the verified isolated official GraspNet
service. YOLO-World remains the real semantic detector in both.

## Audit files

Successful planner dispatches add these files to the ordinary pipeline output:

- `agent_request.json`: user instruction and sanitized request payload;
- `agent_response.json`: provider response or explicit mock response;
- `agent_plan.json`: locally validated plan;
- `agent_generated_plan.py`: canonical controller DSL compiled from that plan;
- `agent_generated_plan_trace.json`: executed and rechecked six-step trace;
- `agent_result.json`: plan, provider metadata and robot result.

The generated Python runs in a restricted worker. Each controller method grants
one capability to the matching real pipeline stage, then blocks until that stage
reports completion. The audit trace therefore records request/start/completion
times from the actual run; it is not merely a pre-dispatch method replay.

No request header or API key is passed to the audit writer. Failed plans are not
executed. A future iteration may add a separate planning-failure directory and a
constrained retry policy.

## Why arbitrary Python is unavailable

Direct execution of model-written Python would grant capabilities beyond the
grasp task and violate the action whitelist. The implemented code-generation
layer therefore compiles only an already schema-valid plan into one function
with six exact `controller` calls. AST validation rejects imports, builtins,
extra statements, arbitrary attributes, reordered steps and target changes. The
code executes with an empty `__builtins__` mapping against `SafeRobotController`.
The real pipeline blocks at each stage until the generated call authorizes it,
and completion releases the program to request the next step. Raw provider
Python is never run.

## Reliability suite

The `agent-evaluate` command reads the fixed bilingual cases in
`configs/agent_instruction_suite.yaml`. It writes one sanitized audit per case,
`cases.csv`, JSON and Markdown summaries. Planner-only cases measure language,
schema and generated-code behavior. An optional bounded number of cases per
target reuses the same response to run the synchronized robot pipeline, avoiding
duplicate API calls and keeping full-chain success separate from planner accuracy.

The actual 2026-08-16 run received 20 successful `deepseek-v4-flash` responses:
20/20 plans passed schema validation, 20/20 selected the labelled target and
20/20 generated programs passed validation. Mean planning latency was 0.877 s
with 6,135 total tokens. Four requested robot rows used real YOLO-World and the
official checkpoint and succeeded 4/4. This is a bounded one-case-per-class
integration check; downstream physical reliability remains the separate 26/40
fixed-seed benchmark.

The full benchmark mode uses `--full-episodes-per-target`. It cycles the five
predeclared paraphrases for each target over deterministic paired seeds and makes
a fresh hosted request for every robot scene. Its reported full-chain rate uses
all requested cases as the denominator, including any plan or validation failure
that safely prevents robot motion.

The actual 40-case run returned 40/40 valid `deepseek-v4-flash` plans, correct
targets and valid generated programs at 0.861 s mean latency and 12,267 tokens.
All 40 synchronized robot tasks executed with real YOLO-World and the official
checkpoint; 26 succeeded. Its target/seed successes and failure reasons exactly
matched the paired downstream benchmark, so DeepSeek introduced no additional
failure in this fixed suite. English and Chinese robot success rates must not be
compared causally because their paraphrases were assigned to different seeds;
planner correctness was 100% for both languages.
