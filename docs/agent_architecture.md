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
  -> strict AST allowlist + no-builtins recorder execution
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

The real hosted path was end-to-end verified on 2026-08-14 using the Chinese
instruction `请帮我抓取桌面上的杯子`. The returned target was `mug`, local
validation passed, and the downstream robot state machine reached `DONE`.

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

No request header or API key is passed to the audit writer. Failed plans are not
executed. A future iteration may add a separate planning-failure directory and a
constrained retry policy.

## Why arbitrary Python is unavailable

Direct execution of model-written Python would grant capabilities beyond the
grasp task and violate the action whitelist. The implemented code-generation
layer therefore compiles only an already schema-valid plan into one function
with six exact `controller` calls. AST validation rejects imports, builtins,
extra statements, arbitrary attributes, reordered steps and target changes. The
code executes with an empty `__builtins__` mapping against a recorder; only a
matching trace dispatches the real pipeline. Raw provider Python is never run.
