# Agent reliability evaluation

Actual instruction cases: **40**.
Requested planner mode: `deepseek`.
Successful DeepSeek responses: **40**.
Total reported tokens: **12267**.

## Overall

- Schema-valid plans: 40/40 (100.0%)
- Correct targets: 40/40 (100.0%)
- Valid generated Python: 40/40 (100.0%)
- Mean planning latency: 0.861 s
- Robot success among executed cases: 26/40 (65.0%)
- Full-chain success among requested cases: 26/40 (65.0%)

## By target

| Target | Cases | Valid plan | Target accuracy | Python valid | Full-chain success |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottle | 10 | 100.0% | 100.0% | 100.0% | 9/10 (90.0%) |
| bowl | 10 | 100.0% | 100.0% | 100.0% | 5/10 (50.0%) |
| box | 10 | 100.0% | 100.0% | 100.0% | 6/10 (60.0%) |
| mug | 10 | 100.0% | 100.0% | 100.0% | 6/10 (60.0%) |

## Interpretation boundary

Planner-only cases validate language understanding, schema compliance and generated-code safety. Rows with `robot_requested=true` form the full-chain denominator; planning, target or runtime failures that prevent execution still count as end-to-end failures.

All values are computed from `cases.csv`; failed cases remain in the denominator.
