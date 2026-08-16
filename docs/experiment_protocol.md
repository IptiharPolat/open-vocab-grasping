# Experiment protocol

Every episode records the resolved YAML, seed, selected mode, structured
candidates (including rejection reasons), RGB, depth/segmentation when relevant,
point cloud, state history, timing, and success measurements. Results must state
whether perception was oracle or real YOLO-World and whether proposals came from
GraspNet or the geometric baseline.

Stage-1 acceptance was one reproducible CPU episode in DIRECT mode. Success
requires both object lift of at least 6 cm and final proximity to the tool. Later
stages retain those success conditions and aggregate fixed-seed target selection,
detection, generation, reachability, grasp and end-to-end rates.

Stage-2 detection success is measured against a bounding box derived from
PyBullet instance segmentation at IoU >= 0.25. Simulator segmentation is read
only after real YOLO-World inference and is written separately as
`oracle_truth.json`; it is never supplied to the model. The highest-confidence
box defines target selection. Zero predictions remain valid measured failures.

Stage-5 single-run success requires all of the following: a real YOLO target box,
at least one geometrically associated candidate, all configured physical and IK
filters passing, completion of the execution state machine, target lift >= 0.06 m,
and final target-to-tool distance <= 0.16 m. Simulator body IDs may support
physics collision queries and post-selection truth metrics, but never semantic
selection. Candidate-level rejection reasons and stage counts are mandatory.

Stage 6 uses paired deterministic seeds. With `seed_start=S`, target index `t`
and episode index `e`, the seed is `S + t*N + e`; the same target/seed pairs are
used in each mode. `--episodes N` means N episodes for every target in every
enabled mode. Oracle and open-vocabulary modes share the same physical, IK and
trajectory filters; only semantic target selection differs. Reports include
all failures. `open-vocab-graspnet` is executable through
`configs/evaluation_graspnet.yaml`. The headline semantic benchmark uses
`configs/evaluation_open_vocab_graspnet.yaml` and ten seeds each for mug,
bottle, bowl and box (40 episodes total). The semantic-free `graspnet-only` mode is executable: target
text is withheld from generation, filtering and ranking, and used only after
selection to measure whether the requested object was lifted.

## DeepSeek Agent protocol

`configs/agent_instruction_suite.yaml` fixes 20 bilingual paraphrases before
evaluation: five each for mug, bottle, bowl and box. Every case calls the selected
planner once and records schema validity, target correctness, generated-Python
AST/execution validity, reported tokens and planning latency. A planner failure
or incorrect target remains in the denominator.

`agent-evaluate` defaults to planner-only evaluation. With
`--robot-cases-per-target N`, the first N labelled cases for each target also run
through the normal Agent entry point using the already obtained response, so no
second API call is made. These rows additionally require synchronized completion
of all six `SafeRobotController` gates and record the downstream perception,
GraspNet, planning and grasp outcome. Planner accuracy and robot success are
reported separately; planner-only rows are never counted as successful grasps.

The first real run used all 20 fixed instructions and one full robot case per
target. It returned 20/20 valid plans, targets and generated programs, then 4/4
successful full-chain executions. Because only one robot seed was used per class,
that 4/4 is integration evidence rather than a general grasp success estimate;
the 40-seed official-GraspNet benchmark remains the reliability reference.

For a true 40-case Agent benchmark, `--full-episodes-per-target 10 --seed-start
0` expands the suite before calling the planner. Target index `t` and episode
index `e` receive seed `0 + t*10 + e`, exactly matching the downstream benchmark:
mug 0-9, bottle 10-19, bowl 20-29 and box 30-39. Every expanded case makes one
new planner request and, only after safe validation, attempts one robot episode.
The end-to-end denominator is all 40 requested cases: planner errors, incorrect
targets, code-validation failures and runtime exceptions count as failures even
when physical motion never starts.

The actual full run completed all 40 hosted calls and robot attempts. Planning,
target selection at the Agent layer and generated-code validation were 40/40;
robot task success was 26/40. Per target and per seed, success/failure labels and
failure reasons exactly matched the paired downstream-only benchmark. Language
groups are descriptive only because instruction language is confounded with seed;
a language-effect study would repeat every paraphrase on identical scenes.
