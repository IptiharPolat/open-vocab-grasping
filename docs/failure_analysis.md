# Failure analysis — current official GraspNet GPU evaluation

Primary source: `outputs/20260815_001417_040250_evaluation/episodes.csv`.
This is an actual fixed-seed experiment: ten seeds each for mug, bottle, bowl
and box in `open-vocab-graspnet`. No successful episode was selected or repeated
to replace a failure. The earlier 20-episode 6/20 result remains in
`PROGRESS.md` as the pre-improvement baseline.

The true Agent source is
`outputs/20260816_140225_904008_agent_evaluation/cases.csv`: all 40 scenes made
a fresh real DeepSeek call before the same downstream pipeline. DeepSeek plan,
target and generated-code validity were 40/40, while full-chain success was
26/40. A target/seed comparison found every success and failure reason identical
to this downstream benchmark, so the failure table below also describes the
complete Agent run.

## Measured results

| Target | Episodes | Detection | YOLO target selection | IK reachable | End-to-end |
| --- | ---: | ---: | ---: | ---: | ---: |
| mug | 10 | 10/10 | 10/10 | 10/10 | **6/10** |
| bottle | 10 | 9/10 | 9/10 | 9/10 | **9/10** |
| bowl | 10 | 10/10 | 9/10 | 9/10 | **5/10** |
| box | 10 | 9/10 | 9/10 | 7/10 | **6/10** |
| overall | 40 | 38/40 (95%) | 37/40 (92.5%) | 35/40 (87.5%) | **26/40 (65%)** |

Candidate generation succeeded in 40/40 episodes. Official GraspNet inference
averaged 5.845 s and total episode time averaged 11.519 s.

## Observed failures

| Failure | Count | Seeds | Interpretation |
| --- | ---: | --- | --- |
| `detection_failed` | 1 | box 38 | No prompt-consensus box survived the real detector; no oracle fallback was used. |
| `no_accepted_candidate` | 4 | bottle 14, bowl 28, box 33/37 | Neural candidates existed, but none passed semantic depth, official collision, gripper geometry, contact, IK and trajectory filters. |
| `object_not_lifted` | 9 | mug 2/5/8/9, bowl 21/23/27/29, box 34 | The state machine executed but stable opposing contact was not maintained through lift. Contact/domain transfer is now the dominant failure type. |

Three target-selection errors are counted independently of the terminal failure
labels. This distinction matters: a detector may emit a box yet choose the wrong
instance, and an execution can be kinematically valid while being semantically
incorrect.

## Successful subsets

- Mug: seeds 0, 1, 3, 4, 6, 7.
- Bottle: seeds 10, 11, 12, 13, 15, 16, 17, 18, 19.
- Bowl: seeds 20, 22, 24, 25, 26.
- Box: seeds 30, 31, 32, 35, 36, 39.

The four seed-0 class demonstrations and all batch artifacts retain raw detector
JSON, candidate rejection reasons, point-cloud visualization and MP4/GIF output.
Semantic selection records `truth_used_for_semantic_selection: false`.

## Next reliability improvements

1. Log per-finger contact points, normals and normal forces during closure/lift,
   then add a short force-stability hold before raising the arm.
2. Use instance masks or open-vocabulary segmentation to reduce background/rim
   points that remain inside a 2D box.
3. Fine-tune or domain-randomize GraspNet on synthetic PyBullet RGB-D while
   preserving the official pretrained checkpoint as the fixed baseline.
4. Add textures, lighting and camera-noise randomization, then calibrate prompt
   consensus thresholds on held-out seeds rather than the reporting seeds.
5. Run at least 30 seeds per target and report bootstrap confidence intervals
   after the method is frozen.
