# Portfolio PPT QA report — 2026-08-23

## Deliverables checked

- `open_vocab_grasping_portfolio_cn.pptx`: editable PPTX generated with the
  project-local PptxGenJS 4.0.1 builder.
- `open_vocab_grasping_portfolio_cn.pdf`: local LibreOffice export.
- `rendered/final/slide-1.png` … `slide-8.png`: final 140-DPI render set.
- `contact_sheet.png`: final 2×4 contact sheet.

## Automated checks

| Check | Result |
| --- | --- |
| Formal CSV guard in PPT builder | Passed: exactly 40 rows; 38/40 detection, 37/40 target selection, 35/40 IK, 26/40 end-to-end; Agent `plan_valid` 40/40. |
| PPTX slide XML count | Passed: 8 slides. |
| Speaker notes | Passed: 8 notes pages; every notes page contains `[Sources]`. |
| PDF export | Passed: 8 pages, 16:9 (960.009 × 540 pt). |
| Final PNG rendering | Passed: 8 files, each 1867 × 1050 px. |
| Placeholder-token scan | Passed: no `TODO`, `lorem`, `ipsum`, `xxx`, or `placeholder` in extracted PDF text. |
| Source syntax check | Passed: `node tools/presentation/build_portfolio.mjs` rebuilt the deck without error. |
| Diff whitespace check | Passed: `git diff --check`. |

## Visual review

The final contact sheet was reviewed, followed by full-size review of all eight
slides. A first-pass visual review found an undesirable automatic line break in
the Slide 6 illustrative DSL text. The source was corrected to explicit,
intentional line breaks, then Slide 6 and the whole deck were re-rendered.

Final review outcome:

- no visible off-slide clipping or element overlap;
- eight page titles remain on one line;
- real project images are not stretched or cropped over their grasp targets;
- stage and category charts retain explicit denominators;
- dark/light alternation, footer system and teal accent are consistent;
- compact captions, source footers and diagram labels are intentionally smaller
  than body copy; main explanatory body copy remains presentation-readable;
- no Mock, oracle, CPU geometric baseline, physical-robot, ROS, or arbitrary
  LLM-code-execution claim appears in the deck.

## Known environment note

`soffice` emitted a non-fatal `javaldx` warning. PDF export and all rendering
checks completed successfully using an isolated LibreOffice user profile. No
system configuration was changed.

## Source ledger

- Formal grasp benchmark:
  `outputs/20260815_001417_040250_evaluation/episodes.csv` and `summary.json`.
- Formal real-DeepSeek benchmark:
  `outputs/20260816_140225_904008_agent_evaluation/cases.csv` and `summary.json`.
- Real visual seed-10 bottle episode:
  `outputs/20260816_140448_414159_run_seed10/`.
- Asset selection:
  `portfolio_assets/manifest.csv` and `used_assets.csv`.
- Source-code evidence boundary:
  `docs/job_material_evidence.md`, `docs/failure_analysis.md`,
  `docs/coordinate_frames.md`, and `docs/agent_architecture.md`.
