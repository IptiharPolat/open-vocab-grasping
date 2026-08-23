# PPT / Slides Skill audit — 2026-08-23

## Scope and local capability check

This review was performed before authoring the portfolio deck. The local Codex
skill catalogue was searched for `Presentations`, `slides`, `pptx`,
`powerpoint`, and `artifact-tool` on 2026-08-23.

- No installed, first-party `Presentations` / `slides` / `pptx` Skill was
  available in this session, so there was no such `SKILL.md` to read. The
  available built-in skills were `imagegen`, `openai-docs`, `review-agent`,
  `skill-installer`, `plugin-creator`, and `skill-creator`.
- Temporary connector-plugin descriptions for Canva, Google Slides, SharePoint,
  and Figma were found, but were not used: this project must keep local
  experiment assets local and no connected cloud presentation service is
  necessary.
- Local, inspectable tools available for a headless editable deck are Node.js
  and LibreOffice (`soffice`). `pptxgenjs` was not preinstalled and
  `@oai/artifact-tool` was not exposed in this runtime.

Consequently, the requested first-party Presentation workflow cannot be invoked
in this environment. The delivered deck uses a project-local, source-controlled
PptxGenJS builder and local LibreOffice rendering instead. This is an explicit
environment limitation, not a claim that a built-in Presentation Skill ran.

The local workflow retains the quality gates expected from a presentation skill:
editable PPTX geometry, speaker notes, source footers, full-deck rendering,
contact-sheet review, and checks for overflow, overlap, cropped images and
placeholder text.

## Search and source-audit method

GitHub/web searches used: `"SKILL.md" presentation pptx`, `"SKILL.md"
PowerPoint`, `"SKILL.md" slides agent`, `Codex presentation skill`, and
`research portfolio slides skill`. Four candidate repositories were shallow
cloned to a temporary `/tmp/ovg-ppt-skill-audit.*` directory and inspected
read-only. No candidate install command, package manager command, build command
or bundled script was run.

The review checked origin URL, pinned commit, last commit date, license,
`SKILL.md`, build dependency, external-service instructions, global-config
changes, shell-download instructions, macros/binaries, examples, and QA
instructions. No macros (`.vba`/`.bas`) or opaque executables were executed.

## Candidate comparison

| Skill | GitHub | License | 主要能力 | 安全风险 | 对本项目的价值 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `pptx` | [jingsiding-arch/skills](https://github.com/jingsiding-arch/skills), `3d36c7f2e2e40dd22f115a240573f529613cb057`, 2026-07-02 | `SKILL.md` declares Proprietary; no usable root OSS license found | PptxGenJS guidance, visual QA, thumbnail/PDF conversion | Requires global `npm install -g`; repository includes unrelated skills with network/download instructions; proprietary terms prevent reuse | Its general visual-QA list is sensible, but it is not license-safe to incorporate | **Excluded**: no suitable open-source license and asks for global installation. |
| `powerpoint` | [the-open-agent/openagent](https://github.com/the-open-agent/openagent), `5ede160808ce308515ffb4e194cd71ab08814715`, 2026-08-21 | Apache-2.0 | Editable PptxGenJS decks, notes, template analysis, render/read QA | Its documented `pptx_write` / `pptx_read` tool contract is not available in this Codex runtime; no macro/network requirement in the inspected skill | Useful confirmation of a source-first editable-PPTX workflow | **Excluded for execution**: requires unavailable proprietary runtime tools. No code copied. |
| `codex-ppt` | [qybaihe/codex-ppt](https://github.com/qybaihe/codex-ppt), `d743a07482a8ad3a1453d3e67f2bc678d171d1c2`, 2026-06-01 | MIT | Image-first 16:9 deck, optional editable reconstruction, storyboard and QA records | Requires an image-generation backend (`gpt-image-2` or compatible third-party API) for its core workflow; default output is a page bitmap, and editable reconstruction adds a multi-step dependency chain | Storyboard discipline is valuable | **Excluded**: would introduce an external image-generation dependency and image-first slides; it does not meet this task's local, evidence-only editable-PPTX preference. |
| `presentation-skill` | [sirilsengolraj-source/presentation-skill](https://github.com/sirilsengolraj-source/presentation-skill), `3a22eed290fa2205b6a1e2de5549b4429c5fffd0`, 2026-07-13 | MIT | PptxGenJS editable rendering; scientific-figure/lab-result variants; source plans; notes; geometry, placeholder and rendered visual QA; example/release galleries | Repository contains optional OpenAI-image and Wikimedia-fetch helpers, and a legacy `python-pptx` route; package dependency installation is required to execute its renderer | Strongest fit for research storytelling, evidence-source plans, local assets and visual QA | **Selected as design reference only**: use its MIT-licensed high-level principles; do not execute scripts, install it, fetch assets, use image generation, or use its Python renderer. |

## Final selection and permitted use

The final deck does **not** execute any third-party Skill. The design/reference
selection is `presentation-skill` at the pinned commit above, limited to these
ideas:

1. evidence-first scientific/lab slide roles rather than sales-card layouts;
2. a persisted source-and-asset plan with one main claim per slide;
3. source footers and speaker-note provenance;
4. render, contact-sheet, overflow/overlap and placeholder-text QA before
   delivery.

The implementation uses an original project-local PptxGenJS script, with local
project assets and CSV/JSON evidence only. It does not copy the candidate's
scripts, templates or generated examples. PptxGenJS is used only after its own
published package license is checked, is installed locally under the project
tooling directory, and is never installed globally.

The following candidate features are explicitly not used: API/image generation,
Wikimedia fetching, curl/wget/sudo install paths, macros, opaque binaries,
external-cloud uploads, `python-pptx`, global skill installation, and any
third-party renderer or QA script. Speaker notes, footers, charts, rendering,
and QA are created and checked locally.

## Design contract for this deck

- Audience: embodied-intelligence, robotics-algorithm, manipulation, and
  research-assistant interviewers.
- Format: eight editable 16:9 slides, Chinese, 5–8 minutes.
- Visual language: dark navy/black-gray background, cyan-green accent,
  high-contrast research figures, sparse editable geometric diagrams; no
  commercial-template imagery or unrelated robot photos.
- Evidence boundary: only the real local seed-10 bottle run and the two
  retained 40-row formal CSV/JSON result sets are used. The deck says
  `PyBullet Simulation`; it does not present oracle, mock, CPU geometric
  baseline, physical robot, ROS, or unconstrained LLM-code execution as final
  results.
