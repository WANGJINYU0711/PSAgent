# PSAgent NeurIPS 2026 Draft

This folder contains the first LaTeX draft derived from the current PSAgent repository.

## Files

- `main.tex`: paper draft
- `neurips_2026.sty`: copied from the provided NeurIPS 2026 template
- `checklist.tex`: copied from the provided NeurIPS 2026 template

## Main evidence used in the draft

- Benchmark construction notes:
  - `notes/temp_responses_store.txt`
  - `notes/telecom_mms_formal_experiment_protocol.md`
  - `notes/telecom_mms_experiment_freeze.md`
- New 100-task benchmark:
  - `data/derived/telecom_mms_fixed_tree_base_v2_100/manifest.json`
  - `data/derived/telecom_mms_fixed_tree_base_v2_100/tasks.json`
- Simulated main experiment:
  - `outputs/telecom_mms_main_experiment/overall_summary.json`
- LLM-backed sanity slice:
  - `outputs/telecom_mms_llm_strong_direct/overall_summary.json`
- Smoke diagnostics:
  - `outputs/telecom_llm_e2e_smoke/20260417_smoke_assessment.md`

## Compile

Example commands on a machine with TeX installed:

```bash
cd paper/psagent_neurips2026_draft
pdflatex main.tex
pdflatex main.tex
```

If `latexmk` is available:

```bash
cd paper/psagent_neurips2026_draft
latexmk -pdf main.tex
```

## Caveat

This environment does not have `pdflatex`, so the draft was not compiled locally in this session.
