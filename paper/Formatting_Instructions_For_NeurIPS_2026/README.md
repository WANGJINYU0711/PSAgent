# NeurIPS 2026 Formatting Package for PSAgent

This folder now contains two layers:

- **Official NeurIPS 2026 files**
  - `neurips_2026.sty`
  - `neurips_2026.tex`
  - `checklist.tex`
- **PSAgent-specific writing aids**
  - `psagent_agent_style_guide.md`
  - `psagent_agent_style_shell.tex`

## What Changed

I did **not** modify the official style file `neurips_2026.sty`.

Instead, I added a PSAgent-oriented writing guide and a LaTeX shell that combine:

- the preferences in `paper/preference.md`
- the newer agent-paper style summary in `paper/agent_reference/agent_writing_preference.md`

This keeps the template safe while giving you a directly usable writing workflow.

## Recommended Usage

If you want to check official formatting rules:

- read `neurips_2026.tex`

If you want to draft the actual paper in the target style:

- read `psagent_agent_style_guide.md`
- start from `psagent_agent_style_shell.tex`

## Writing Direction for PSAgent

The current recommendation is to write PSAgent as an **agent/system + benchmark + protocol** paper rather than a pure algorithm paper. In particular:

- keep `Related Work` in the main paper
- use a strong `Figure 1` on page 1
- emphasize concrete bottlenecks early
- treat efficiency as a first-class result
- frame contributions as:
  - problem diagnosis
  - new mechanism / benchmark / protocol
  - empirical gains with cost-aware analysis

## Compile

Example:

```bash
cd paper/Formatting_Instructions_For_NeurIPS_2026
pdflatex psagent_agent_style_shell.tex
pdflatex psagent_agent_style_shell.tex
```

Or:

```bash
cd paper/Formatting_Instructions_For_NeurIPS_2026
latexmk -pdf psagent_agent_style_shell.tex
```

