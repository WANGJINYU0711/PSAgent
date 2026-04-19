# PSAgent NeurIPS 2026 Writing Guide

> This guide merges:
> - `paper/preference.md`
> - `paper/agent_reference/agent_writing_preference.md`
>
> Goal: make the NeurIPS 2026 template usable for a **PSAgent-style agent paper** rather than a generic template paper.

---

## 1. Recommended Paper Positioning

PSAgent should currently be written as:

- a **multi-stage agent systems** paper
- with a **benchmark / protocol / interaction mechanism** emphasis
- plus preliminary or full algorithmic evidence

Avoid framing it as only:

- a pure theory paper
- a pure prompt engineering paper
- a pure LLM benchmark without mechanism insight

The strongest framing is:

1. multi-stage agent systems suffer from end-to-end and partial-share bottlenecks
2. current evaluation protocols do not cleanly expose these bottlenecks
3. PSAgent provides a fixed structured benchmark and interaction protocol
4. experiments show both performance and efficiency differences

---

## 2. Recommended Main Structure

Use the following as the default structure:

1. `Abstract`
2. `Introduction`
3. `Related Work`
4. `Problem Setup`
5. `PSAgent Framework`
6. `Interaction / Sharing Mechanism`
7. `Experiments`
8. `Conclusion`

If space is tight:

- merge `Problem Setup` and `PSAgent Framework`

If motivation experiments are especially important:

- insert a short `Empirical Motivation` section before the main method

---

## 3. Page-1 Strategy

For this paper family, page 1 should do more than satisfy formatting:

- abstract should stay single-paragraph
- add a strong `Figure 1` early
- the first page should communicate:
  - what PSAgent is
  - what bottleneck it addresses
  - why the benchmark/protocol matters
  - at least one headline quantitative takeaway

Recommended `Figure 1` composition:

- left: the five-stage PSAgent pipeline
- middle: shared vs. unshared path semantics or partial-share mechanism
- right: one teaser result, such as regret/cost/interface effect

---

## 4. Abstract Style

Follow this logic:

1. introduce the multi-stage agent setting
2. identify the bottleneck
3. present PSAgent
4. state its main components
5. report the strongest empirical result
6. optionally mention release / protocol / benchmark

Recommended abstract ingredients:

- concrete task setting
- concrete limitation
- method/benchmark name
- 2-3 major components
- 1-2 results with actual numbers

Avoid:

- vague claims like "this problem is important"
- abstract-only theory framing if the work is benchmark-centric
- long motivation without method definition

---

## 5. Introduction Style

Use a 4-paragraph introduction.

### Paragraph 1: Setting and importance

State that many agent systems are naturally multi-stage and that end-to-end quality depends on the full path rather than a single local choice.

### Paragraph 2: Bottlenecks

List concrete failure modes. For PSAgent, likely candidates are:

- stage-local optimization misses end-to-end effects
- partial information is inconsistently shared
- benchmark protocols confound optimizer quality with interface quality
- current evaluation misses cost and interaction structure

Use concrete examples and numbers where possible.

### Paragraph 3: PSAgent overview

State what PSAgent is in one sentence, then unpack:

- structured multi-stage benchmark
- shared/unshared path semantics
- interaction mechanism or protocol
- evaluation protocol

### Paragraph 4: Contributions

Recommended contribution pattern:

1. a new problem/benchmark formulation
2. a new system or interaction mechanism
3. comprehensive experiments with performance + efficiency

---

## 6. Related Work Placement

For PSAgent, keep `Related Work` in the **main paper**, not the appendix.

Suggested subsections:

- Multi-stage or hierarchical agent systems
- LLM-based agent benchmarks and environments
- Agent interaction / communication / memory / planning
- End-to-end or partial-feedback learning in structured systems

Each subsection should end with one explicit sentence of the form:

- `Unlike prior work, PSAgent ...`

---

## 7. Method Section Style

The method section should not jump straight into equations.

Preferred logic:

1. overall framework
2. formal task setup
3. shared vs. unshared semantics
4. learning / interaction mechanism
5. evaluation objective

For each component:

- explain why it is needed
- define how it works
- clarify how it fits the full system

This is especially important for agent papers because reviewers want the system-level story, not just local mechanics.

---

## 8. Experiment Style

Experiments should answer more than "is it better?"

Your experiment section should explicitly answer:

- is PSAgent non-trivial?
- does the interaction mechanism matter?
- does partial sharing matter?
- what is the efficiency tradeoff?
- how stable/generalizable are the conclusions?

Suggested experiment order:

1. Main results
2. Mechanism analysis
3. Ablation study
4. Efficiency / cost analysis
5. Error analysis or case study

Important metrics to foreground:

- regret
- total cost
- exact match / success rate
- token usage
- number of turns or tool calls
- wall-clock time or API cost

For this paper family, efficiency is a main result, not a side note.

---

## 9. Figure and Table Preferences

### Figures

Use figures to carry argument, not decoration.

Recommended:

- `Figure 1`: system overview + teaser result
- `Figure 2`: benchmark or protocol illustration
- `Figure 3`: main quantitative comparison or mechanism analysis

### Tables

Main tables should combine:

- multiple settings
- multiple methods
- at least one efficiency-oriented column

Good column types for PSAgent:

- method
- exact match / SR
- mean total cost
- regret
- token cost / turns / runtime

---

## 10. Writing Style

Use an engineering-forward tone:

- direct
- concrete
- evidence-heavy
- low on vague hype

Prefer:

- `To address this challenge, we propose ...`
- `Our framework consists of ...`
- `We observe that ...`
- `These results show that ...`

Avoid:

- inflated novelty claims without ablations
- overly philosophical openings
- long disconnected related-work catalogues

---

## 11. Concrete PSAgent Advice

For the current PSAgent draft, the biggest writing upgrades are:

- make the bottleneck more explicit in the introduction
- move from "draft summary" tone to "paper claim" tone
- separate benchmark contribution from algorithm contribution
- foreground mechanism effects as a main finding
- add at least one efficiency-centric comparison in the main results

If the evidence remains benchmark-heavy, title and framing should reflect that clearly.

---

## 12. Safe Contribution Template

If you need a conservative contribution block, use this pattern:

1. We formulate a fixed multi-stage agent benchmark with explicit partial-share semantics and structured end-to-end evaluation.
2. We introduce a protocol for comparing learning and interaction mechanisms under shared and unshared path structure.
3. We present empirical results showing non-trivial performance gaps, strong mechanism effects, and clear efficiency tradeoffs.

This is safer than overclaiming a universal algorithmic win.

