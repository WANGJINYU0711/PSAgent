# PS Telecom MMS Experiment Memory (2026-04-30)

This note captures the working interpretation for the telecom MMS profile-switch / shared-basin experiments.

## What the experiment is doing

We are evaluating whether a tree-structured agent workflow can convert agent profile preferences into lower terminal cost under a fixed-stage telecom MMS task family. The key comparison is between methods such as `risky_ps`, `direct_multistage_exp3`, and `epsilon_exp3` under the same schedule, tree topology, and cost normalization. The main diagnostic question is whether a profile-matched route yields lower terminal quality cost than a mismatched route, and whether PS can exploit the tree structure better than direct bandit-style routing.

## Current interpretation

- Execution appears stable enough that when PS reaches the right profile/path region, it converts well into lower cost.
- The remaining weakness is upstream routing: the tree is not yet forcing PS into the good path region reliably enough.
- In the current smoke runs, matched profile/task pairs have much lower cost than mismatched pairs, supporting the hypothesis that profile and terminal cost are aligned.
- The highest-cost failures concentrate in complex post-switch local repair patterns, especially the `fdddd`-like cases involving APN / roaming / SIM / permission issues.

## Tree design hypothesis

The important control knob is not just global branching factor. The tree should be redesigned to:

1. Reduce early sibling similarity, especially near the root and the target corridor.
2. Move pruning earlier so that post-switch target paths enter a narrow corridor sooner.
3. Reduce exposure of trap/barrier siblings inside the post-switch target region.
4. Preserve enough structure for hard-transfer control as a separate variant if needed.

## Practical takeaway

The best next step is a router-style tree variant, not a cost-function overhaul. The tree should more aggressively separate `fast` vs `deep` behavior early, and shrink the terminal target corridor so that PS can realize its profile advantage more consistently.
