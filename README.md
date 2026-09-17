# Reproducibility package — OLAP Dataset/Attribute Recommender (Information, MDPI)

This folder contains the three seeded, deterministic Python scripts behind
every number and figure reported in the paper's evaluation section
(Tables `tab:strategy`, `tab:sensitivity`, `tab:byrole`, `tab:robustness`,
and Figures `fig:strategybar`, `fig:ablation`).

## Files

- **`build_usage_log.py`** — generates the persona-driven simulated usage
  log (seed = 42; 15 personas × 4 sessions; 308 events, 60 sessions) and
  writes it to an Excel workbook (`SessionLog`, `Datasets`, `Attributs`,
  `Similarity_Datasets`, `Similarity_Attributs`, `README` sheets).
- **`evaluate_extended.py`** — leave-one-out evaluation: Random,
  Popularity, Hybrid (proposed), and an item-based $k$NN baseline.
  Produces `summary_by_strategy.csv`, `summary_by_role.csv`,
  `significance_foldlevel.csv`, `significance_personalevel.csv`,
  `fig_precision_ndcg_with_knn.png`, and `results_readable.md`.
  → Tables `tab:strategy`, `tab:sensitivity`, `tab:byrole`.
- **`evaluate_v2.py`** — (a) persona-level paired Wilcoxon significance
  test (n = 15), (b) a second, low-consistency simulated condition
  (alt-fact probability 50% vs. 15% standard) for the robustness check,
  and (c) the own-only / others-only / similarity-only component
  ablation. Produces `robustness_comparison.csv`, `ablation_standard.csv`,
  `ablation_low_consistency.csv`, `fig_robustness.png`, `fig_ablation.png`.
  → Table `tab:robustness`, Figure `fig:ablation`.
- **`scale_experiment.py`** — exploratory, not part of the main pipeline.
  Replicates each of the 15 personas into $N$ independent synthetic
  analysts (same profile, independent draws) at $N \in \{1,3,10,30\}$ and
  re-runs the own-only/others-only/similarity-only/hybrid ablation at
  each scale. Answers "does more data (more analysts sharing the same 15
  profiles) reverse the own-usage-dominance finding?" → no; the gap
  narrows then plateaus. Self-contained; writes
  `scale_experiment_results.csv`.
- **`overlap_experiment.py`** — exploratory, not part of the main
  pipeline. Builds four analyst "families" sharing a fixed fact and a
  deliberately small, shared pool of candidate dimensions/measures, with
  each analyst keeping a fixed personal subset of that pool (individual
  self-consistency preserved) — a condition engineered to be maximally
  favourable to collaborative filtering. Answers the same question under
  a different, more favourable scale-up strategy → same conclusion.
  Self-contained; writes `overlap_experiment_results.csv`.

Both `tab:scaleup` values in the paper (Discussion, Section 7.1) come
directly from these last two scripts' console/CSV output.

## Dependencies

```
pip install numpy scipy matplotlib openpyxl
```

## How to reproduce every reported number

```bash
python3 build_usage_log.py          # writes OLAP_Recommender_UsageLog.xlsx
python3 evaluate_extended.py        # writes /home/claude/eval_out_extended/... (edit OUT path)
python3 evaluate_v2.py              # writes /home/claude/eval_v2_out/...      (edit OUT_ROOT path)
```

`evaluate_extended.py` and `evaluate_v2.py` both `import` directly from
`build_usage_log.py` (`PERSONAS`, `DATASETS`, `ATTRS`, `COLD`,
`attrs_of`, and, for the extended script, the already-generated
`events`), so all three files must sit in the same directory. The two
evaluation scripts hard-code an absolute output directory
(`/home/claude/eval_out_extended`, `/home/claude/eval_v2_out`) — change
`OUT` / `OUT_ROOT` near the top of each file to a path that exists on
your machine before running.

Everything downstream of `build_usage_log.py`'s single `random.seed(42)`
call is deterministic, so re-running this pipeline on an unmodified copy
of these files will regenerate the exact log (308 events / 60 sessions)
and the exact figures reported in the paper.

## Known-and-fixed generator bug

An earlier version of `build_usage_log.py` (and, independently, of the
event generator embedded in `evaluate_v2.py`) sampled a session's
Measure attribute(s) unconditionally from `spec["measures"]`, which are
tied to a persona's **default** fact. When the 15% (50% in the
low-consistency condition) alternate-fact exploration branch fired and
drew a different fact from `spec["alt_facts"]`, the session could end
up recording a Fact and a Measure belonging to two different datasets —
violating the intended invariant that every Fact selection is
accompanied by at least one Measure on the same dataset (and
symmetrically for Dimension/Parameter).

**Fix** (present in the files in this folder): when the drawn fact
differs from the persona's default fact, the Measure pool is drawn from
that fact's own attributes (`attrs_of(fact)`) instead of
`spec["measures"]`. This was applied identically in both
`build_usage_log.py` and the `generate_events()` function inside
`evaluate_v2.py`, since the two contain independent copies of the same
generation logic.

Verified post-fix, over the full regenerated log: zero Fact-without-
Measure or Dimension-without-Parameter cases, at the session, persona,
and global level.

Because both files draw from a single shared `random` stream across all
15 personas in sequence, fixing this bug shifts the pseudo-random draws
for personas processed after the fix point too (not only the directly
affected sessions) — this is an unavoidable, legitimate consequence of
correcting a bug in a single-seed generator, not a sign of a second
error. The regenerated log therefore differs slightly in size from the
original (308 vs. 301 events; 303 vs. 286 evaluable leave-one-out folds)
while remaining fully deterministic under seed = 42.

## Suggested citation / archiving

If deposited on GitHub for reviewer access, we recommend tagging the
exact commit used to produce the submitted manuscript's numbers (e.g.
via a Zenodo DOI) so that "seed = 42" unambiguously points to one
specific version of these three files.
