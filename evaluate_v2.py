"""
Evaluation v2 -- addresses three methodological weaknesses identified in review:

  FIX 1 (persona-level significance): the pooled fold-level Wilcoxon test in
      evaluate.py treats 286 folds as i.i.d., but they cluster into only 15
      independent generative processes (personas). This version aggregates to
      one mean NDCG@k per persona per strategy (n=15 paired observations) and
      re-runs the significance test on that, reporting it as the primary claim.

  FIX 2 (low-consistency robustness variant): the original personas are highly
      self-consistent by construction (85% probability of sticking to a typical
      fact, etc.), which lets "own prior usage" trivially recover the held-out
      item. This version re-generates the usage log with much weaker
      self-consistency (50% alt-fact, 60% alt-dimension, 40% exploratory noise)
      and re-runs the full pipeline, to test whether the hybrid method's
      advantage survives when personas are noisier / less predictable.

  FIX 3 (component ablation): adds own_only / others_only / similarity_only
      strategies (each isolating exactly one signal used by the hybrid ranker)
      to quantify how much of the hybrid's advantage is attributable to each
      component, rather than reporting only the combined effect.

Outputs land in /home/claude/eval_v2_out/{standard,low_consistency}/ plus a
top-level comparison and ablation report.
"""
import random, math, os, csv
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_usage_log import PERSONAS, DATASETS, ATTRS, COLD, attrs_of

OUT_ROOT = "/home/claude/eval_v2_out"
os.makedirs(OUT_ROOT, exist_ok=True)

K_VALUES = [1, 3, 5]
MAIN_STRATEGIES = ["random", "popularity", "hybrid"]
ABLATION_STRATEGIES = ["own_only", "others_only", "similarity_only", "hybrid"]
ALL_STRATEGIES = ["random", "popularity", "own_only", "others_only", "similarity_only", "hybrid"]

ANALYSTS = list(PERSONAS.keys()) + COLD

# ------------------------------------------------------------- generation
def generate_events(seed, sessions_per_persona=4, alt_fact_prob=0.15,
                     dim_alt_prob=0.25, explore_prob=0.10):
    rng = random.Random(seed)
    events = []
    sid = 0

    def add(session, persona, rnd, itype, name, role):
        events.append((session, persona, rnd, itype, name, role))

    for persona, spec in PERSONAS.items():
        for rnd in range(1, sessions_per_persona + 1):
            sid += 1
            session = f"S{sid:03d}"

            fact = spec["fact"]
            if spec["alt_facts"] and rng.random() < alt_fact_prob:
                fact = rng.choice(spec["alt_facts"])
            add(session, persona, rnd, "Dataset", fact, "F")

            dim_pool = list(spec["dims"])
            if spec["dim_alt"] and rng.random() < dim_alt_prob:
                dim_pool = dim_pool + [rng.choice(spec["dim_alt"])]
            n_dims = min(len(dim_pool), rng.randint(1, max(1, len(dim_pool))))
            chosen_dims = rng.sample(dim_pool, n_dims)
            for d in chosen_dims:
                add(session, persona, rnd, "Dataset", d, "D")

            # Measures must belong to whichever dataset was actually chosen as
            # fact (bug fix, mirrors the one applied to build_usage_log.py):
            # previously spec["measures"] was used unconditionally, which are
            # tied to spec["fact"], so an alt_fact session could record a
            # Measure for a dataset different from the Fact it belonged to.
            if fact == spec["fact"]:
                measures_pool = spec["measures"]
            else:
                measures_pool = attrs_of(fact) or spec["measures"]
            n_meas = min(len(measures_pool), rng.randint(1, max(1, len(measures_pool))))
            chosen_meas = rng.sample(measures_pool, n_meas)
            for m in chosen_meas:
                add(session, persona, rnd, "Attribute", m, "M")

            for d in chosen_dims:
                plist = spec["params"].get(d, attrs_of(d)[:1])
                n_p = min(len(plist), rng.randint(1, max(1, len(plist))))
                for p in rng.sample(plist, n_p):
                    add(session, persona, rnd, "Attribute", p, "P")

            if rng.random() < explore_prob:
                pool = attrs_of(fact)
                if pool:
                    add(session, persona, rnd, "Attribute", rng.choice(pool), "M")

    return events

# ------------------------------------------------------------- shared utilities
def dataset_of_attr(attr):
    for d in DATASETS:
        if attr.startswith(d + "."):
            return d
    return None

def build_counts(event_list):
    ds_counts = {a: {d: {'F': 0, 'D': 0} for d in DATASETS} for a in ANALYSTS}
    at_counts = {a: {x: {'M': 0, 'P': 0} for x in ATTRS} for a in ANALYSTS}
    for (sid, persona, rnd, itype, name, role) in event_list:
        if itype == "Dataset":
            ds_counts[persona][name][role] += 1
        else:
            at_counts[persona][name][role] += 1
    return ds_counts, at_counts

def cosine_sim(items, vec_fn):
    vecs = {it: vec_fn(it) for it in items}
    norms = {it: np.linalg.norm(v) for it, v in vecs.items()}
    sim = {}
    for i in items:
        for j in items:
            ni, nj = norms[i], norms[j]
            sim[(i, j)] = float(np.dot(vecs[i], vecs[j]) / (ni * nj)) if ni > 0 and nj > 0 else 0.0
    return sim

def ds_vec_fn(ds_counts):
    def f(d):
        v = []
        for a in ANALYSTS:
            v.append(ds_counts[a][d]['F']); v.append(ds_counts[a][d]['D'])
        return np.array(v, dtype=float)
    return f

def at_vec_fn(at_counts):
    def f(x):
        v = []
        for a in ANALYSTS:
            v.append(at_counts[a][x]['M']); v.append(at_counts[a][x]['P'])
        return np.array(v, dtype=float)
    return f

def ndcg_at_k(rank, k):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def session_context(events_excl, sid):
    return [e for e in events_excl if e[0] == sid]

def build_fold(events, idx):
    held = events[idx]
    sid, persona, rnd, itype, name, role = held
    masked_events = events[:idx] + events[idx + 1:]
    ctx = session_context(masked_events, sid)

    ds_counts, at_counts = build_counts(masked_events)
    sim_ds = cosine_sim(DATASETS, ds_vec_fn(ds_counts))
    sim_at = cosine_sim(ATTRS, at_vec_fn(at_counts))

    if role == "F":
        candidates = list(DATASETS)
        own_lookup = lambda c: ds_counts[persona][c]['F']
        others_lookup = lambda c: sum(ds_counts[a][c]['F'] for a in ANALYSTS if a != persona)
        reference = [d for d in DATASETS if ds_counts[persona][d]['F'] > 0
                     or sum(ds_counts[a][d]['F'] for a in ANALYSTS if a != persona) > 0]
        sim_matrix = sim_ds
    elif role == "D":
        session_fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
        other_dims = {e[4] for e in ctx if e[3] == "Dataset" and e[5] == "D"}
        candidates = [d for d in DATASETS if d != session_fact and d not in other_dims]
        own_lookup = lambda c: ds_counts[persona][c]['D']
        others_lookup = lambda c: sum(ds_counts[a][c]['D'] for a in ANALYSTS if a != persona)
        reference = [session_fact] if session_fact else []
        sim_matrix = sim_ds
    elif role == "M":
        session_fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
        if session_fact is None:
            return None
        other_meas = {e[4] for e in ctx if e[3] == "Attribute" and e[5] == "M"}
        pool = [a for a in ATTRS if dataset_of_attr(a) == session_fact and a not in other_meas]
        candidates = pool
        own_lookup = lambda c: at_counts[persona][c]['M']
        others_lookup = lambda c: sum(at_counts[a][c]['M'] for a in ANALYSTS if a != persona)
        reference = [a for a in pool if at_counts[persona][a]['M'] > 0
                     or sum(at_counts[a2][a]['M'] for a2 in ANALYSTS if a2 != persona) > 0]
        sim_matrix = sim_at
    elif role == "P":
        target_ds = dataset_of_attr(name)
        session_dims = {e[4] for e in ctx if e[3] == "Dataset" and e[5] == "D"}
        if target_ds not in session_dims:
            return None
        other_params = {e[4] for e in ctx if e[3] == "Attribute" and e[5] == "P"
                         and dataset_of_attr(e[4]) == target_ds}
        pool = [a for a in ATTRS if dataset_of_attr(a) == target_ds and a not in other_params]
        candidates = pool
        own_lookup = lambda c: at_counts[persona][c]['P']
        others_lookup = lambda c: sum(at_counts[a][c]['P'] for a in ANALYSTS if a != persona)
        reference = [a for a in pool if at_counts[persona][a]['P'] > 0
                     or sum(at_counts[a2][a]['P'] for a2 in ANALYSTS if a2 != persona) > 0]
        sim_matrix = sim_at
    else:
        return None

    if name not in candidates or len(candidates) == 0:
        return None

    return dict(idx=idx, sid=sid, persona=persona, rnd=rnd, role=role, truth=name,
                candidates=candidates, own_lookup=own_lookup, others_lookup=others_lookup,
                sim_matrix=sim_matrix, reference=reference)

def rank_candidates(candidates, own_lookup, others_lookup, sim_matrix, reference_items, strategy, rng):
    cand = list(candidates)
    if strategy == "random":
        rng.shuffle(cand)
        return cand

    def simscore(c):
        if not reference_items:
            return 0.0
        vals = [sim_matrix.get((c, r), 0.0) for r in reference_items if r != c]
        return max(vals) if vals else 0.0

    scored = []
    for c in cand:
        own = own_lookup(c)
        oth = others_lookup(c)
        sim = simscore(c)
        if strategy == "popularity" or strategy == "others_only":
            key = (oth, rng.random())
        elif strategy == "own_only":
            key = (own, rng.random())
        elif strategy == "similarity_only":
            key = (sim, rng.random())
        else:  # hybrid
            key = (own, oth, sim, rng.random())
        scored.append((key, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]

def build_folds(events):
    folds = []
    for i in range(len(events)):
        if events[i][1] in COLD:
            continue
        f = build_fold(events, i)
        if f is not None:
            folds.append(f)
    return folds

def run_strategies(folds, strategies, seed):
    results = []
    rngs = {s: random.Random(seed) for s in strategies}
    for f in folds:
        for strat in strategies:
            ranked = rank_candidates(f["candidates"], f["own_lookup"], f["others_lookup"],
                                      f["sim_matrix"], f["reference"], strat, rngs[strat])
            rank = ranked.index(f["truth"]) + 1 if f["truth"] in ranked else None
            row = dict(idx=f["idx"], sid=f["sid"], persona=f["persona"], role=f["role"],
                       coldstart=(f["rnd"] == 1), strategy=strat, rank=rank,
                       pool_size=len(f["candidates"]))
            for k in K_VALUES:
                row[f"hit@{k}"] = 1 if (rank is not None and rank <= k) else 0
                row[f"ndcg@{k}"] = ndcg_at_k(rank, k)
            results.append(row)
    return results

# ------------------------------------------------------------- summarizing
def summarize(rows):
    n = len(rows)
    line = {"n_folds": n}
    for k in K_VALUES:
        prec = np.mean([r[f"hit@{k}"] / k for r in rows]) if n else 0.0
        rec = np.mean([r[f"hit@{k}"] for r in rows]) if n else 0.0
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        ndcg = np.mean([r[f"ndcg@{k}"] for r in rows]) if n else 0.0
        line[f"P@{k}"] = round(prec, 4); line[f"R@{k}"] = round(rec, 4)
        line[f"F1@{k}"] = round(f1, 4); line[f"NDCG@{k}"] = round(ndcg, 4)
    return line

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ------------------------------------------------------------- FIX 1: persona-level significance
def persona_level_test(results, strat_a, strat_b, k):
    """Aggregate to one mean NDCG@k per persona per strategy, paired test on n=15 (or fewer)."""
    personas = sorted(set(r["persona"] for r in results))
    a_vals, b_vals = [], []
    for p in personas:
        a_rows = [r[f"ndcg@{k}"] for r in results if r["persona"] == p and r["strategy"] == strat_a]
        b_rows = [r[f"ndcg@{k}"] for r in results if r["persona"] == p and r["strategy"] == strat_b]
        if a_rows and b_rows:
            a_vals.append(np.mean(a_rows)); b_vals.append(np.mean(b_rows))
    a_vals, b_vals = np.array(a_vals), np.array(b_vals)
    n = len(a_vals)
    diffs = a_vals - b_vals
    n_nonzero = int(np.sum(diffs != 0))
    try:
        w_stat, w_p = wilcoxon(a_vals, b_vals)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    try:
        t_stat, t_p = ttest_rel(a_vals, b_vals)
    except Exception:
        t_stat, t_p = float("nan"), float("nan")
    return dict(k=k, n_personas=n, n_nonzero_diffs=n_nonzero,
                mean_a=round(float(np.mean(a_vals)), 4), mean_b=round(float(np.mean(b_vals)), 4),
                mean_diff=round(float(np.mean(diffs)), 4),
                wilcoxon_p=w_p, ttest_p=t_p)

# ------------------------------------------------------------- run one condition
def run_condition(label, seed, alt_fact_prob, dim_alt_prob, explore_prob, sessions_per_persona=4):
    out = f"{OUT_ROOT}/{label}"
    os.makedirs(out, exist_ok=True)
    events = generate_events(seed, sessions_per_persona, alt_fact_prob, dim_alt_prob, explore_prob)
    folds = build_folds(events)
    results = run_strategies(folds, ALL_STRATEGIES, seed=7)
    write_csv(results, f"{out}/per_fold_results.csv")

    by_strategy = []
    for strat in ALL_STRATEGIES:
        rows = [r for r in results if r["strategy"] == strat]
        s = summarize(rows); s["strategy"] = strat
        by_strategy.append(s)
    write_csv(by_strategy, f"{out}/summary_by_strategy.csv")

    # FIX 1: persona-level significance, hybrid vs popularity and hybrid vs random
    sig_rows = []
    for k in K_VALUES:
        sig_rows.append(dict(comparison="hybrid_vs_popularity", **persona_level_test(results, "hybrid", "popularity", k)))
        sig_rows.append(dict(comparison="hybrid_vs_random", **persona_level_test(results, "hybrid", "random", k)))
    write_csv(sig_rows, f"{out}/significance_persona_level.csv")

    n_events = len(events)
    print(f"[{label}] events={n_events} folds={len(folds)} "
          f"hybrid P@1={next(s['P@1'] for s in by_strategy if s['strategy']=='hybrid')} "
          f"popularity P@1={next(s['P@1'] for s in by_strategy if s['strategy']=='popularity')}")
    return dict(label=label, events=events, folds=folds, results=results,
                by_strategy=by_strategy, sig_rows=sig_rows, out=out)

# =============================================================== MAIN
if __name__ == "__main__":
    print("=== Condition A: STANDARD (original, self-consistent personas) ===")
    standard = run_condition("standard", seed=42, alt_fact_prob=0.15, dim_alt_prob=0.25, explore_prob=0.10)

    print("\n=== Condition B: LOW-CONSISTENCY (weakened persona self-consistency) ===")
    low = run_condition("low_consistency", seed=42, alt_fact_prob=0.50, dim_alt_prob=0.60, explore_prob=0.40)

    # ---------- FIX 2: robustness comparison table ----------
    comparison = []
    for cond in (standard, low):
        s_by = {s["strategy"]: s for s in cond["by_strategy"]}
        comparison.append(dict(
            condition=cond["label"], n_events=len(cond["events"]),
            hybrid_P1=s_by["hybrid"]["P@1"], popularity_P1=s_by["popularity"]["P@1"],
            random_P1=s_by["random"]["P@1"],
            gap_hybrid_minus_popularity_P1=round(s_by["hybrid"]["P@1"] - s_by["popularity"]["P@1"], 4),
            hybrid_NDCG5=s_by["hybrid"]["NDCG@5"], popularity_NDCG5=s_by["popularity"]["NDCG@5"],
            gap_hybrid_minus_popularity_NDCG5=round(s_by["hybrid"]["NDCG@5"] - s_by["popularity"]["NDCG@5"], 4),
        ))
    write_csv(comparison, f"{OUT_ROOT}/robustness_comparison.csv")

    # ---------- FIX 3: component ablation (on standard condition) ----------
    ablation = [s for s in standard["by_strategy"] if s["strategy"] in ABLATION_STRATEGIES]
    order = {s: i for i, s in enumerate(ABLATION_STRATEGIES)}
    ablation.sort(key=lambda r: order[r["strategy"]])
    write_csv(ablation, f"{OUT_ROOT}/ablation_standard.csv")

    ablation_low = [s for s in low["by_strategy"] if s["strategy"] in ABLATION_STRATEGIES]
    ablation_low.sort(key=lambda r: order[r["strategy"]])
    write_csv(ablation_low, f"{OUT_ROOT}/ablation_low_consistency.csv")

    # ---------- figures ----------
    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels = ["Random", "Popularity", "Own-only", "Others-only", "Similarity-only", "Hybrid"]
    keys = ["random", "popularity", "own_only", "others_only", "similarity_only", "hybrid"]
    std_p1 = [next(s["P@1"] for s in standard["by_strategy"] if s["strategy"] == k) for k in keys]
    low_p1 = [next(s["P@1"] for s in low["by_strategy"] if s["strategy"] == k) for k in keys]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, std_p1, w, label="Standard personas")
    ax.bar(x + w/2, low_p1, w, label="Low-consistency personas")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Precision@1"); ax.set_title("Precision@1 by strategy and persona consistency")
    ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{OUT_ROOT}/fig_robustness.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    abl_labels = ["Own-only", "Others-only", "Similarity-only", "Hybrid"]
    abl_p1 = [next(s["P@1"] for s in standard["by_strategy"] if s["strategy"] == k) for k in ABLATION_STRATEGIES]
    abl_ndcg5 = [next(s["NDCG@5"] for s in standard["by_strategy"] if s["strategy"] == k) for k in ABLATION_STRATEGIES]
    x = np.arange(len(abl_labels)); w = 0.35
    ax.bar(x - w/2, abl_p1, w, label="Precision@1")
    ax.bar(x + w/2, abl_ndcg5, w, label="NDCG@5")
    ax.set_xticks(x); ax.set_xticklabels(abl_labels)
    ax.set_ylabel("Score"); ax.set_title("Component ablation (standard personas)")
    ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(f"{OUT_ROOT}/fig_ablation.png", dpi=200)
    plt.close(fig)

    # ---------- readable summary ----------
    with open(f"{OUT_ROOT}/results_readable.md", "w") as fh:
        fh.write("# Evaluation v2 -- Persona-level significance, robustness, ablation\n\n")

        fh.write("## Fix 1: Persona-level significance (standard condition)\n\n")
        fh.write("| comparison | k | n_personas | n_nonzero_diffs | mean_a | mean_b | mean_diff | wilcoxon_p | ttest_p |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in standard["sig_rows"]:
            fh.write(f"| {r['comparison']} | {r['k']} | {r['n_personas']} | {r['n_nonzero_diffs']} | "
                      f"{r['mean_a']} | {r['mean_b']} | {r['mean_diff']} | {r['wilcoxon_p']:.4g} | {r['ttest_p']:.4g} |\n")

        fh.write("\n## Fix 2: Robustness under low-consistency personas\n\n")
        fh.write("| condition | n_events | hybrid P@1 | popularity P@1 | random P@1 | gap (hybrid-pop) P@1 | hybrid NDCG@5 | popularity NDCG@5 | gap NDCG@5 |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in comparison:
            fh.write(f"| {r['condition']} | {r['n_events']} | {r['hybrid_P1']} | {r['popularity_P1']} | "
                      f"{r['random_P1']} | {r['gap_hybrid_minus_popularity_P1']} | {r['hybrid_NDCG5']} | "
                      f"{r['popularity_NDCG5']} | {r['gap_hybrid_minus_popularity_NDCG5']} |\n")

        fh.write("\n## Fix 3: Component ablation (standard personas)\n\n")
        fh.write("| strategy | n_folds | P@1 | R@1 | NDCG@5 |\n|---|---|---|---|---|\n")
        for r in ablation:
            fh.write(f"| {r['strategy']} | {r['n_folds']} | {r['P@1']} | {r['R@1']} | {r['NDCG@5']} |\n")

        fh.write("\n## Component ablation (low-consistency personas)\n\n")
        fh.write("| strategy | n_folds | P@1 | R@1 | NDCG@5 |\n|---|---|---|---|---|\n")
        for r in ablation_low:
            fh.write(f"| {r['strategy']} | {r['n_folds']} | {r['P@1']} | {r['R@1']} | {r['NDCG@5']} |\n")

    print("\nDone. See", OUT_ROOT, "/ results_readable.md")
    print(open(f"{OUT_ROOT}/results_readable.md").read())
