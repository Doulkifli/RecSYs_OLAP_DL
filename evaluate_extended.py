"""
Extended leave-one-out evaluation: adds an item-based kNN collaborative-
filtering baseline alongside Random / Popularity / Hybrid, and adds the
persona-level (n=15) paired Wilcoxon significance test reported in the paper,
in addition to the fold-level test already in evaluate.py.

kNN baseline (item-based CF, standard weighted-sum prediction):
  score(c) = sum_i sim(c, i) * ownusage(analyst, i)   for all items i (same
             pool/type as c) that the analyst has used at least once in that
             role, normalized by sum_i |sim(c, i)| over that same item set.
  Falls back to 0 (i.e., unranked / tied with any other zero-scored item,
  broken by the same random tiebreak used elsewhere) if the analyst has no
  usage at all for that role -- this is the honest cold-start behavior of a
  pure item-based kNN recommender, with no metadata fallback of its own.

This mirrors the "similarity-only" ablation already in the paper but uses
the analyst's *entire* own usage history as the neighborhood (as a real
item-based kNN recommender would), rather than only the current session's
already-chosen reference items.
"""
import random, math, os, csv
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_usage_log import events, ANALYSTS, DATASETS, ATTRS, PERSONAS, COLD

OUT = "/home/claude/eval_out_extended"
os.makedirs(OUT, exist_ok=True)

EVAL_SEED = 7
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)

K_VALUES = [1, 3, 5]
STRATEGIES = ["random", "popularity", "hybrid", "knn"]

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

def rank_candidates(candidates, analyst, role, own_lookup, others_lookup, sim_matrix,
                     reference_items, all_role_items, strategy, rng):
    cand = list(candidates)
    if strategy == "random":
        rng.shuffle(cand)
        return cand

    if strategy == "knn":
        neighborhood = [i for i in all_role_items if own_lookup(i) > 0]
        scored = []
        for c in cand:
            if not neighborhood:
                pred = 0.0
            else:
                num = sum(sim_matrix.get((c, i), 0.0) * own_lookup(i) for i in neighborhood if i != c)
                den = sum(abs(sim_matrix.get((c, i), 0.0)) for i in neighborhood if i != c)
                pred = (num / den) if den > 0 else 0.0
            scored.append(((pred, rng.random()), c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored]

    scored = []
    for c in cand:
        oth = others_lookup(c)
        if strategy == "popularity":
            key = (oth, rng.random())
        else:  # hybrid
            own = own_lookup(c)
            simscore = 0.0
            if reference_items:
                vals = [sim_matrix.get((c, r), 0.0) for r in reference_items if r != c]
                simscore = max(vals) if vals else 0.0
            key = (own, oth, simscore, rng.random())
        scored.append((key, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]

def ndcg_at_k(rank, k):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def session_context(events_excl, sid):
    return [e for e in events_excl if e[0] == sid]

def build_fold(idx):
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
        prior_relevant = [d for d in DATASETS if ds_counts[persona][d]['F'] > 0
                           or sum(ds_counts[a][d]['F'] for a in ANALYSTS if a != persona) > 0]
        reference = prior_relevant
        sim_matrix = sim_ds
        all_role_items = list(DATASETS)
    elif role == "D":
        session_fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
        other_dims = {e[4] for e in ctx if e[3] == "Dataset" and e[5] == "D"}
        candidates = [d for d in DATASETS if d != session_fact and d not in other_dims]
        own_lookup = lambda c: ds_counts[persona][c]['D']
        others_lookup = lambda c: sum(ds_counts[a][c]['D'] for a in ANALYSTS if a != persona)
        reference = [session_fact] if session_fact else []
        sim_matrix = sim_ds
        all_role_items = list(DATASETS)
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
        all_role_items = pool
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
        all_role_items = pool
    else:
        return None

    if name not in candidates or len(candidates) == 0:
        return None

    return dict(idx=idx, sid=sid, persona=persona, rnd=rnd, role=role, truth=name,
                candidates=candidates, own_lookup=own_lookup, others_lookup=others_lookup,
                sim_matrix=sim_matrix, reference=reference, all_role_items=all_role_items)

folds = []
for i in range(len(events)):
    if events[i][1] in COLD:
        continue
    f = build_fold(i)
    if f is not None:
        folds.append(f)

print(f"Built {len(folds)} evaluable leave-one-out folds "
      f"(F={sum(f['role']=='F' for f in folds)}, D={sum(f['role']=='D' for f in folds)}, "
      f"M={sum(f['role']=='M' for f in folds)}, P={sum(f['role']=='P' for f in folds)})")

results = []
rngs = {s: random.Random(EVAL_SEED) for s in STRATEGIES}

for f in folds:
    for strat in STRATEGIES:
        ranked = rank_candidates(f["candidates"], f["persona"], f["role"],
                                  f["own_lookup"], f["others_lookup"], f["sim_matrix"],
                                  f["reference"], f["all_role_items"], strat, rngs[strat])
        rank = ranked.index(f["truth"]) + 1 if f["truth"] in ranked else None
        row = dict(idx=f["idx"], sid=f["sid"], persona=f["persona"], role=f["role"],
                   coldstart=(f["rnd"] == 1), strategy=strat, rank=rank,
                   pool_size=len(f["candidates"]))
        for k in K_VALUES:
            row[f"hit@{k}"] = 1 if (rank is not None and rank <= k) else 0
            row[f"ndcg@{k}"] = ndcg_at_k(rank, k)
        results.append(row)

with open(f"{OUT}/per_fold_results.csv", "w", newline="") as fh:
    cols = ["idx", "sid", "persona", "role", "coldstart", "strategy", "rank", "pool_size"] + \
           [f"hit@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES]
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in results:
        w.writerow(r)

def summarize(rows):
    n = len(rows)
    line = {"n_folds": n}
    for k in K_VALUES:
        prec = np.mean([r[f"hit@{k}"] / k for r in rows])
        rec = np.mean([r[f"hit@{k}"] for r in rows])
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        ndcg = np.mean([r[f"ndcg@{k}"] for r in rows])
        line[f"P@{k}"] = round(prec, 4)
        line[f"R@{k}"] = round(rec, 4)
        line[f"F1@{k}"] = round(f1, 4)
        line[f"NDCG@{k}"] = round(ndcg, 4)
    return line

def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

by_strategy = []
for strat in STRATEGIES:
    rows = [r for r in results if r["strategy"] == strat]
    s = summarize(rows)
    s["group"] = strat
    by_strategy.append(s)
write_csv(by_strategy, f"{OUT}/summary_by_strategy.csv")

by_role = []
for strat in STRATEGIES:
    for role in ["F", "D", "M", "P"]:
        rows = [r for r in results if r["strategy"] == strat and r["role"] == role]
        if not rows:
            continue
        s = summarize(rows)
        s["group"] = f"{strat} / {role}"
        by_role.append(s)
write_csv(by_role, f"{OUT}/summary_by_role.csv")

# ---- fold-level Wilcoxon (as in original evaluate.py) ----
sig_rows = []
for k in K_VALUES:
    hyb = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "hybrid"])
    pop = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "popularity"])
    rnd = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "random"])
    knn = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "knn"])
    def safe_w(a, b):
        try:
            return wilcoxon(a, b)
        except ValueError:
            return (float("nan"), float("nan"))
    _, p_hp = safe_w(hyb, pop)
    _, p_hr = safe_w(hyb, rnd)
    _, p_hk = safe_w(hyb, knn)
    sig_rows.append({"k": k, "hybrid_vs_popularity_p_foldlevel": p_hp,
                      "hybrid_vs_random_p_foldlevel": p_hr,
                      "hybrid_vs_knn_p_foldlevel": p_hk})
write_csv(sig_rows, f"{OUT}/significance_foldlevel.csv")

# ---- persona-level (n=15) paired Wilcoxon, matching the paper's protocol ----
personas_only = [p for p in PERSONAS.keys()]
persona_sig_rows = []
for k in K_VALUES:
    def persona_means(strategy):
        out = []
        for p in personas_only:
            vals = [r[f"ndcg@{k}"] for r in results if r["strategy"] == strategy and r["persona"] == p]
            out.append(np.mean(vals) if vals else 0.0)
        return np.array(out)
    hyb_m = persona_means("hybrid")
    pop_m = persona_means("popularity")
    rnd_m = persona_means("random")
    knn_m = persona_means("knn")
    def safe_w(a, b):
        try:
            return wilcoxon(a, b)
        except ValueError:
            return (float("nan"), float("nan"))
    _, p_hp = safe_w(hyb_m, pop_m)
    _, p_hr = safe_w(hyb_m, rnd_m)
    _, p_hk = safe_w(hyb_m, knn_m)
    persona_sig_rows.append({"k": k, "n_personas": len(personas_only),
                              "hybrid_vs_popularity_p": p_hp,
                              "hybrid_vs_random_p": p_hr,
                              "hybrid_vs_knn_p": p_hk})
write_csv(persona_sig_rows, f"{OUT}/significance_personalevel.csv")

# ---- figures ----
fig, ax = plt.subplots(figsize=(7, 4))
labels = STRATEGIES
p5 = [next(s["P@5"] for s in by_strategy if s["group"] == strat) for strat in labels]
n5 = [next(s["NDCG@5"] for s in by_strategy if s["group"] == strat) for strat in labels]
x = np.arange(len(labels)); w = 0.35
ax.bar(x - w/2, p5, w, label="Precision@5")
ax.bar(x + w/2, n5, w, label="NDCG@5")
ax.set_xticks(x); ax.set_xticklabels(["Random", "Popularity", "Hybrid\n(proposed)", "Item-based\nkNN"])
ax.set_ylabel("Score"); ax.set_title("Recommendation quality by strategy")
ax.legend(); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_precision_ndcg_with_knn.png", dpi=200)
plt.close(fig)

def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)

cols_main = ["group", "n_folds"] + [f"{m}@{k}" for k in K_VALUES for m in ["P", "R", "F1", "NDCG"]]

with open(f"{OUT}/results_readable.md", "w") as fh:
    fh.write("# Leave-One-Out Evaluation Results (with kNN baseline)\n\n")
    fh.write(f"Total evaluable folds: {len(folds)}\n\n")
    fh.write("## Table: Overall accuracy by strategy\n\n")
    fh.write(md_table(by_strategy, cols_main) + "\n\n")
    fh.write("## Table: Accuracy by strategy x role\n\n")
    fh.write(md_table(by_role, cols_main) + "\n\n")
    fh.write("## Table: Fold-level Wilcoxon (NDCG@k), matches original evaluate.py's test\n\n")
    fh.write(md_table(sig_rows, ["k", "hybrid_vs_popularity_p_foldlevel", "hybrid_vs_random_p_foldlevel", "hybrid_vs_knn_p_foldlevel"]) + "\n\n")
    fh.write("## Table: Persona-level (n=15) paired Wilcoxon (NDCG@k), matches paper's Sec. 6 protocol\n\n")
    fh.write(md_table(persona_sig_rows, ["k", "n_personas", "hybrid_vs_popularity_p", "hybrid_vs_random_p", "hybrid_vs_knn_p"]) + "\n\n")
    fh.write("Figure: fig_precision_ndcg_with_knn.png\n")

print("Done. Results in", OUT)
print(open(f"{OUT}/results_readable.md").read())
