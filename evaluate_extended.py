"""
evaluate_extended.py
=====================
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

WHAT THIS SCRIPT PRODUCES, AND WHICH PAPER TABLE EACH FEEDS:
  - summary_by_strategy.csv    -> Table `tab:strategy`, Table `tab:sensitivity`
  - summary_by_role.csv        -> Table `tab:byrole`
  - significance_personalevel.csv -> the persona-level Wilcoxon p-values
                                      quoted in the text (Section 6)
  - fig_precision_ndcg_with_knn.png -> the bar chart figure in the paper

HOW LEAVE-ONE-OUT WORKS HERE (read this once, it explains every fold):
  For every single logged event (one Fact/Dimension/Measure/Parameter
  selection), we pretend that ONE event never happened ("mask" it),
  recompute all usage counts and similarity matrices from the
  remaining events, then ask each strategy to rank every possible
  candidate for that slot. If the masked (true) item comes back near
  the top of the ranking, that's a "hit" -- this is exactly the
  standard recommender-systems leave-one-out protocol, applied once
  per event in the log (so ~300 folds from ~300 events).
"""
import random, math, os, csv
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")  # no display available in this environment; write PNGs directly
import matplotlib.pyplot as plt

# Pulls the already-generated log (list of events) and the fixed
# vocabularies (which analysts/datasets/attributes/personas exist)
# straight from build_usage_log.py -- this script does NOT regenerate
# the log itself, it only evaluates the one build_usage_log.py made.
from build_usage_log import events, ANALYSTS, DATASETS, ATTRS, PERSONAS, COLD

OUT = "./eval_out_extended"
os.makedirs(OUT, exist_ok=True)

# A SEPARATE seed from the log generator's SEED=42. This one only
# controls random tie-breaking during ranking (see rng.random() calls
# below) and NumPy's RNG (unused directly here but seeded for safety);
# it has no effect on which log was generated, only on how ties are
# broken when two candidates score identically.
EVAL_SEED = 7
random.seed(EVAL_SEED)
np.random.seed(EVAL_SEED)

K_VALUES = [1, 3, 5]  # we report Precision/NDCG at these three cutoffs
STRATEGIES = ["random", "popularity", "hybrid", "knn"]

def dataset_of_attr(attr):
    """Inverse of build_usage_log.attrs_of(): given an attribute name
    like 'SE_mysql.Users.Reputation', return its owning dataset
    'SE_mysql.Users'. Relies on the same naming convention (attribute
    names are prefixed by 'dataset.')."""
    for d in DATASETS:
        if attr.startswith(d + "."):
            return d
    return None

def build_counts(event_list):
    """Turn a list of raw events into two nested count dictionaries:
      ds_counts[analyst][dataset]['F' or 'D']  -> how many times that
        analyst used that dataset in that role
      at_counts[analyst][attribute]['M' or 'P'] -> same, for attributes
    This is the Python equivalent of the Datasets/Attributs sheets in
    the Excel workbook. Called once per fold on the MASKED event list
    (i.e. with one event removed), which is what makes this a true
    leave-one-out count -- the held-out event's own count doesn't
    leak into the ranking."""
    ds_counts = {a: {d: {'F': 0, 'D': 0} for d in DATASETS} for a in ANALYSTS}
    at_counts = {a: {x: {'M': 0, 'P': 0} for x in ATTRS} for a in ANALYSTS}
    for (sid, persona, rnd, itype, name, role) in event_list:
        if itype == "Dataset":
            ds_counts[persona][name][role] += 1
        else:
            at_counts[persona][name][role] += 1
    return ds_counts, at_counts

def cosine_sim(items, vec_fn):
    """Given a list of items and a function that returns each item's
    usage vector, return a dict {(item_i, item_j): cosine_similarity}
    for every ordered pair (including i==j, which is always 1.0).
    This is the Python equivalent of the Similarity_Datasets /
    Similarity_Attributs sheets in the Excel workbook -- same cosine
    formula, just computed with numpy instead of spreadsheet
    SUMPRODUCT/SUMSQ formulas."""
    vecs = {it: vec_fn(it) for it in items}
    norms = {it: np.linalg.norm(v) for it, v in vecs.items()}
    sim = {}
    for i in items:
        for j in items:
            ni, nj = norms[i], norms[j]
            # guard against a zero-length vector (an item nobody has
            # ever used) dividing by zero; similarity is defined as 0
            # in that case, matching the workbook's IFERROR(...,0).
            sim[(i, j)] = float(np.dot(vecs[i], vecs[j]) / (ni * nj)) if ni > 0 and nj > 0 else 0.0
    return sim

def ds_vec_fn(ds_counts):
    """Returns a function mapping a dataset -> its usage vector across
    ALL analysts, interleaving each analyst's F-count then D-count.
    E.g. for 17 analysts, the vector has 34 entries. Two datasets are
    "similar" (high cosine) if they tend to be used, as Fact or
    Dimension, by the same analysts in the same proportions."""
    def f(d):
        v = []
        for a in ANALYSTS:
            v.append(ds_counts[a][d]['F']); v.append(ds_counts[a][d]['D'])
        return np.array(v, dtype=float)
    return f

def at_vec_fn(at_counts):
    """Same idea as ds_vec_fn, but for attributes: each attribute's
    vector interleaves every analyst's Measure-count then
    Parameter-count."""
    def f(x):
        v = []
        for a in ANALYSTS:
            v.append(at_counts[a][x]['M']); v.append(at_counts[a][x]['P'])
        return np.array(v, dtype=float)
    return f

def rank_candidates(candidates, analyst, role, own_lookup, others_lookup, sim_matrix,
                     reference_items, all_role_items, strategy, rng):
    """THE core ranking function: given a list of candidate items for
    one fold, return them sorted best-first according to `strategy`.
    Every strategy below is a different, self-contained recommender:

      - "random": no signal at all -- just shuffle. This is the
        weakest possible baseline; any real strategy should beat it.

      - "popularity": rank purely by how many OTHER analysts (not the
        current one) have used this candidate in this role. A classic
        non-personalized baseline.

      - "hybrid" (the paper's PROPOSED method): lexicographic priority
        -- first compare candidates by the current analyst's OWN
        usage count (`own`), and only break ties using OTHERS' usage
        count (`oth`), and only break further ties using maximum
        cosine similarity to `reference_items` (e.g. the session's
        chosen Fact, for a Dimension candidate). A random number is
        appended last purely as a final, reproducible tie-breaker so
        the sort is always well-defined even when everything else is
        tied at 0.

      - "knn": item-based collaborative filtering, computed from
        scratch here (see the module docstring above for the formula).
        `all_role_items` is the full candidate pool for this role
        (needed to build the analyst's "neighborhood": every item of
        that type/role the analyst has used at least once).
    """
    cand = list(candidates)
    if strategy == "random":
        rng.shuffle(cand)
        return cand

    if strategy == "knn":
        # The analyst's own "neighborhood": every item (of the same
        # type/role) they have used at least once. If this is empty
        # (a genuinely brand-new analyst), kNN has literally nothing
        # to predict from -- every candidate scores 0, and ties are
        # broken purely at random (this is the honest cold-start
        # behavior of a pure item-based kNN system with no metadata
        # fallback of its own, as noted in the module docstring).
        neighborhood = [i for i in all_role_items if own_lookup(i) > 0]
        scored = []
        for c in cand:
            if not neighborhood:
                pred = 0.0
            else:
                # weighted-sum prediction: how similar is candidate c
                # to each item i the analyst already likes, weighted
                # by how much they used i, normalized by the total
                # similarity "mass" so the score stays comparable
                # across candidates with different neighborhood sizes.
                num = sum(sim_matrix.get((c, i), 0.0) * own_lookup(i) for i in neighborhood if i != c)
                den = sum(abs(sim_matrix.get((c, i), 0.0)) for i in neighborhood if i != c)
                pred = (num / den) if den > 0 else 0.0
            scored.append(((pred, rng.random()), c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored]

    # popularity and hybrid share this loop; they only differ in what
    # sort key they build per candidate.
    scored = []
    for c in cand:
        oth = others_lookup(c)
        if strategy == "popularity":
            key = (oth, rng.random())
        else:  # hybrid
            own = own_lookup(c)
            simscore = 0.0
            if reference_items:
                # similarity to whichever reference items are already
                # "known relevant" this fold (e.g. the chosen Fact when
                # ranking Dimensions) -- take the BEST match, not the
                # average, since we only need one good analogy.
                vals = [sim_matrix.get((c, r), 0.0) for r in reference_items if r != c]
                simscore = max(vals) if vals else 0.0
            # Python tuple comparison is lexicographic: `own` decides
            # first, `oth` only breaks ties on `own`, `simscore` only
            # breaks ties on both -- exactly the paper's stated
            # priority order (own usage > others' usage > similarity).
            key = (own, oth, simscore, rng.random())
        scored.append((key, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]

def ndcg_at_k(rank, k):
    """NDCG@k for a single ground-truth item: since there is exactly
    one relevant item per fold, the ideal DCG@k is always 1 (relevant
    item at rank 1), so NDCG@k reduces to 1/log2(rank+1) if the true
    item is found within the top k, else 0."""
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def session_context(events_excl, sid):
    """All OTHER events belonging to the same OLAP session `sid`
    (after masking) -- e.g. if we're currently evaluating a Measure
    selection, this lets us find that same session's chosen Fact, so
    we know which dataset's attributes are even valid Measure
    candidates."""
    return [e for e in events_excl if e[0] == sid]

def build_fold(idx):
    """Build ONE leave-one-out fold: hide events[idx], recompute
    counts/similarity from what remains, and figure out exactly what
    the valid candidate pool and 'known relevant' reference items are
    for this fold's specific role (F/D/M/P). Returns None if this
    event can't form a valid fold (e.g. a Measure event whose
    session's Fact was itself masked/missing -- shouldn't normally
    happen here since we only mask one event at a time, but the guard
    keeps this function safe regardless)."""
    held = events[idx]
    sid, persona, rnd, itype, name, role = held
    # `events[:idx] + events[idx+1:]` is the whole log MINUS this one
    # event -- the leave-one-out masking step. Every count and
    # similarity computed below is on this masked list, so the
    # candidate ranking never "cheats" by seeing the very fact we're
    # about to test it against.
    masked_events = events[:idx] + events[idx + 1:]
    ctx = session_context(masked_events, sid)

    ds_counts, at_counts = build_counts(masked_events)
    sim_ds = cosine_sim(DATASETS, ds_vec_fn(ds_counts))
    sim_at = cosine_sim(ATTRS, at_vec_fn(at_counts))

    if role == "F":
        # Fact candidates: any of the 15 datasets. "Reference items"
        # for the hybrid similarity tier are all datasets with SOME
        # prior evidence (own or others' usage) -- there's no earlier
        # "known relevant" anchor for a Fact the way there is for a
        # Dimension (which can anchor on the session's chosen Fact).
        candidates = list(DATASETS)
        own_lookup = lambda c: ds_counts[persona][c]['F']
        others_lookup = lambda c: sum(ds_counts[a][c]['F'] for a in ANALYSTS if a != persona)
        prior_relevant = [d for d in DATASETS if ds_counts[persona][d]['F'] > 0
                           or sum(ds_counts[a][d]['F'] for a in ANALYSTS if a != persona) > 0]
        reference = prior_relevant
        sim_matrix = sim_ds
        all_role_items = list(DATASETS)
    elif role == "D":
        # Dimension candidates: any dataset EXCEPT the session's own
        # Fact and any dimension already chosen this session (you
        # wouldn't recommend re-picking the same dimension twice, or
        # picking the fact as its own dimension). Reference = the
        # session's Fact, which the hybrid strategy uses as the
        # similarity anchor ("what's structurally similar to the Fact
        # I just picked?").
        session_fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
        other_dims = {e[4] for e in ctx if e[3] == "Dataset" and e[5] == "D"}
        candidates = [d for d in DATASETS if d != session_fact and d not in other_dims]
        own_lookup = lambda c: ds_counts[persona][c]['D']
        others_lookup = lambda c: sum(ds_counts[a][c]['D'] for a in ANALYSTS if a != persona)
        reference = [session_fact] if session_fact else []
        sim_matrix = sim_ds
        all_role_items = list(DATASETS)
    elif role == "M":
        # Measure candidates: MUST belong to the same dataset as the
        # session's Fact (this is exactly the invariant the generator
        # bug once violated -- see build_usage_log.py's comments). If
        # this session has no Fact recorded at all (shouldn't happen
        # in a well-formed log), we simply can't build this fold.
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
        # Parameter candidates: symmetric to Measure, but tied to the
        # DIMENSION dataset the parameter belongs to, and only valid
        # if that dimension was actually chosen this session.
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

    # Sanity guard: the held-out item itself must be a valid member of
    # its own candidate pool (it always should be, by construction of
    # the log), and the pool must be non-empty.
    if name not in candidates or len(candidates) == 0:
        return None

    return dict(idx=idx, sid=sid, persona=persona, rnd=rnd, role=role, truth=name,
                candidates=candidates, own_lookup=own_lookup, others_lookup=others_lookup,
                sim_matrix=sim_matrix, reference=reference, all_role_items=all_role_items)

# -----------------------------------------------------------------------
# BUILD ALL FOLDS: one fold per event, EXCEPT events belonging to the
# two zero-history COLD analysts (COLD1/COLD2 never have any events
# anyway, so this `if` never actually triggers here -- it's a defensive
# guard carried over in case the log generator is ever extended to give
# them events, which would otherwise pollute the main personas'
# results with a genuinely-untestable cold-start case).
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# RUN EVERY STRATEGY ON EVERY FOLD. Each strategy gets its OWN
# random.Random(EVAL_SEED) instance so that, e.g., "random"'s shuffling
# doesn't consume random numbers that would otherwise affect
# "hybrid"'s tie-breaking -- keeping every strategy's results
# independently reproducible from the same EVAL_SEED.
# -----------------------------------------------------------------------
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

# Raw, one-row-per-(fold,strategy) results -- the finest-grained output,
# useful if you want to re-slice the data any other way later without
# rerunning the whole evaluation.
with open(f"{OUT}/per_fold_results.csv", "w", newline="") as fh:
    cols = ["idx", "sid", "persona", "role", "coldstart", "strategy", "rank", "pool_size"] + \
           [f"hit@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES]
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for r in results:
        w.writerow(r)

def summarize(rows):
    """Aggregate a list of per-fold result rows into one summary line:
    for each k, Precision@k (= hit@k / k, since there's exactly one
    relevant item per fold -- see the paper's Section 6 for why this
    equals the standard leave-one-out hit-rate formulation), Recall@k
    (= hit@k, a 0/1 indicator here), F1@k (harmonic mean of the two),
    and NDCG@k (averaged over all folds)."""
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

# One summary row per strategy (Random/Popularity/Hybrid/kNN) -- this
# is the direct source of Table `tab:strategy` and Table `tab:sensitivity`.
by_strategy = []
for strat in STRATEGIES:
    rows = [r for r in results if r["strategy"] == strat]
    s = summarize(rows)
    s["group"] = strat
    by_strategy.append(s)
write_csv(by_strategy, f"{OUT}/summary_by_strategy.csv")

# Same summary, but broken down further by role (F/D/M/P) -- this is
# the direct source of Table `tab:byrole`.
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
# Wilcoxon signed-rank test, treating every FOLD's NDCG@k as one
# (non-independent, technically -- see the persona-level test below
# for the more rigorous version) paired observation between two
# strategies. Kept for continuity with the earlier evaluate.py script.
sig_rows = []
for k in K_VALUES:
    hyb = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "hybrid"])
    pop = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "popularity"])
    rnd = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "random"])
    knn = np.array([r[f"ndcg@{k}"] for r in results if r["strategy"] == "knn"])
    def safe_w(a, b):
        # wilcoxon() raises ValueError if all paired differences are
        # zero (nothing to test); treat that degenerate case as "no
        # result" (NaN) rather than crashing the whole script.
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
# THIS is the test actually reported in the paper's text: rather than
# treating every fold as an independent sample (folds from the SAME
# persona are correlated, since they share that persona's habits), we
# first average each persona's NDCG@k across all of THEIR folds, then
# run Wilcoxon on those 15 persona-level means. This is the more
# statistically defensible unit of analysis (n=15 independent
# personas, not n=hundreds of non-independent folds).
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
# Grouped bar chart: Precision@5 and NDCG@5 side by side for each of
# the four strategies. This is the exact PNG embedded in the paper.
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
    """Turn a list of dict rows into a GitHub-flavoured Markdown table
    string -- purely for the human-readable results_readable.md report
    below, not used by anything numeric."""
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)

cols_main = ["group", "n_folds"] + [f"{m}@{k}" for k in K_VALUES for m in ["P", "R", "F1", "NDCG"]]

# One consolidated, paste-ready Markdown report combining every table
# above -- convenient for pasting straight into a lab notebook, an
# issue, or an email, without opening each CSV individually.
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
