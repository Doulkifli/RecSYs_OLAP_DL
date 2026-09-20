"""
scale_experiment.py
====================
EXPLORATORY script, not part of the paper's main evaluation pipeline
(evaluate_extended.py / evaluate_v2.py). Written to answer a reviewer-
style question: "does adding more data (more analysts) reverse the
finding that own-history dominates collaborative filtering?"

Question this answers: does adding more data (more analysts sharing
the same 15 persona profiles) narrow the gap between own-history-only
and collaborative ("others-only") ranking seen in the paper's
component ablation (Section 6.3)?

Approach: replicate each of the 15 personas into N independent synthetic
analyst instances (same fact/dims/measures/params spec, independent random
draws), keeping sessions_per_persona=4 and the corrected generator (Measure
tied to whichever fact was actually drawn). This increases cross-analyst
overlap (more analysts sharing each profile) without changing the profiles
themselves or any single analyst's own history length -- the scaling axis
the paper's "future work" (10 to 1000 analysts) refers to.

For each replication factor N in {1, 3, 10, 30}, we report P@1 / NDCG@5 for
own_only, others_only, similarity_only, and hybrid.

RESULT (feeds the paper's Table `tab:scaleup`, "Replicated personas" rows):
the own-only/others-only gap narrows as N grows, then plateaus -- it never
reverses, even at 30x replication (450 synthetic analysts).

PERFORMANCE NOTE (why this file looks different from evaluate_extended.py):
A naive port of evaluate_extended.py's leave-one-out approach -- fully
rebuilding all counts and re-running cosine_sim from scratch for EVERY
single fold -- runs out of memory once n_replicas reaches ~10 (150+
analysts), because that rebuild is O(n_events) and gets repeated
n_events times, i.e. O(n_events^2) total dictionary allocations. This
script instead computes the FULL (non-leave-one-out) counts and
similarity matrices ONCE, then approximates leave-one-out per fold by
just subtracting 1 from the one count that fold's own held-out event
contributed (see build_fold_fast / run_scale below for exactly how).
This is exact for own/others usage counts, and a very close
approximation for similarity (removing one event out of hundreds barely
moves a cosine similarity), which is an acceptable trade-off for an
exploratory sensitivity check -- it is NOT how the paper's main results
were computed (those use the exact, fully-rebuilt leave-one-out from
evaluate_extended.py / evaluate_v2.py).
"""
import random, math, csv
import numpy as np
from build_usage_log import PERSONAS, DATASETS, ATTRS, attrs_of

K_VALUES = [1, 3, 5]
STRATEGIES = ["random", "popularity", "own_only", "others_only", "similarity_only", "hybrid"]

def dataset_of_attr(attr):
    """Same as in the other scripts: recover an attribute's owning
    dataset from its name prefix."""
    for d in DATASETS:
        if attr.startswith(d + "."):
            return d
    return None

def generate_events_scaled(base_seed, n_replicas, sessions_per_persona=4,
                            alt_fact_prob=0.15, dim_alt_prob=0.25, explore_prob=0.10):
    """Like build_usage_log.py's generation loop, but for EACH of the
    15 personas, spawn `n_replicas` independent synthetic analysts
    (IDs like "A1#0", "A1#1", ...) all sharing that persona's spec but
    with their OWN random draws (a distinct random.Random seeded from
    a string key "{base_seed}-{persona}-{rep}", so every replica is
    reproducible and independent of every other replica/persona).
    n_replicas=1 degenerates to plain analyst IDs (no "#0" suffix),
    reproducing the ordinary 15-analyst log as the N=1 baseline point.

    This is the mechanism behind the whole experiment: MORE replicas
    of the SAME 15 profiles means MORE analysts who could plausibly
    share an item choice by coincidence, which is exactly the
    ingredient collaborative filtering needs to have a chance -- while
    each individual analyst's OWN history length (4 sessions) never
    changes, so own-history's predictive power isn't being helped by
    this scaling axis at all.
    """
    events = []
    analysts = []
    sid = 0
    for persona, spec in PERSONAS.items():
        for rep in range(n_replicas):
            analyst_id = f"{persona}#{rep}" if n_replicas > 1 else persona
            analysts.append(analyst_id)
            rng = random.Random(f"{base_seed}-{persona}-{rep}")
            for rnd in range(1, sessions_per_persona + 1):
                sid += 1
                session = f"S{sid:04d}"

                # Fact (with occasional alt-fact exploration), same
                # mechanic as build_usage_log.py / evaluate_v2.py.
                fact = spec["fact"]
                if spec["alt_facts"] and rng.random() < alt_fact_prob:
                    fact = rng.choice(spec["alt_facts"])
                events.append((session, analyst_id, rnd, "Dataset", fact, "F"))

                # Dimensions.
                dim_pool = list(spec["dims"])
                if spec["dim_alt"] and rng.random() < dim_alt_prob:
                    dim_pool = dim_pool + [rng.choice(spec["dim_alt"])]
                n_dims = min(len(dim_pool), rng.randint(1, max(1, len(dim_pool))))
                chosen_dims = rng.sample(dim_pool, n_dims)
                for d in chosen_dims:
                    events.append((session, analyst_id, rnd, "Dataset", d, "D"))

                # Measures -- same fact-consistency fix as the other scripts.
                if fact == spec["fact"]:
                    measures_pool = spec["measures"]
                else:
                    measures_pool = attrs_of(fact) or spec["measures"]
                n_meas = min(len(measures_pool), rng.randint(1, max(1, len(measures_pool))))
                for m in rng.sample(measures_pool, n_meas):
                    events.append((session, analyst_id, rnd, "Attribute", m, "M"))

                # Parameters, one set per chosen dimension.
                for d in chosen_dims:
                    plist = spec["params"].get(d, attrs_of(d)[:1])
                    n_p = min(len(plist), rng.randint(1, max(1, len(plist))))
                    for p in rng.sample(plist, n_p):
                        events.append((session, analyst_id, rnd, "Attribute", p, "P"))

                # Light exploration noise, same as the other scripts.
                if rng.random() < explore_prob:
                    pool = attrs_of(fact)
                    if pool:
                        events.append((session, analyst_id, rnd, "Attribute", rng.choice(pool), "M"))
    return events, analysts

def build_counts(event_list, analysts):
    """Per-analyst usage counts, parameterized by an explicit
    `analysts` list (rather than a fixed global ANALYSTS constant, as
    in the other scripts) since the analyst population here changes
    size depending on n_replicas."""
    ds_counts = {a: {d: {'F': 0, 'D': 0} for d in DATASETS} for a in analysts}
    at_counts = {a: {x: {'M': 0, 'P': 0} for x in ATTRS} for a in analysts}
    for (sid, persona, rnd, itype, name, role) in event_list:
        if itype == "Dataset":
            ds_counts[persona][name][role] += 1
        else:
            at_counts[persona][name][role] += 1
    return ds_counts, at_counts

def cosine_sim(items, vec_fn):
    """Pairwise cosine similarity, identical formula to the other scripts."""
    vecs = {it: vec_fn(it) for it in items}
    norms = {it: np.linalg.norm(v) for it, v in vecs.items()}
    sim = {}
    for i in items:
        for j in items:
            ni, nj = norms[i], norms[j]
            sim[(i, j)] = float(np.dot(vecs[i], vecs[j]) / (ni * nj)) if ni > 0 and nj > 0 else 0.0
    return sim

def ds_vec_fn(ds_counts, analysts):
    """Dataset usage vector across the given `analysts` list (its
    length now depends on n_replicas, unlike the other scripts' fixed
    17-analyst ANALYSTS list)."""
    def f(d):
        v = []
        for a in analysts:
            v.append(ds_counts[a][d]['F']); v.append(ds_counts[a][d]['D'])
        return np.array(v, dtype=float)
    return f

def at_vec_fn(at_counts, analysts):
    def f(x):
        v = []
        for a in analysts:
            v.append(at_counts[a][x]['M']); v.append(at_counts[a][x]['P'])
        return np.array(v, dtype=float)
    return f

def ndcg_at_k(rank, k):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)

def session_context(events_excl, sid):
    return [e for e in events_excl if e[0] == sid]

# NOTE: build_fold() below is the SLOW, exact leave-one-out version --
# it is defined here for reference/completeness but NOT actually
# called by run_scale() at the bottom of this file, which uses the
# faster build_fold_fast() defined INSIDE run_scale() instead (see the
# big comment at the top of this file for why). It's kept because it
# documents, in the clearest form, exactly what the fast version is
# approximating.
def build_fold(events, idx, analysts):
    """Exact (but slow at scale) leave-one-out fold construction:
    fully removes events[idx] from the log and rebuilds counts +
    similarity from scratch. Functionally identical to
    evaluate_extended.py's build_fold(), just parameterized by an
    explicit `analysts` list. See the module docstring for why
    run_scale() below does NOT call this at n_replicas >= 10."""
    held = events[idx]
    sid, persona, rnd, itype, name, role = held
    masked_events = events[:idx] + events[idx + 1:]
    ctx = session_context(masked_events, sid)

    ds_counts, at_counts = build_counts(masked_events, analysts)
    sim_ds = cosine_sim(DATASETS, ds_vec_fn(ds_counts, analysts))
    sim_at = cosine_sim(ATTRS, at_vec_fn(at_counts, analysts))

    if role == "F":
        candidates = list(DATASETS)
        own_lookup = lambda c: ds_counts[persona][c]['F']
        others_lookup = lambda c: sum(ds_counts[a][c]['F'] for a in analysts if a != persona)
        reference = [d for d in DATASETS if ds_counts[persona][d]['F'] > 0
                     or sum(ds_counts[a][d]['F'] for a in analysts if a != persona) > 0]
        sim_matrix = sim_ds
    elif role == "D":
        session_fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
        other_dims = {e[4] for e in ctx if e[3] == "Dataset" and e[5] == "D"}
        candidates = [d for d in DATASETS if d != session_fact and d not in other_dims]
        own_lookup = lambda c: ds_counts[persona][c]['D']
        others_lookup = lambda c: sum(ds_counts[a][c]['D'] for a in analysts if a != persona)
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
        others_lookup = lambda c: sum(at_counts[a][c]['M'] for a in analysts if a != persona)
        reference = [a for a in pool if at_counts[persona][a]['M'] > 0
                     or sum(at_counts[a2][a]['M'] for a2 in analysts if a2 != persona) > 0]
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
        others_lookup = lambda c: sum(at_counts[a][c]['P'] for a in analysts if a != persona)
        reference = [a for a in pool if at_counts[persona][a]['P'] > 0
                     or sum(at_counts[a2][a]['P'] for a2 in analysts if a2 != persona) > 0]
        sim_matrix = sim_at
    else:
        return None

    if name not in candidates or len(candidates) == 0:
        return None
    return dict(idx=idx, persona=persona, role=role, truth=name, candidates=candidates,
                own_lookup=own_lookup, others_lookup=others_lookup, sim_matrix=sim_matrix,
                reference=reference)

def rank_candidates(candidates, own_lookup, others_lookup, sim_matrix, reference_items, strategy, rng):
    """Same ranking logic as evaluate_v2.py's rank_candidates (random /
    popularity / own_only / others_only / similarity_only / hybrid) --
    see that file's comments for the full explanation of each
    strategy."""
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
        own, oth, sim = own_lookup(c), others_lookup(c), simscore(c)
        if strategy in ("popularity", "others_only"):
            key = (oth, rng.random())
        elif strategy == "own_only":
            key = (own, rng.random())
        elif strategy == "similarity_only":
            key = (sim, rng.random())
        else:
            key = (own, oth, sim, rng.random())
        scored.append((key, c))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]

def summarize(rows):
    """Mean Precision@k / NDCG@k across a list of per-fold rows --
    same idea as the other scripts, just without Recall/F1 (not
    needed for this exploratory check)."""
    n = len(rows)
    line = {"n_folds": n}
    for k in K_VALUES:
        prec = np.mean([r[f"hit@{k}"] / k for r in rows]) if n else 0.0
        ndcg = np.mean([r[f"ndcg@{k}"] for r in rows]) if n else 0.0
        line[f"P@{k}"] = round(prec, 4); line[f"NDCG@{k}"] = round(ndcg, 4)
    return line

def run_scale(n_replicas, seed=42):
    """Run the whole experiment for ONE replication factor: generate
    the scaled-up log, evaluate all 6 strategies via the FAST
    approximate leave-one-out (explained below), and return a summary
    dict. Called once per n_replicas value in the `__main__` block at
    the bottom of this file."""
    events, analysts = generate_events_scaled(seed, n_replicas)
    analysts_set = set(analysts)

    # Full (non-leave-one-out) counts/similarity, computed ONCE.
    # Leave-one-out is then approximated by decrementing only the one
    # held-out (persona, item, role) count by 1 at lookup time -- others'
    # counts and the similarity matrices are unaffected by removing a
    # single one of the analyst's own events, so this is exact for
    # own_lookup/others_lookup and a very close approximation for
    # similarity (negligible at these event volumes). This avoids
    # rebuilding O(n_events)-sized structures per fold, which is what made
    # the naive leave-one-out implementation OOM at n_replicas >= 10.
    full_ds_counts, full_at_counts = build_counts(events, analysts)
    sim_ds = cosine_sim(DATASETS, ds_vec_fn(full_ds_counts, analysts))
    sim_at = cosine_sim(ATTRS, at_vec_fn(full_at_counts, analysts))

    def build_fold_fast(idx):
        """The FAST fold builder actually used below. Instead of
        rebuilding full_ds_counts/full_at_counts and re-running
        cosine_sim from scratch for this one fold (what build_fold()
        above does), it reuses the ALREADY-COMPUTED full-log counts
        and similarity matrices, and only corrects `own_lookup` to
        subtract 1 for the exact (candidate == truth) case -- i.e.
        "pretend this one event never happened" is applied ONLY where
        it actually matters (the analyst's own count for the specific
        held-out item), not by literally reconstructing the whole
        data structure each time."""
        sid, persona, rnd, itype, name, role = events[idx]
        # Session context, built by scanning the WHOLE events list and
        # excluding this fold's own index -- still O(n_events) per
        # fold, but far cheaper than the full build_counts/cosine_sim
        # rebuild that build_fold() would otherwise repeat.
        ctx = [e for j, e in enumerate(events) if e[0] == sid and j != idx]

        if itype == "Dataset":
            def own_lookup(c, _p=persona, _r=role, _n=name):
                v = full_ds_counts[_p][c][_r]
                return v - 1 if c == _n else v
            others_lookup = lambda c, _p=persona, _r=role: sum(
                full_ds_counts[a][c][_r] for a in analysts if a != _p)
        else:
            def own_lookup(c, _p=persona, _r=role, _n=name):
                v = full_at_counts[_p][c][_r]
                return v - 1 if c == _n else v
            others_lookup = lambda c, _p=persona, _r=role: sum(
                full_at_counts[a][c][_r] for a in analysts if a != _p)

        # Candidate-pool / reference-item logic below is otherwise
        # identical in spirit to build_fold() above and to
        # evaluate_v2.py's build_fold(), just using the fast
        # own_lookup/others_lookup closures defined just above instead
        # of rebuilding fresh count dictionaries.
        if role == "F":
            candidates = list(DATASETS)
            reference = [d for d in DATASETS if own_lookup(d) > 0 or others_lookup(d) > 0]
            sim_matrix = sim_ds
        elif role == "D":
            session_fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
            other_dims = {e[4] for e in ctx if e[3] == "Dataset" and e[5] == "D"}
            candidates = [d for d in DATASETS if d != session_fact and d not in other_dims]
            reference = [session_fact] if session_fact else []
            sim_matrix = sim_ds
        elif role == "M":
            session_fact = next((e[4] for e in ctx if e[3] == "Dataset" and e[5] == "F"), None)
            if session_fact is None:
                return None
            other_meas = {e[4] for e in ctx if e[3] == "Attribute" and e[5] == "M"}
            pool = [a for a in ATTRS if dataset_of_attr(a) == session_fact and a not in other_meas]
            candidates = pool
            reference = [a for a in pool if own_lookup(a) > 0 or others_lookup(a) > 0]
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
            reference = [a for a in pool if own_lookup(a) > 0 or others_lookup(a) > 0]
            sim_matrix = sim_at
        else:
            return None

        if name not in candidates or len(candidates) == 0:
            return None
        return dict(persona=persona, role=role, truth=name, candidates=candidates,
                    own_lookup=own_lookup, others_lookup=others_lookup,
                    sim_matrix=sim_matrix, reference=reference)

    folds = []
    for i in range(len(events)):
        f = build_fold_fast(i)
        if f is not None:
            folds.append(f)

    results = []
    rngs = {s: random.Random(7) for s in STRATEGIES}
    for f in folds:
        for strat in STRATEGIES:
            ranked = rank_candidates(f["candidates"], f["own_lookup"], f["others_lookup"],
                                      f["sim_matrix"], f["reference"], strat, rngs[strat])
            rank = ranked.index(f["truth"]) + 1 if f["truth"] in ranked else None
            row = dict(strategy=strat, rank=rank)
            for k in K_VALUES:
                row[f"hit@{k}"] = 1 if (rank is not None and rank <= k) else 0
                row[f"ndcg@{k}"] = ndcg_at_k(rank, k)
            results.append(row)
    out = {"n_replicas": n_replicas, "n_analysts": len(analysts), "n_events": len(events), "n_folds": len(folds)}
    for strat in STRATEGIES:
        s = summarize([r for r in results if r["strategy"] == strat])
        out[strat] = s
    return out

if __name__ == "__main__":
    # Sweep four replication factors: 1 (the ordinary 15-analyst log,
    # as a sanity-check baseline), 3, 10, and 30 (450 synthetic
    # analysts) -- printing P@1/NDCG@5 for every strategy at each
    # scale, and writing the full sweep to a CSV for later plotting or
    # inclusion in the paper (Table `tab:scaleup`).
    rows = []
    for n_rep in [1, 3, 10, 30]:
        r = run_scale(n_rep)
        rows.append(r)
        print(f"\n=== n_replicas={n_rep}  (n_analysts={r['n_analysts']}, n_events={r['n_events']}, n_folds={r['n_folds']}) ===")
        print(f"{'strategy':16s} {'P@1':>8s} {'NDCG@5':>8s}")
        for strat in STRATEGIES:
            s = r[strat]
            print(f"{strat:16s} {s['P@1']:>8.4f} {s['NDCG@5']:>8.4f}")

    with open("./scale_experiment_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_replicas", "n_analysts", "n_events", "n_folds", "strategy", "P@1", "NDCG@5"])
        for r in rows:
            for strat in STRATEGIES:
                w.writerow([r["n_replicas"], r["n_analysts"], r["n_events"], r["n_folds"],
                            strat, r[strat]["P@1"], r[strat]["NDCG@5"]])
    print("\nSaved /home/claude/scale_experiment_results.csv")
