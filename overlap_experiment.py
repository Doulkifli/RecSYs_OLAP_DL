"""
Overlapping-personas experiment (follow-up to scale_experiment.py).

scale_experiment.py replicated each of the 15 *existing* personas into many
copies -- same profile, independent draws. That narrowed but did not close
the own-only vs. others-only gap, and it plateaued.

This script instead builds a small number of "families" that share a fixed
fact and a SMALL shared pool of candidate dimensions/measures/parameters.
Each family's analysts are independently random, but because the pool they
draw from is small, distinct analysts frequently land on the exact same
dimension/measure/parameter by chance -- genuine content overlap, not
persona replication. This is the condition under which collaborative
filtering is supposed to shine.
"""
import random, math, csv
import numpy as np
from build_usage_log import DATASETS, ATTRS, attrs_of

K_VALUES = [1, 3, 5]
STRATEGIES = ["random", "popularity", "own_only", "others_only", "similarity_only", "hybrid"]

def dataset_of_attr(attr):
    for d in DATASETS:
        if attr.startswith(d + "."):
            return d
    return None

# Four shared-pool families: fixed fact, small shared dims/measures pools.
FAMILIES = {
    "USERS": dict(fact="SE_mysql.Users",
                  dims_pool=["SE_mysql.postes", "SE_mysql.Votes", "SE_mongodb.Comments"],
                  measures_pool=attrs_of("SE_mysql.Users")),
    "POSTS_DOC": dict(fact="SE_mongodb.Posts",
                       dims_pool=["SE_neo4j.Tags", "SE_mysql.Posttypes", "SE_neo4j.Posts"],
                       measures_pool=attrs_of("SE_mongodb.Posts")),
    "VOTES": dict(fact="SE_mysql.Votes",
                   dims_pool=["SE_mysql.Votetypes", "SE_mysql.postes", "SE_mysql.Flagtypes"],
                   measures_pool=attrs_of("SE_mysql.Votes")),
    "POSTS_GRAPH": dict(fact="SE_neo4j.Posts",
                         dims_pool=["SE_neo4j.postlinks", "SE_neo4j.Tags", "SE_neo4j.Comments"],
                         measures_pool=attrs_of("SE_neo4j.Posts")),
}

def generate_events_overlap(seed, n_per_family, sessions_per_analyst=4,
                             dim_explore_prob=0.20, meas_explore_prob=0.15):
    events = []
    analysts = []
    sid = 0
    for fam_name, fam in FAMILIES.items():
        for rep in range(n_per_family):
            analyst_id = f"{fam_name}#{rep}"
            analysts.append(analyst_id)
            rng = random.Random(f"{seed}-{fam_name}-{rep}")

            n_dims_fixed = rng.randint(1, min(2, len(fam["dims_pool"])))
            my_dims = rng.sample(fam["dims_pool"], n_dims_fixed)
            n_meas_fixed = rng.randint(1, min(2, len(fam["measures_pool"])))
            my_measures = rng.sample(fam["measures_pool"], n_meas_fixed)
            my_params = {}
            for d in my_dims:
                plist = attrs_of(d)[:3] or attrs_of(d)
                if plist:
                    n_p = rng.randint(1, min(2, len(plist)))
                    my_params[d] = rng.sample(plist, n_p)

            for rnd in range(1, sessions_per_analyst + 1):
                sid += 1
                session = f"S{sid:05d}"
                fact = fam["fact"]
                events.append((session, analyst_id, rnd, "Dataset", fact, "F"))

                session_dims = list(my_dims)
                # Occasional exploration: swap in one dim from the family
                # pool outside my_dims (adds realistic session-to-session
                # noise so own-history is not a trivially perfect signal).
                rest = [d for d in fam["dims_pool"] if d not in my_dims]
                if rest and rng.random() < dim_explore_prob:
                    session_dims = session_dims + [rng.choice(rest)]
                for d in session_dims:
                    events.append((session, analyst_id, rnd, "Dataset", d, "D"))

                session_meas = list(my_measures)
                rest_m = [m for m in fam["measures_pool"] if m not in my_measures]
                if rest_m and rng.random() < meas_explore_prob:
                    session_meas = session_meas + [rng.choice(rest_m)]
                for m in session_meas:
                    events.append((session, analyst_id, rnd, "Attribute", m, "M"))

                for d in session_dims:
                    plist = my_params.get(d) or (attrs_of(d)[:3] or attrs_of(d))
                    if plist:
                        for p in plist:
                            events.append((session, analyst_id, rnd, "Attribute", p, "P"))
    return events, analysts

# ---- shared machinery (same approach as scale_experiment.py) ----
def build_counts(event_list, analysts):
    ds_counts = {a: {d: {'F': 0, 'D': 0} for d in DATASETS} for a in analysts}
    at_counts = {a: {x: {'M': 0, 'P': 0} for x in ATTRS} for a in analysts}
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

def ds_vec_fn(ds_counts, analysts):
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
    n = len(rows)
    line = {"n_folds": n}
    for k in K_VALUES:
        prec = np.mean([r[f"hit@{k}"] / k for r in rows]) if n else 0.0
        ndcg = np.mean([r[f"ndcg@{k}"] for r in rows]) if n else 0.0
        line[f"P@{k}"] = round(prec, 4); line[f"NDCG@{k}"] = round(ndcg, 4)
    return line

def run_overlap(n_per_family, seed=42):
    events, analysts = generate_events_overlap(seed, n_per_family)
    full_ds_counts, full_at_counts = build_counts(events, analysts)
    sim_ds = cosine_sim(DATASETS, ds_vec_fn(full_ds_counts, analysts))
    sim_at = cosine_sim(ATTRS, at_vec_fn(full_at_counts, analysts))

    def build_fold_fast(idx):
        sid, persona, rnd, itype, name, role = events[idx]
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
    out = {"n_per_family": n_per_family, "n_analysts": len(analysts), "n_events": len(events), "n_folds": len(folds)}
    for strat in STRATEGIES:
        s = summarize([r for r in results if r["strategy"] == strat])
        out[strat] = s
    return out

if __name__ == "__main__":
    rows = []
    for n_pf in [5, 20, 50, 150]:
        r = run_overlap(n_pf)
        rows.append(r)
        print(f"\n=== n_per_family={n_pf}  (n_analysts={r['n_analysts']}, n_events={r['n_events']}, n_folds={r['n_folds']}) ===")
        print(f"{'strategy':16s} {'P@1':>8s} {'NDCG@5':>8s}")
        for strat in STRATEGIES:
            s = r[strat]
            print(f"{strat:16s} {s['P@1']:>8.4f} {s['NDCG@5']:>8.4f}")

    with open("/home/claude/overlap_experiment_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n_per_family", "n_analysts", "n_events", "n_folds", "strategy", "P@1", "NDCG@5"])
        for r in rows:
            for strat in STRATEGIES:
                w.writerow([r["n_per_family"], r["n_analysts"], r["n_events"], r["n_folds"],
                            strat, r[strat]["P@1"], r[strat]["NDCG@5"]])
    print("\nSaved /home/claude/overlap_experiment_results.csv")
