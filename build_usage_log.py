"""
build_usage_log.py
===================
Generates the seeded, deterministic, persona-driven SIMULATED usage log
that stands in for organically-collected multi-analyst usage data
(Section 6.1 of the paper). It also writes that log, plus derived
count/similarity matrices, into an Excel workbook so a human reviewer
can inspect it directly (open OLAP_Recommender_UsageLog.xlsx).

High-level idea: we invent 15 "personas" (synthetic analysts), each with
a fixed analytical habit (a typical Fact dataset, typical Dimensions,
typical Measures, typical Dimension-Parameters), and let each persona
run 4 simulated OLAP-cube-design sessions. Each session mostly repeats
the persona's typical choices, with small amounts of randomness
(occasionally picking an alternate fact, an extra dimension, an extra
measure) so the log isn't perfectly deterministic/trivial.

Output: a list of "events" -- one row per (session, analyst, item,
role) selection -- which is exactly what the evaluation scripts
(evaluate_extended.py, evaluate_v2.py) read via
`from build_usage_log import events, ANALYSTS, DATASETS, ATTRS,
PERSONAS, COLD, attrs_of`.
"""
import random, openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment

# Fixing the seed makes every run of this script byte-for-byte
# reproducible: same personas + same seed = same log, always. This is
# what "seed = 42" in the paper refers to.
SEED = 42
random.seed(SEED)

# ---------------------------------------------------------------------
# The polystore's candidate items. These mirror the actual StackExchange
# polystore schema described in Section 5 (Implementation): 15 datasets
# spread across the three stores (SE_mysql.*, SE_mongodb.*, SE_neo4j.*),
# playing the role of candidate Facts/Dimensions in the recommender.
# ---------------------------------------------------------------------
DATASETS = ["SE_mysql.postes","SE_mysql.Posttypes","SE_mysql.Users","SE_mysql.Flagtypes",
    "SE_mysql.Votes","SE_mysql.Votetypes","SE_mongodb.Badges","SE_mongodb.Posts",
    "SE_mongodb.Users","SE_mongodb.Comments","SE_neo4j.postlinks","SE_neo4j.Posts",
    "SE_neo4j.Comments","SE_neo4j.Users","SE_neo4j.Tags"]

# 63 candidate attributes ("dataset.AttributeName"), playing the role of
# candidate Measures/Dimension-Parameters. Each attribute's owning
# dataset is recoverable from its name prefix (see attrs_of/dataset_of
# below) -- this naming convention is how the whole codebase knows
# "which dataset does this attribute belong to" without a separate
# lookup table.
ATTRS = ["SE_mysql.postes.Id","SE_mysql.postes.PostTypeId","SE_mysql.postes.Title",
    "SE_mysql.postes.OwnerUserId","SE_mysql.postes.ViewCount","SE_mysql.postes.LastEditDate",
    "SE_mysql.postes.LastEditorUserId","SE_mysql.postes.CreationDate","SE_mysql.Posttypes.Id",
    "SE_mysql.Posttypes.Name","SE_mysql.Users.Id","SE_mysql.Users.Views","SE_mysql.Users.UpVotes",
    "SE_mysql.Users.DisplayName","SE_mysql.Users.DownVotes","SE_mysql.Users.LastAccessDate",
    "SE_mysql.Users.Reputation","SE_mysql.Flagtypes.Id","SE_mysql.Flagtypes.Name","SE_mysql.Votes.Id",
    "SE_mysql.Votes.UserId","SE_mysql.Votes.BountyAmount","SE_mysql.Votes.CreationDate",
    "SE_mysql.Votes.PostId","SE_mysql.Votes.VoteTypeId","SE_mysql.Votetypes.Id","SE_mysql.Votetypes.Name",
    "SE_mongodb.Badges.Id","SE_mongodb.Badges.UserId","SE_mongodb.Badges.Name","SE_mongodb.Badges.Date",
    "SE_mongodb.Posts.Id","SE_mongodb.Posts.Tags","SE_mongodb.Posts.Body","SE_mongodb.Users.Id",
    "SE_mongodb.Users.DisplayName","SE_mongodb.Users.AboutMe","SE_mongodb.Comments.Id",
    "SE_mongodb.Comments.UserId","SE_mongodb.Comments.Text","SE_mongodb.Comments.PostId",
    "SE_neo4j.postlinks.id","SE_neo4j.postlinks.LinkTypeId","SE_neo4j.postlinks.RelatedPostId",
    "SE_neo4j.postlinks.PostIdd","SE_neo4j.postlinks.CreationDate","SE_neo4j.Posts.id",
    "SE_neo4j.Posts.posttypeid","SE_neo4j.Posts.parentid","SE_neo4j.Posts.owneruserid",
    "SE_neo4j.Posts.acceptedanswerid","SE_neo4j.Comments.id","SE_neo4j.Comments.creationdate",
    "SE_neo4j.Comments.score","SE_neo4j.Users.id","SE_neo4j.Users.accountid","SE_neo4j.Users.displayname",
    "SE_neo4j.Users.creationdate","SE_neo4j.Tags.id","SE_neo4j.Tags.count","SE_neo4j.Tags.wikipostid",
    "SE_neo4j.Tags.tagname","SE_neo4j.Tags.excerptpostid"]

def attrs_of(ds):
    """All attributes belonging to dataset `ds` (matched by name prefix
    'ds.'). Used constantly: to build a persona's Measure/Parameter
    pools, and by the evaluator to know which attributes are even
    *candidates* for a given Fact/Dimension."""
    return [a for a in ATTRS if a.startswith(ds + ".")]

def find(ds, substr):
    """Convenience lookup used only while WRITING the PERSONAS table
    below: 'give me the one attribute of dataset ds whose name contains
    substr' (case-insensitive), e.g. find("SE_mysql.Users","Reputation")
    -> "SE_mysql.Users.Reputation". Raises if no match, so a typo in a
    persona definition fails loudly at import time instead of silently
    producing an empty pool."""
    for a in attrs_of(ds):
        if substr.lower() in a.lower():
            return a
    raise ValueError(f"not found: {ds}.{substr}")

# ---------------------------------------------------------------------
# PERSONAS: the heart of the simulation. Each of the 15 keys (A1, A2,
# ..., X2) is one synthetic analyst with a FIXED analytical habit:
#   - cluster:    a thematic grouping label (A/B/C/D/E/X), not used by
#                 the generation logic itself, just documentation/
#                 grouping for humans reading this file.
#   - fact:       the dataset this persona typically picks as OLAP Fact.
#   - alt_facts:  dataset(s) this persona OCCASIONALLY picks instead of
#                 `fact` (see the 15% branch below) -- models a bit of
#                 realistic session-to-session variation.
#   - dims:       the dataset(s) this persona typically picks as OLAP
#                 Dimension(s).
#   - dim_alt:    an occasional EXTRA dimension (see the 25% branch
#                 below), added on top of `dims`, not a replacement.
#   - measures:   this persona's typical Measure attribute(s) -- must
#                 belong to `fact`'s dataset (find() enforces this by
#                 construction, since it looks up attrs_of(fact)).
#   - params:     a dict {dimension_dataset: [attribute, ...]} giving
#                 this persona's typical Dimension-Parameter(s) for
#                 each of its dims.
# ---------------------------------------------------------------------
PERSONAS = {
 "A1": dict(cluster="A", fact="SE_mysql.Users",
    alt_facts=["SE_mysql.postes"],
    dims=["SE_mysql.Votes","SE_mysql.postes"], dim_alt=["SE_mysql.Posttypes"],
    measures=[find("SE_mysql.Users","Reputation"), find("SE_mysql.Users","UpVotes")],
    params={"SE_mysql.Votes":[find("SE_mysql.Votes","CreationDate"), find("SE_mysql.Votes","VoteTypeId")],
            "SE_mysql.postes":[find("SE_mysql.postes","CreationDate")]}),
 "A2": dict(cluster="A", fact="SE_mysql.Users",
    alt_facts=["SE_mongodb.Users"],
    dims=["SE_mysql.postes","SE_mongodb.Comments"], dim_alt=["SE_mysql.Votes"],
    measures=[find("SE_mysql.Users","Views"), find("SE_mysql.Users","Reputation")],
    params={"SE_mysql.postes":[find("SE_mysql.postes","CreationDate")],
            "SE_mongodb.Comments":[find("SE_mongodb.Comments","PostId")]}),
 "A3": dict(cluster="A", fact="SE_neo4j.Users",
    alt_facts=["SE_mongodb.Users","SE_mysql.Users"],
    dims=["SE_mongodb.Users","SE_mysql.Users"], dim_alt=[],
    measures=[find("SE_neo4j.Users","creationdate"), find("SE_mysql.Users","Reputation")],
    params={"SE_mongodb.Users":[find("SE_mongodb.Users","DisplayName")],
            "SE_mysql.Users":[find("SE_mysql.Users","DisplayName")]}),
 "B1": dict(cluster="B", fact="SE_neo4j.Tags",
    alt_facts=["SE_neo4j.Posts"],
    dims=["SE_neo4j.Posts"], dim_alt=["SE_mongodb.Posts"],
    measures=[find("SE_neo4j.Tags","count")],
    params={"SE_neo4j.Posts":[find("SE_neo4j.Posts","posttypeid")]}),
 "B2": dict(cluster="B", fact="SE_mongodb.Posts",
    alt_facts=["SE_neo4j.Posts"],
    dims=["SE_neo4j.Tags"], dim_alt=["SE_mysql.Posttypes"],
    measures=[find("SE_mongodb.Posts","Tags"), find("SE_mongodb.Posts","Body")],
    params={"SE_neo4j.Tags":[find("SE_neo4j.Tags","tagname"), find("SE_neo4j.Tags","count")]}),
 "B3": dict(cluster="B", fact="SE_neo4j.postlinks",
    alt_facts=["SE_neo4j.Posts"],
    dims=["SE_neo4j.Posts"], dim_alt=[],
    measures=[find("SE_neo4j.postlinks","LinkTypeId")],
    params={"SE_neo4j.Posts":[find("SE_neo4j.Posts","id")]}),
 "C1": dict(cluster="C", fact="SE_mysql.Votes",
    alt_facts=["SE_mysql.postes"],
    dims=["SE_mysql.Votetypes","SE_mysql.postes"], dim_alt=[],
    measures=[find("SE_mysql.Votes","BountyAmount")],
    params={"SE_mysql.Votetypes":[find("SE_mysql.Votetypes","Name")],
            "SE_mysql.postes":[find("SE_mysql.postes","CreationDate")]}),
 "C2": dict(cluster="C", fact="SE_mongodb.Comments",
    alt_facts=["SE_neo4j.Comments"],
    dims=["SE_mongodb.Posts"], dim_alt=["SE_neo4j.Posts"],
    measures=[find("SE_mongodb.Comments","Text")],
    params={"SE_mongodb.Posts":[find("SE_mongodb.Posts","Id")]}),
 "C3": dict(cluster="C", fact="SE_mysql.postes",
    alt_facts=["SE_neo4j.Posts"],
    dims=["SE_mysql.Users","SE_mysql.Posttypes"], dim_alt=[],
    measures=[find("SE_mysql.postes","ViewCount")],
    params={"SE_mysql.Users":[find("SE_mysql.Users","Reputation")],
            "SE_mysql.Posttypes":[find("SE_mysql.Posttypes","Name")]}),
 "D1": dict(cluster="D", fact="SE_mysql.Flagtypes",
    alt_facts=[],
    dims=["SE_mysql.postes"], dim_alt=[],
    measures=[find("SE_mysql.Flagtypes","Name")],
    params={"SE_mysql.postes":[find("SE_mysql.postes","OwnerUserId")]}),
 "D2": dict(cluster="D", fact="SE_mongodb.Badges",
    alt_facts=[],
    dims=["SE_mongodb.Users"], dim_alt=["SE_mysql.Users"],
    measures=[find("SE_mongodb.Badges","Name"), find("SE_mongodb.Badges","Date")],
    params={"SE_mongodb.Users":[find("SE_mongodb.Users","DisplayName")]}),
 "E1": dict(cluster="E", fact="SE_mysql.postes",
    alt_facts=["SE_neo4j.Posts"],
    dims=["SE_mysql.Posttypes"], dim_alt=["SE_mysql.Users"],
    measures=[find("SE_mysql.postes","ViewCount"), find("SE_mysql.postes","CreationDate")],
    params={"SE_mysql.Posttypes":[find("SE_mysql.Posttypes","Name")]}),
 "E2": dict(cluster="E", fact="SE_mysql.Users",
    alt_facts=[],
    dims=["SE_mysql.postes"], dim_alt=["SE_mysql.Votes"],
    measures=[find("SE_mysql.Users","Views")],
    params={"SE_mysql.postes":[find("SE_mysql.postes","CreationDate")]}),
 "X1": dict(cluster="X", fact="SE_mysql.Users",
    alt_facts=["SE_mysql.postes"],
    dims=["SE_neo4j.Tags","SE_mysql.postes"], dim_alt=[],
    measures=[find("SE_mysql.Users","Reputation")],
    params={"SE_neo4j.Tags":[find("SE_neo4j.Tags","tagname")],
            "SE_mysql.postes":[find("SE_mysql.postes","ViewCount")]}),
 "X2": dict(cluster="X", fact="SE_mysql.Votes",
    alt_facts=["SE_mysql.Flagtypes"],
    dims=["SE_mysql.Flagtypes","SE_mysql.postes"], dim_alt=[],
    measures=[find("SE_mysql.Votes","BountyAmount")],
    params={"SE_mysql.Flagtypes":[find("SE_mysql.Flagtypes","Name")],
            "SE_mysql.postes":[find("SE_mysql.postes","Id")]}),
}

# COLD1/COLD2: two placeholder analyst IDs that get NO events at all
# (they simply never appear in the generation loop below). They exist
# purely so the evaluation scripts have two "genuinely zero-history"
# analysts to exercise the pure metadata-fallback / cold-start code
# path (Section 6.4 of the paper), as opposed to the 15 personas who
# always have at least some history.
COLD = ["COLD1", "COLD2"]

SESSIONS_PER_PERSONA = 4
# Every analyst ID the evaluation scripts will build count-matrices
# over -- the 15 personas plus the 2 (event-less) cold-start IDs.
ANALYSTS = list(PERSONAS.keys()) + COLD

events = []  # each entry: (session_id, persona, round_no, item_type, item_name, role)
sid = 0

def add(session, persona, rnd, itype, name, role):
    """Tiny helper so every event-append below reads as one line
    instead of repeating events.append((...)) everywhere."""
    events.append((session, persona, rnd, itype, name, role))

# ---------------------------------------------------------------------
# THE GENERATION LOOP: for every persona, run SESSIONS_PER_PERSONA=4
# independent "OLAP cube design sessions". Each session produces
# exactly one Fact event, 1+ Dimension events, 1+ Measure events, and
# 1+ Parameter event(s) per chosen dimension -- mirroring what a real
# analyst does when building a cube (pick a fact, pick dimensions,
# pick measures, pick how to slice each dimension).
# ---------------------------------------------------------------------
for persona, spec in PERSONAS.items():
    for rnd in range(1, SESSIONS_PER_PERSONA + 1):
        sid += 1
        session = f"S{sid:03d}"

        # --- FACT ---
        # 85% of the time: the persona's usual fact. 15% of the time
        # (only if alt_facts is non-empty): explore an alternate fact
        # instead. This is what makes the log non-trivial -- without
        # it, "own-history-only" ranking would be a perfect predictor
        # every single time.
        fact = spec["fact"]
        if spec["alt_facts"] and random.random() < 0.15:
            fact = random.choice(spec["alt_facts"])
        add(session, persona, rnd, "Dataset", fact, "F")

        # --- MEASURE POOL SELECTION (bug fix lives here) ---
        # Measures must belong to whichever dataset was actually chosen
        # as fact THIS session. If we always used spec["measures"]
        # (which are tied to spec["fact"], the persona's DEFAULT fact),
        # then an alt_fact session would record a Measure belonging to
        # a different dataset than the Fact actually picked above --
        # silently violating the intended invariant "every Fact
        # selection has at least one Measure on the same dataset".
        # Fix: when the alt-fact branch fired, fall back to that
        # fact's OWN attributes (attrs_of(fact)) instead of the
        # persona's default measure list.
        if fact == spec["fact"]:
            measures_pool = spec["measures"]
        else:
            measures_pool = attrs_of(fact) or spec["measures"]

        # --- DIMENSIONS ---
        # Start from the persona's usual dimension(s). 25% of the time
        # (if dim_alt is non-empty), ADD one extra "exploration"
        # dimension on top (not a replacement, unlike the fact branch
        # above). Then randomly choose how many of the resulting pool
        # to actually use this session (at least 1).
        dim_pool = list(spec["dims"])
        if spec["dim_alt"] and random.random() < 0.25:
            dim_pool = dim_pool + [random.choice(spec["dim_alt"])]
        n_dims = min(len(dim_pool), random.randint(1, max(1, len(dim_pool))))
        chosen_dims = random.sample(dim_pool, n_dims)
        for d in chosen_dims:
            add(session, persona, rnd, "Dataset", d, "D")

        # --- MEASURES ---
        # Same "choose a random-size subset" pattern, but drawing from
        # measures_pool (the fact-consistent pool computed above).
        n_meas = min(len(measures_pool), random.randint(1, max(1, len(measures_pool))))
        chosen_meas = random.sample(measures_pool, n_meas)
        for m in chosen_meas:
            add(session, persona, rnd, "Attribute", m, "M")

        # --- PARAMETERS ---
        # For EVERY dimension chosen this session, record 1+ Parameter
        # attribute(s) belonging to that dimension's dataset. Uses the
        # persona's declared params for that dimension if present,
        # otherwise falls back to that dimension's first attribute
        # (attrs_of(d)[:1]) as a safe default so no dimension is ever
        # left without a parameter.
        for d in chosen_dims:
            plist = spec["params"].get(d, attrs_of(d)[:1])
            n_p = min(len(plist), random.randint(1, max(1, len(plist))))
            for p in random.sample(plist, n_p):
                add(session, persona, rnd, "Attribute", p, "P")

        # --- LIGHT EXPLORATION NOISE ---
        # 10% chance of touching one extra Measure attribute of the
        # fact dataset, on top of the "normal" measures above. Purely
        # adds a bit more realistic noise to the log; does not affect
        # the Fact->Measure invariant since it draws from attrs_of(fact)
        # (the CURRENT session's fact), same safety as the fix above.
        if random.random() < 0.10:
            pool = attrs_of(fact)
            if pool:
                add(session, persona, rnd, "Attribute", random.choice(pool), "M")

print(f"Generated {len(events)} events across {sid} sessions, {len(PERSONAS)} personas.")

# =======================================================================
# EVERYTHING BELOW THIS LINE writes the same `events` list into an Excel
# workbook, purely so a human can open it and inspect the log directly.
# It does not affect the evaluation scripts, which import `events`
# straight from Python (the code above) -- the workbook is a reporting
# artifact, not the source of truth.
# =======================================================================
wb = openpyxl.Workbook()
FONT = "Calibri"
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
CELL_FONT = Font(name=FONT, size=10)

def style_header(ws, ncols, row=1):
    """Bold white-on-navy header row + centered wrapped text, applied
    to every sheet below for a consistent look."""
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

# --- Sheet 1: SessionLog -- the raw event list, one row per event ---
# This is the literal ground truth: every other sheet in the workbook
# is DERIVED from this one via spreadsheet formulas (not hardcoded),
# so opening the workbook and editing a row here would ripple through
# to the Datasets/Attributs/Similarity sheets automatically.
ws_log = wb.active
ws_log.title = "SessionLog"
ws_log.append(["SessionID", "Persona", "Round", "ItemType", "ItemName", "Role"])
style_header(ws_log, 6)
for row in events:
    ws_log.append(list(row))
for col, w in zip("ABCDEF", [10, 10, 8, 12, 34, 8]):
    ws_log.column_dimensions[col].width = w
ws_log.freeze_panes = "A2"

n_events = len(events)
LOG_LAST = n_events + 1  # last data row in SessionLog (row 1 is the header)

# --- Sheet 2: Datasets -- per-analyst Fact/Dimension usage COUNTS ---
# One row per analyst, two columns per dataset ("(F)" count, "(D)"
# count). Every cell is a COUNTIFS formula over SessionLog, e.g.
# "how many times did this analyst use this exact dataset in the Fact
# role?" -- this is exactly the ds_counts[persona][dataset]['F'] /
# ['D'] structure the evaluation scripts build in Python, just
# expressed as spreadsheet formulas here for human inspection.
ws_ds = wb.create_sheet("Datasets")
header = ["Analyste"]
for d in DATASETS:
    header += [f"{d} (F)", f"{d} (D)"]
ws_ds.append(header)
style_header(ws_ds, len(header))
ws_ds.column_dimensions["A"].width = 10

for r, analyst in enumerate(ANALYSTS, start=2):
    ws_ds.cell(row=r, column=1, value=analyst).font = CELL_FONT
    col = 2
    for d in DATASETS:
        for role in ("F", "D"):
            formula = (f'=COUNTIFS(SessionLog!$B$2:$B${LOG_LAST},$A{r},'
                       f'SessionLog!$E$2:$E${LOG_LAST},"{d}",'
                       f'SessionLog!$F$2:$F${LOG_LAST},"{role}")')
            ws_ds.cell(row=r, column=col, value=formula).font = CELL_FONT
            col += 1
for c in range(2, len(header) + 1):
    ws_ds.column_dimensions[get_column_letter(c)].width = 13
ws_ds.freeze_panes = "B2"

N_AN = len(ANALYSTS)
DS_FIRST_ROW, DS_LAST_ROW = 2, 1 + N_AN

# --- Sheet 3: Attributs -- same idea, but per-analyst Measure/Parameter
# usage counts for every attribute (mirrors at_counts in Python). ---
ws_at = wb.create_sheet("Attributs")
header2 = ["Analyste"]
for a in ATTRS:
    header2 += [f"{a} (M)", f"{a} (P)"]
ws_at.append(header2)
style_header(ws_at, len(header2))
ws_at.column_dimensions["A"].width = 10

for r, analyst in enumerate(ANALYSTS, start=2):
    ws_at.cell(row=r, column=1, value=analyst).font = CELL_FONT
    col = 2
    for a in ATTRS:
        for role in ("M", "P"):
            formula = (f'=COUNTIFS(SessionLog!$B$2:$B${LOG_LAST},$A{r},'
                       f'SessionLog!$E$2:$E${LOG_LAST},"{a}",'
                       f'SessionLog!$F$2:$F${LOG_LAST},"{role}")')
            ws_at.cell(row=r, column=col, value=formula).font = CELL_FONT
            col += 1
for c in range(2, len(header2) + 1):
    ws_at.column_dimensions[get_column_letter(c)].width = 13
ws_at.freeze_panes = "B2"

AT_FIRST_ROW, AT_LAST_ROW = 2, 1 + N_AN

# --- Sheet 4: Similarity_Datasets -- a 15x15 cosine-similarity matrix
# between datasets, computed directly from the Datasets sheet. ---
# For each pair (di, dj), cosine similarity = dot-product of their
# usage vectors (F-count, D-count across all analysts) divided by the
# product of their vector norms. Written here as raw SUMPRODUCT/SUMSQ
# spreadsheet formulas rather than a Python computation, so the
# workbook is self-contained and re-derives everything from SessionLog
# alone if you edit a raw event by hand.
ws_sd = wb.create_sheet("Similarity_Datasets")
ws_sd.cell(row=1, column=1, value="")
for j, d in enumerate(DATASETS, start=2):
    ws_sd.cell(row=1, column=j, value=d)
for i, d in enumerate(DATASETS, start=2):
    ws_sd.cell(row=i, column=1, value=d)
for c in range(1, len(DATASETS) + 2):
    ws_sd.cell(row=1, column=c).fill = HEAD_FILL
    ws_sd.cell(row=1, column=c).font = HEAD_FONT
for r in range(2, len(DATASETS) + 2):
    ws_sd.cell(row=r, column=1).fill = HEAD_FILL
    ws_sd.cell(row=r, column=1).font = HEAD_FONT
ws_sd.column_dimensions["A"].width = 20
for c in range(2, len(DATASETS) + 2):
    ws_sd.column_dimensions[get_column_letter(c)].width = 12

def ds_cols(idx):
    """Given a dataset's position `idx` in DATASETS, return the two
    Excel column letters (F-count, D-count) holding its usage vector
    in the Datasets sheet -- each dataset occupies 2 consecutive
    columns there, starting at column B (index 2)."""
    f_col = 2 + 2 * idx
    return get_column_letter(f_col), get_column_letter(f_col + 1)

for i, di in enumerate(DATASETS):
    fi, dii = ds_cols(i)
    for j, dj in enumerate(DATASETS):
        fj, dj_ = ds_cols(j)
        # dot-product across BOTH the F-count and D-count sub-vectors
        dot = (f'SUMPRODUCT(Datasets!${fi}${DS_FIRST_ROW}:${fi}${DS_LAST_ROW},Datasets!${fj}${DS_FIRST_ROW}:${fj}${DS_LAST_ROW})'
               f'+SUMPRODUCT(Datasets!${dii}${DS_FIRST_ROW}:${dii}${DS_LAST_ROW},Datasets!${dj_}${DS_FIRST_ROW}:${dj_}${DS_LAST_ROW})')
        # ||di|| and ||dj|| (Euclidean norm of each dataset's usage vector)
        normi = (f'SQRT(SUMSQ(Datasets!${fi}${DS_FIRST_ROW}:${fi}${DS_LAST_ROW})'
                 f'+SUMSQ(Datasets!${dii}${DS_FIRST_ROW}:${dii}${DS_LAST_ROW}))')
        normj = (f'SQRT(SUMSQ(Datasets!${fj}${DS_FIRST_ROW}:${fj}${DS_LAST_ROW})'
                 f'+SUMSQ(Datasets!${dj_}${DS_FIRST_ROW}:${dj_}${DS_LAST_ROW}))')
        # cos(theta) = dot / (||di|| * ||dj||); IFERROR guards against a
        # brand-new/never-used dataset having a zero-length vector,
        # which would otherwise divide by zero.
        formula = f'=IFERROR(({dot})/(({normi})*({normj})),0)'
        ws_sd.cell(row=2 + i, column=2 + j, value=formula).font = CELL_FONT
ws_sd.freeze_panes = "B2"

# --- Sheet 5: Similarity_Attributs -- the same cosine-similarity
# construction as Sheet 4, but a 63x63 matrix over ATTRS using their
# (M-count, P-count) usage vectors from the Attributs sheet. ---
ws_sa = wb.create_sheet("Similarity_Attributs")
for j, a in enumerate(ATTRS, start=2):
    ws_sa.cell(row=1, column=j, value=a)
for i, a in enumerate(ATTRS, start=2):
    ws_sa.cell(row=i, column=1, value=a)
for c in range(1, len(ATTRS) + 2):
    ws_sa.cell(row=1, column=c).fill = HEAD_FILL
    ws_sa.cell(row=1, column=c).font = HEAD_FONT
for r in range(2, len(ATTRS) + 2):
    ws_sa.cell(row=r, column=1).fill = HEAD_FILL
    ws_sa.cell(row=r, column=1).font = HEAD_FONT
ws_sa.column_dimensions["A"].width = 26
for c in range(2, len(ATTRS) + 2):
    ws_sa.column_dimensions[get_column_letter(c)].width = 10

def at_cols(idx):
    """Same idea as ds_cols() above, but for the Attributs sheet's
    (M-count, P-count) column pairs."""
    m_col = 2 + 2 * idx
    return get_column_letter(m_col), get_column_letter(m_col + 1)

for i, ai in enumerate(ATTRS):
    mi, pi = at_cols(i)
    for j, aj in enumerate(ATTRS):
        mj, pj = at_cols(j)
        dot = (f'SUMPRODUCT(Attributs!${mi}${AT_FIRST_ROW}:${mi}${AT_LAST_ROW},Attributs!${mj}${AT_FIRST_ROW}:${mj}${AT_LAST_ROW})'
               f'+SUMPRODUCT(Attributs!${pi}${AT_FIRST_ROW}:${pi}${AT_LAST_ROW},Attributs!${pj}${AT_FIRST_ROW}:${pj}${AT_LAST_ROW})')
        normi = (f'SQRT(SUMSQ(Attributs!${mi}${AT_FIRST_ROW}:${mi}${AT_LAST_ROW})'
                 f'+SUMSQ(Attributs!${pi}${AT_FIRST_ROW}:${pi}${AT_LAST_ROW}))')
        normj = (f'SQRT(SUMSQ(Attributs!${mj}${AT_FIRST_ROW}:${mj}${AT_LAST_ROW})'
                 f'+SUMSQ(Attributs!${pj}${AT_FIRST_ROW}:${pj}${AT_LAST_ROW}))')
        formula = f'=IFERROR(({dot})/(({normi})*({normj})),0)'
        ws_sa.cell(row=2 + i, column=2 + j, value=formula).font = CELL_FONT
ws_sa.freeze_panes = "B2"

# --- Sheet 6: README -- plain-text documentation embedded IN the
# workbook itself, so it's self-explanatory even if separated from
# this .py file. Mirrors the docstring at the top of this file. ---
ws_r = wb.create_sheet("README")
ws_r.column_dimensions["A"].width = 110
lines = [
 "OLAP Recommender -- Simulated Multi-Analyst Usage Log",
 "",
 f"Random seed: {SEED}  |  Personas: {len(PERSONAS)}  |  Sessions per persona: {SESSIONS_PER_PERSONA}  |  Total sessions: {sid}  |  Total events: {n_events}",
 "",
 "Methodology (for paper Section 6.1):",
 "This usage log was generated by a documented, seeded stochastic simulation of 15 analyst",
 "personas (see persona_protocol.md), each with a fixed analytical intent (typical fact,",
 "dimensions, measures, dimension parameters), run for 4 sessions each. At each session the",
 "persona's typical fact is selected with 85% probability (15% alternate-fact exploration);",
 "1..N dimensions/measures are sampled from the persona's typical pool (with 25% chance of",
 "one out-of-pool 'exploration' dimension); dimension parameters are sampled per chosen",
 "dimension; and a 10% chance of one extra exploratory attribute use is added per session.",
 "Two additional near-empty analysts (COLD1, COLD2) are included with zero prior sessions,",
 "used as the cold-start subset (Section 6.4).",
 "",
 "Sheet structure:",
 "  - SessionLog: raw event log, one row per (session, persona, item, role) selection.",
 "    This is the ground truth for leave-one-out masking in Section 6.2.",
 "  - Datasets / Attributs: aggregated F/D and M/P count matrices, computed via COUNTIFS",
 "    formulas over SessionLog (not hardcoded) -- matches the format of Tables 3.1/3.3.",
 "  - Similarity_Datasets / Similarity_Attributs: cosine similarity matrices computed via",
 "    SUMPRODUCT/SUMSQ formulas over the Datasets/Attributs sheets, matching the thesis's",
 "    cosine-similarity formula (Section 3.6). IFERROR(...,0) guards against divide-by-zero",
 "    for items with zero usage.",
 "",
 "Reproducibility: rerun build_usage_log.py with the same SEED to regenerate identically,",
 "or change SEED / SESSIONS_PER_PERSONA / persona definitions to produce variants.",
 "",
 "Known limitation to disclose in Section 7 (threats to validity): this log is generated by",
 "a single author role-playing personas, not collected from independent human subjects.",
]
for i, line in enumerate(lines, start=1):
    c = ws_r.cell(row=i, column=1, value=line)
    c.font = Font(name=FONT, size=11, bold=(i == 1))

wb.save("./OLAP_Recommender_UsageLog.xlsx")
print("saved")
