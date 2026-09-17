import random, openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Alignment

SEED = 42
random.seed(SEED)

DATASETS = ["SE_mysql.postes","SE_mysql.Posttypes","SE_mysql.Users","SE_mysql.Flagtypes",
    "SE_mysql.Votes","SE_mysql.Votetypes","SE_mongodb.Badges","SE_mongodb.Posts",
    "SE_mongodb.Users","SE_mongodb.Comments","SE_neo4j.postlinks","SE_neo4j.Posts",
    "SE_neo4j.Comments","SE_neo4j.Users","SE_neo4j.Tags"]

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
    return [a for a in ATTRS if a.startswith(ds + ".")]

def find(ds, substr):
    for a in attrs_of(ds):
        if substr.lower() in a.lower():
            return a
    raise ValueError(f"not found: {ds}.{substr}")

# ---- Persona definitions: fact, alt_facts (occasional), dims, dim_alt, measures, params(per dim) ----
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

# Two deliberately near-empty analysts for the cold-start subset (Sec. 6.4)
COLD = ["COLD1", "COLD2"]

SESSIONS_PER_PERSONA = 4
ANALYSTS = list(PERSONAS.keys()) + COLD

events = []  # (session_id, persona, round_no, item_type, item_name, role)
sid = 0

def add(session, persona, rnd, itype, name, role):
    events.append((session, persona, rnd, itype, name, role))

for persona, spec in PERSONAS.items():
    for rnd in range(1, SESSIONS_PER_PERSONA + 1):
        sid += 1
        session = f"S{sid:03d}"

        fact = spec["fact"]
        if spec["alt_facts"] and random.random() < 0.15:
            fact = random.choice(spec["alt_facts"])
        add(session, persona, rnd, "Dataset", fact, "F")

        # Measures must belong to whichever dataset was actually chosen as fact.
        # (Bug fix: previously spec["measures"] was used unconditionally, which
        # are tied to spec["fact"] -- so an alt_fact session recorded a Measure
        # for a dataset different from the Fact it was supposed to belong to.)
        if fact == spec["fact"]:
            measures_pool = spec["measures"]
        else:
            measures_pool = attrs_of(fact) or spec["measures"]

        dim_pool = list(spec["dims"])
        if spec["dim_alt"] and random.random() < 0.25:
            dim_pool = dim_pool + [random.choice(spec["dim_alt"])]
        n_dims = min(len(dim_pool), random.randint(1, max(1, len(dim_pool))))
        chosen_dims = random.sample(dim_pool, n_dims)
        for d in chosen_dims:
            add(session, persona, rnd, "Dataset", d, "D")

        n_meas = min(len(measures_pool), random.randint(1, max(1, len(measures_pool))))
        chosen_meas = random.sample(measures_pool, n_meas)
        for m in chosen_meas:
            add(session, persona, rnd, "Attribute", m, "M")

        for d in chosen_dims:
            plist = spec["params"].get(d, attrs_of(d)[:1])
            n_p = min(len(plist), random.randint(1, max(1, len(plist))))
            for p in random.sample(plist, n_p):
                add(session, persona, rnd, "Attribute", p, "P")

        # light exploratory noise: occasionally touch one extra attribute of the fact dataset
        if random.random() < 0.10:
            pool = attrs_of(fact)
            if pool:
                add(session, persona, rnd, "Attribute", random.choice(pool), "M")

print(f"Generated {len(events)} events across {sid} sessions, {len(PERSONAS)} personas.")

# ---------------- Build workbook ----------------
wb = openpyxl.Workbook()
FONT = "Calibri"
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
HEAD_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
CELL_FONT = Font(name=FONT, size=10)

def style_header(ws, ncols, row=1):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

# --- Sheet 1: SessionLog ---
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
LOG_LAST = n_events + 1  # last data row in SessionLog

# --- Sheet 2: Datasets (aggregated via COUNTIFS formulas) ---
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

# --- Sheet 3: Attributs (aggregated via COUNTIFS formulas) ---
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

# --- Sheet 4: Similarity_Datasets (cosine similarity via SUMPRODUCT/SUMSQ) ---
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
    f_col = 2 + 2 * idx
    return get_column_letter(f_col), get_column_letter(f_col + 1)

for i, di in enumerate(DATASETS):
    fi, dii = ds_cols(i)
    for j, dj in enumerate(DATASETS):
        fj, dj_ = ds_cols(j)
        dot = (f'SUMPRODUCT(Datasets!${fi}${DS_FIRST_ROW}:${fi}${DS_LAST_ROW},Datasets!${fj}${DS_FIRST_ROW}:${fj}${DS_LAST_ROW})'
               f'+SUMPRODUCT(Datasets!${dii}${DS_FIRST_ROW}:${dii}${DS_LAST_ROW},Datasets!${dj_}${DS_FIRST_ROW}:${dj_}${DS_LAST_ROW})')
        normi = (f'SQRT(SUMSQ(Datasets!${fi}${DS_FIRST_ROW}:${fi}${DS_LAST_ROW})'
                 f'+SUMSQ(Datasets!${dii}${DS_FIRST_ROW}:${dii}${DS_LAST_ROW}))')
        normj = (f'SQRT(SUMSQ(Datasets!${fj}${DS_FIRST_ROW}:${fj}${DS_LAST_ROW})'
                 f'+SUMSQ(Datasets!${dj_}${DS_FIRST_ROW}:${dj_}${DS_LAST_ROW}))')
        formula = f'=IFERROR(({dot})/(({normi})*({normj})),0)'
        ws_sd.cell(row=2 + i, column=2 + j, value=formula).font = CELL_FONT
ws_sd.freeze_panes = "B2"

# --- Sheet 5: Similarity_Attributs (cosine similarity via SUMPRODUCT/SUMSQ) ---
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

# --- Sheet 6: README ---
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

wb.save("/home/claude/OLAP_Recommender_UsageLog.xlsx")
print("saved")
