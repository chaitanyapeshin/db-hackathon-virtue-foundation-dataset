# Databricks notebook source
# MAGIC %md
# MAGIC # CareCompass: the right care, at the right speed, with the evidence shown
# MAGIC **CMU Datathon Fall 2026 · Virtue Foundation India facility dataset**
# MAGIC
# MAGIC A community health worker types *"accident hua hai, behosh hai, near Lucknow"* or *"dialysis twice a week near Gaya"*. CareCompass decides **how urgent** this is, picks the facilities most likely to help, checks them against **government scheme lists**, and explains every recommendation without stating weak evidence as fact.
# MAGIC
# MAGIC **Our core idea: urgency changes what counts as success.** For planned care, success is *"a facility that reliably offers it, reasonably close"*. For an emergency, success is *"competent hands within the clinical time window"*, even if that's a small hospital that stabilises the patient and transfers them on.
# MAGIC
# MAGIC ```
# MAGIC  Given dataset ──► Severity classifier ──┬──► LOW  severity (dialysis, MRI, chemo…)
# MAGIC  (10k facilities)  (hard-coded clinical   │      Graph + text evidence pipeline → best facility for the service
# MAGIC                     priors + red-flag     │      (multi-criteria ranking, uncertainty ranges, Pareto trade-off)
# MAGIC                     words, cost-sensitive)│
# MAGIC                                           └──► HIGH severity (trauma, PPH, stroke, snakebite…)
# MAGIC                                                  Monte Carlo over capability × location × travel time
# MAGIC                                                  (direct care vs stabilise-then-transfer)
# MAGIC                         ──► Verify against PM-JAY / state scheme / de-empanelled lists
# MAGIC                         ──► Recommendation in hedged language (never weak evidence as fact)
# MAGIC ```
# MAGIC
# MAGIC | § | Section | Judging criterion |
# MAGIC |---|---|---|
# MAGIC | 1–2 | Setup, discovery, EDA | Exploration |
# MAGIC | 3 | Record trust and location repair | Exploration / Modelling |
# MAGIC | 4 | Evidence extraction with citations | Modelling |
# MAGIC | 5 | Graph propagation + masked-keyword plausibility model | Modelling |
# MAGIC | 6 | Government registry linking (Fellegi–Sunter-style) | Modelling / Impact |
# MAGIC | 7 | Calibrated P(capable) with uncertainty ranges | Modelling |
# MAGIC | 8 | Severity classifier + safety test set | Modelling / Impact |
# MAGIC | 9 | Low-severity branch | Modelling |
# MAGIC | 10 | High-severity branch: Monte Carlo | Modelling |
# MAGIC | 11 | Validation: ablations vs naive lookup | Modelling |
# MAGIC | 12 | Impact: golden-hour coverage, verification priorities, policy | Impact |
# MAGIC | 13 | Gold tables + export for the dashboard | Delivery |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and data discovery

# COMMAND ----------

# MAGIC %pip install -q plotly scikit-learn rapidfuzz

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, re, json, ast, math, glob, difflib, warnings
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 120)

# The query-time engine lives in app/engine.py so the notebook and the dashboard share one implementation.
try:
    _nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    sys.path.insert(0, "/Workspace" + os.path.dirname(_nb) + "/app")
except Exception:
    sys.path.insert(0, os.path.join(os.getcwd(), "app"))
import engine
from engine import SERVICES, haversine_km

SOURCE_CATALOG = "databricks_virtue_foundation_dataset_dais_2026"
FACILITY_TABLE = None            # None = auto-discover from information_schema
OUTPUT_CATALOG, OUTPUT_SCHEMA = "workspace", "care_compass"
REGISTRY_DIR = f"/Volumes/{OUTPUT_CATALOG}/{OUTPUT_SCHEMA}/registries"
EXPORT_DIR = f"/Volumes/{OUTPUT_CATALOG}/{OUTPUT_SCHEMA}/exports"
RNG = np.random.default_rng(42)
print("engine loaded with", len(SERVICES), "services")

# COMMAND ----------

tables_df = spark.sql(f"""
    SELECT table_schema, table_name, table_type FROM {SOURCE_CATALOG}.information_schema.tables
    WHERE table_schema <> 'information_schema'""").toPandas()
columns_df = spark.sql(f"""
    SELECT table_schema, table_name, column_name, data_type, ordinal_position
    FROM {SOURCE_CATALOG}.information_schema.columns
    WHERE table_schema <> 'information_schema' ORDER BY table_schema, table_name, ordinal_position""").toPandas()
display(tables_df)

def pick_facility_table(columns_df):
    signals = ["name", "lat", "lon", "description", "capab", "procedure", "equipment", "special", "city", "state"]
    scores = {}
    for (schema, table), g in columns_df.groupby(["table_schema", "table_name"]):
        cols = " ".join(g["column_name"].str.lower())
        scores[f"{SOURCE_CATALOG}.{schema}.{table}"] = sum(s in cols for s in signals) + len(g) / 100
    return max(scores, key=scores.get)

FACILITY_TABLE = FACILITY_TABLE or pick_facility_table(columns_df)
print("Using facility table:", FACILITY_TABLE)
raw_pdf = spark.table(FACILITY_TABLE).toPandas()
print(f"Loaded {len(raw_pdf):,} rows × {raw_pdf.shape[1]} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Column mapping and normalisation
# MAGIC We map *logical* fields to whatever the physical columns are called and print the mapping (override with `MANUAL_COLUMN_MAP`). List-like fields arrive as JSON strings, Spark arrays or text; we normalise them to Python lists and split camelCase (`internalMedicine` → `internal Medicine`) so evidence rules can match.

# COMMAND ----------

LOGICAL_FIELDS = {
    "id": ["pk_unique_id", "unique_id", "facility_id", "id", "uuid"],
    "name": ["name", "facility_name", "organization_name", "hospital_name"],
    "state": ["address_stateorregion", "state", "address_state", "state_name"],
    "city": ["address_city", "city", "town"],
    "district": ["district", "address_district"],
    "postcode": ["address_ziporpostcode", "postcode", "pincode", "pin_code", "zip", "postal_code"],
    "lat": ["latitude", "lat", "address_latitude"],
    "lon": ["longitude", "lon", "lng", "long", "address_longitude"],
    "address": ["address_line1", "address", "full_address", "street"],
    "description": ["description", "facility_description"],
    "specialties": ["specialties", "specialty", "specialities"],
    "capability": ["capability", "capabilities"],
    "procedure": ["procedure", "procedures"],
    "equipment": ["equipment", "equipments"],
    "n_doctors": ["numberdoctors", "number_doctors", "num_doctors", "doctors"],
    "capacity": ["capacity", "bed_capacity", "beds", "number_of_beds"],
    "year_est": ["yearestablished", "year_established", "established"],
    "facility_type": ["facilitytypeid", "facility_type", "facility_type_id", "facilitytype"],
    "source_url": ["source_url", "source_urls", "sources", "url"],
    "website": ["officialwebsite", "official_website", "websites", "website"],
    "phone": ["officialphone", "phone_numbers", "phone", "phones"],
}
MANUAL_COLUMN_MAP = {}

def _nc(c): return re.sub(r"[^a-z0-9]", "", str(c).lower())

def map_columns(df):
    normed = {_nc(c): c for c in df.columns}
    mapping, used = {}, set()
    for field, cands in LOGICAL_FIELDS.items():
        if field in MANUAL_COLUMN_MAP:
            mapping[field] = MANUAL_COLUMN_MAP[field]; used.add(mapping[field]); continue
        hit = next((normed[_nc(c)] for c in cands if _nc(c) in normed and normed[_nc(c)] not in used), None)
        if hit is None:
            hit = next((o for c in cands for n, o in normed.items() if len(_nc(c)) >= 4 and _nc(c) in n and o not in used), None)
        if hit:
            mapping[field] = hit; used.add(hit)
    return mapping

COLMAP = map_columns(raw_pdf)
display(pd.DataFrame([{"logical_field": k, "physical_column": COLMAP.get(k, "— not found —")} for k in LOGICAL_FIELDS]))

NULLS = {"", "null", "none", "nan", "n/a", "na", "[]", "{}", "unknown", "-"}
def to_list(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return []
    if isinstance(v, (list, tuple, set, np.ndarray)):
        out = []
        for x in v: out.extend(to_list(x))
        return out
    s = str(v).strip()
    if s.lower() in NULLS: return []
    if s[:1] in "[(":
        for parser in (json.loads, ast.literal_eval):
            try:
                p = parser(s)
                if isinstance(p, (list, tuple)): return to_list(list(p))
            except Exception: pass
        s = s.strip("[]()")
    return [s]
split_camel = lambda s: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
clean_text = lambda v: " ".join(to_list(v)) or None
def first_num(v):
    for it in to_list(v):
        m = re.search(r"-?\d+(?:\.\d+)?", str(it).replace(",", ""))
        if m: return float(m.group(0))
    return np.nan

def build_facilities(raw, cm):
    f = pd.DataFrame(index=raw.index)
    g = lambda k: raw[cm[k]] if k in cm else pd.Series([None] * len(raw), index=raw.index)
    f["facility_id"] = g("id").astype(str) if "id" in cm else [f"F{i:06d}" for i in range(len(raw))]
    for k in ["name", "state", "city", "district", "address", "description", "facility_type"]:
        f[k] = g(k).map(clean_text)
    f["postcode"] = g("postcode").map(lambda v: re.sub(r"\D", "", clean_text(v) or "")[:6] or None)
    for k in ["lat", "lon", "n_doctors", "capacity", "year_est"]:
        f[k] = g(k).map(first_num)
    for k in ["specialties", "capability", "procedure", "equipment"]:
        f[k] = g(k).map(lambda v: [split_camel(x) for x in to_list(v)])
    for k in ["source_url", "website", "phone"]:
        f[k] = g(k).map(to_list)
    f["name"] = f["name"].fillna("(unnamed facility)")
    for k in ["state", "city", "district"]:
        f[k + "_norm"] = f[k].map(lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip() if isinstance(s, str) else None)
    return f.reset_index(drop=True)

fac = build_facilities(raw_pdf, COLMAP)
display(fac.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exploratory data analysis

# COMMAND ----------

def is_filled(v):
    if isinstance(v, list): return len(v) > 0
    if v is None or (isinstance(v, float) and math.isnan(v)): return False
    return str(v).strip() != ""

EDA_FIELDS = ["name", "state", "city", "postcode", "lat", "description", "specialties", "capability",
              "procedure", "equipment", "year_est", "n_doctors", "capacity", "source_url", "phone"]
BRIEF = {"description": 100, "capability": 99.7, "procedure": 92.5, "equipment": 77.0, "year_est": 47.8, "n_doctors": 36.4, "capacity": 25.2}
cov = pd.DataFrame({"field": EDA_FIELDS, "coverage_pct": [100 * fac[c].map(is_filled).mean() for c in EDA_FIELDS]})
cov["brief_pct"] = cov.field.map(BRIEF)
cov = cov.sort_values("coverage_pct")
fig = px.bar(cov, x="coverage_pct", y="field", orientation="h", text=cov.coverage_pct.round(1),
             title="Field coverage: text claims are near-universal, scale fields are mostly missing")
fig.add_scatter(x=cov.brief_pct, y=cov.field, mode="markers", name="stated in brief", marker=dict(symbol="diamond", size=10, color="black"))
fig.update_layout(height=500, xaxis_range=[0, 105]); fig.show()

# COMMAND ----------

for k in ["specialties", "capability", "procedure", "equipment"]:
    fac[f"n_{k}"] = fac[k].map(len)
fac["n_claims"] = fac[["n_capability", "n_procedure", "n_equipment"]].sum(axis=1)
fac["desc_len"] = fac.description.fillna("").str.len()
fac["n_sources"] = fac.source_url.map(len)
display(fac[["n_specialties", "n_capability", "n_procedure", "n_equipment", "n_claims", "desc_len", "n_sources"]].describe().T.round(1))

c1 = fac.state.fillna("(missing)").value_counts().head(30).rename_axis("state").reset_index(name="facilities")
px.bar(c1, x="state", y="facilities", title="Facilities per state (top 30)").update_layout(height=420).show()
c2 = fac.facility_type.fillna("(missing)").value_counts().head(12).rename_axis("type").reset_index(name="facilities")
px.bar(c2, x="facilities", y="type", orientation="h", title="Facility type").show()
px.histogram(fac, x="n_claims", nbins=80, log_y=True, title="Claims per facility (capability + procedure + equipment items)").show()
for col, ttl in [("n_doctors", "Doctors (where reported)"), ("capacity", "Beds (where reported)")]:
    px.histogram(fac.dropna(subset=[col]), x=col, nbins=60, log_y=True, title=ttl).show()

# COMMAND ----------

# Coordinates: valid, missing, zero, swapped, outside India
LAT, LON = (6.0, 37.6), (68.0, 97.5)
lat_ok, lon_ok = fac.lat.between(*LAT), fac.lon.between(*LON)
swapped = fac.lon.between(*LAT) & fac.lat.between(*LON) & ~(lat_ok & lon_ok)
fac["coord_status"] = np.select([fac.lat.isna() | fac.lon.isna(), (fac.lat == 0) & (fac.lon == 0), swapped, lat_ok & lon_ok],
                                ["missing", "zero", "swapped", "ok"], default="outside_india")
sw = fac.coord_status == "swapped"
fac.loc[sw, ["lat", "lon"]] = fac.loc[sw, ["lon", "lat"]].values
display(fac.coord_status.value_counts().to_frame("facilities"))
geo = fac[fac.coord_status.isin(["ok", "swapped"])]
fig = px.scatter_geo(geo, lat="lat", lon="lon", color="state", hover_name="name", opacity=0.6, title="Facility locations")
fig.update_geos(fitbounds="locations", showcountries=True); fig.update_layout(height=620, showlegend=False); fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Record trust and location repair
# MAGIC | Signal | Logic | Penalty |
# MAGIC |---|---|---|
# MAGIC | Unusable coordinates | missing, (0,0) or outside India | −0.30 |
# MAGIC | Location contradicts state | > 1.5 × state's 90th-percentile spread and > 200 km from state median | −0.25 |
# MAGIC | Location contradicts city | > 60 km from city median (cities with ≥ 3 records) | −0.10 |
# MAGIC | PIN contradicts state | PIN's 2-digit postal region almost never seen with this state | −0.10 |
# MAGIC | Possible merged record | name joins two *distinctly named* facilities, description mentions branches/chain, or names ≥ 2 other cities | −0.15 |
# MAGIC | Over-claiming | claim count far above what reported size supports | −0.15 |
# MAGIC
# MAGIC `trust = (1 − Σ penalties) × (0.6 + 0.4 × documentation)`. Contradicted coordinates are **repaired** (city centroid → state centroid), and every facility gets a **location uncertainty radius `sigma_km`** that the Monte Carlo uses.

# COMMAND ----------

valid = fac.coord_status.isin(["ok", "swapped"])
state_med = fac[valid].groupby("state_norm")[["lat", "lon"]].median().rename(columns={"lat": "state_lat", "lon": "state_lon"})
fac = fac.join(state_med, on="state_norm")
fac["dist_state_km"] = haversine_km(fac.lat, fac.lon, fac.state_lat, fac.state_lon)
fac = fac.join(fac[valid].groupby("state_norm").dist_state_km.quantile(0.9).rename("state_p90"), on="state_norm")
fac["flag_geo_state"] = valid & (fac.dist_state_km > np.maximum(1.5 * fac.state_p90.fillna(0), 200))

fac["city_key"] = fac.city_norm.fillna("") + "|" + fac.state_norm.fillna("")
cs = fac[valid & ~fac.flag_geo_state].groupby("city_key").agg(city_lat=("lat", "median"), city_lon=("lon", "median"), city_n=("lat", "size"))
fac = fac.join(cs, on="city_key")
fac["dist_city_km"] = haversine_km(fac.lat, fac.lon, fac.city_lat, fac.city_lon)
fac["flag_geo_city"] = valid & (fac.city_n >= 3) & (fac.dist_city_km > 60) & ~fac.flag_geo_state

fac["pin2"] = fac.postcode.map(lambda p: p[:2] if isinstance(p, str) and len(p) == 6 and p[0] != "0" else None)
ps = fac.dropna(subset=["pin2", "state_norm"]).groupby(["pin2", "state_norm"]).size().rename("n").reset_index()
ps["pin_state_share"] = ps.n / ps.groupby("pin2").n.transform("sum")
fac = fac.merge(ps[["pin2", "state_norm", "pin_state_share"]], on=["pin2", "state_norm"], how="left")
fac["flag_pin_state"] = fac.pin2.notna() & (fac.pin_state_share < 0.05)

MERGE_SPLIT = re.compile(r"\s(?:/|\||&|and|\+)\s", re.I)
FAC_NOUN = re.compile(r"\b(hospital|clinic|nursing home|centre|center|medical|polyclinic|dispensary)\b", re.I)
GENERIC = {"research", "centre", "center", "hospital", "hospitals", "clinic", "institute", "diagnostic", "diagnostics", "maternity",
           "nursing", "home", "trauma", "heart", "eye", "dental", "multispeciality", "multispecialty", "super", "speciality",
           "specialty", "medical", "care", "cancer", "children", "women", "polyclinic", "the", "of", "and", "college", "sciences",
           "science", "health", "healthcare", "surgical", "laboratory", "lab", "imaging", "ivf", "fertility"}
def two_names(name):
    segs = MERGE_SPLIT.split(name or "")
    return len([s for s in segs if FAC_NOUN.search(s) and any(w not in GENERIC for w in re.findall(r"[a-z]+", s.lower()))]) >= 2
MERGE_DESC = re.compile(r"\b(branches|chain of|group of hospitals|network of (hospitals|clinics)|multiple locations|units across|other branch)\b", re.I)
cv = fac.city_norm.value_counts()
city_vocab = [c for c in cv[cv >= 5].index if isinstance(c, str) and len(c) >= 5]
CITY_RX = re.compile(r"\b(" + "|".join(map(re.escape, sorted(city_vocab, key=len, reverse=True))) + r")\b") if city_vocab else None
fac["n_other_cities"] = [0 if CITY_RX is None or not isinstance(d, str) else len(set(CITY_RX.findall(d.lower())) - {c})
                         for d, c in zip(fac.description, fac.city_norm)]
fac["flag_merge"] = fac.name.map(two_names) | fac.description.fillna("").str.contains(MERGE_DESC) | (fac.n_other_cities >= 2)
fac["name_norm"] = fac.name.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
fac["dup_group"] = fac.groupby(["name_norm", "city_norm"], dropna=False).ngroup()
fac["dup_size"] = fac.groupby("dup_group").facility_id.transform("size")

size_proxy = fac[["n_doctors", "capacity"]].max(axis=1)
fac["flag_overclaim"] = ((fac.n_claims > fac.n_claims.quantile(0.95)) & (size_proxy.fillna(9999) <= 10)) | ((fac.n_claims > 25) & (fac.n_doctors <= 2))
doc_fields = ["description", "capability", "procedure", "equipment", "year_est", "n_doctors", "capacity", "postcode", "phone", "website"]
fac["documentation"] = 0.8 * fac[doc_fields].apply(lambda r: np.mean([is_filled(v) for v in r]), axis=1) + 0.2 * np.clip(fac.n_sources / 3, 0, 1)
fac["coord_bad"] = ~valid
PENALTIES = {"coord_bad": 0.30, "flag_geo_state": 0.25, "flag_geo_city": 0.10, "flag_pin_state": 0.10, "flag_merge": 0.15, "flag_overclaim": 0.15}
fac["integrity"] = np.clip(1 - sum(fac[k].astype(float) * w for k, w in PENALTIES.items()), 0.05, 1)
fac["trust"] = fac.integrity * (0.6 + 0.4 * fac.documentation)

# Location repair + uncertainty radius
fac["coord_source"] = np.where(valid & ~fac.flag_geo_state, "record", None)
fac["sigma_km"] = np.where(fac.coord_source == "record", np.where(fac.flag_geo_city, 15.0, 1.0), np.nan)
for cond_col, (latc, lonc), label, sig in [
        ("city_lat", ("city_lat", "city_lon"), "city centroid (record coordinates unusable or contradictory)", 6.0),
        ("state_lat", ("state_lat", "state_lon"), "state centroid (location very uncertain)", 60.0)]:
    need = fac.coord_source.isna() & fac[cond_col].notna()
    fac.loc[need, ["lat", "lon"]] = fac.loc[need, [latc, lonc]].values
    fac.loc[need, "coord_source"] = label
    fac.loc[need, "sigma_km"] = sig
fac["coord_source"] = fac.coord_source.fillna("none")
fac["geo_ok"] = (fac.coord_source == "record") & ~fac.flag_geo_city

flags = list(PENALTIES)
display(pd.DataFrame({"flag": flags, "facilities": [int(fac[c].sum()) for c in flags], "pct": [round(100 * fac[c].mean(), 2) for c in flags]}))
display(fac.coord_source.value_counts().to_frame("facilities"))
px.histogram(fac, x="trust", nbins=40, title="Record trust score").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Service evidence extraction (with citations)
# MAGIC Each service in `engine.SERVICES` has **core** terms (direct statements), **support** terms (things real providers also have) and **prerequisites** (what the service can't run without). Every match is stored as a citation: *facility, field, exact snippet, matched term, negated?* Matches after *"no / refer / planned / coming soon"* are kept but down-weighted and surfaced as warnings.
# MAGIC
# MAGIC Score = noisy-OR over hits (`p = 0.55 × field_weight` for core, `0.20 ×` for support, `×0.3` if negated), plus a corroboration bonus when ≥ 2 different fields agree, times `0.85` per missing prerequisite.

# COMMAND ----------

FIELD_WEIGHT = {"procedure": 1.0, "equipment": 1.0, "capability": 0.9, "description": 0.7, "specialties": 0.6, "name": 0.8}
NEGATION = re.compile(r"\b(no|not|without|lack(s|ing)?|unavailable|refer(s|red|ral)?( to| out)?|plan(s|ned|ning)? to|"
                      r"coming soon|proposed|under construction|upcoming|will be|soon to)\b[^.]{0,40}$", re.I)
alt = lambda pats, suffix="": re.compile("|".join(f"(?:{p}){suffix}" for p in pats), re.I)
COMPILED = {s: {"core": alt(d["core"]), "support": alt(d["support"]),
                "prereq": {k: alt(v) for k, v in d["prereq"].items()}} for s, d in SERVICES.items()}

def facility_items(r):
    for fld in ["procedure", "equipment", "capability", "specialties"]:
        for it in r[fld]:
            if it and len(it) > 1: yield fld, it
    for sent in re.split(r"(?<=[.!?;])\s+|\n+", r["description"] or ""):
        if len(sent.strip()) > 3: yield "description", sent.strip()
    yield "name", r["name"]

def snippet(t, m, w=220):
    if len(t) <= w: return t
    a, b = max(0, m.start() - w // 2), min(len(t), m.end() + w // 2)
    return ("…" if a else "") + t[a:b] + ("…" if b < len(t) else "")

rows = []
for _, r in fac.iterrows():
    items = list(facility_items(r))
    for svc, cp in COMPILED.items():
        for fld, txt in items:
            for kind in ("core", "support"):
                m = cp[kind].search(txt)
                if m:
                    rows.append((r.facility_id, svc, kind, fld, m.group(0), snippet(txt, m), bool(NEGATION.search(txt[:m.start()]))))
evidence = pd.DataFrame(rows, columns=["facility_id", "service", "kind", "field", "matched", "snippet", "negated"]) \
             .drop_duplicates(["facility_id", "service", "kind", "field", "snippet"])
print(f"{len(evidence):,} citations")

ev = evidence.copy()
ev["p"] = np.where(ev.kind == "core", 0.55, 0.20) * ev.field.map(FIELD_WEIGHT).fillna(0.5) * np.where(ev.negated, 0.3, 1.0)
ev["l"] = np.log1p(-ev.p.clip(upper=0.99))
agg = ev.groupby(["facility_id", "service", "kind"]).agg(l=("l", "sum"), n_fields=("field", "nunique"), n_neg=("negated", "sum")).reset_index()
wide = agg.pivot_table(index=["facility_id", "service"], columns="kind", values=["l", "n_fields", "n_neg"], fill_value=0)
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()
for c in ["l_core", "l_support", "n_fields_core", "n_neg_core"]:
    if c not in wide: wide[c] = 0.0
core = 1 - np.exp(wide.l_core); support = np.minimum(1 - np.exp(wide.l_support), 0.6)
wide["evidence_raw"] = np.clip(core + (1 - core) * 0.35 * support + np.where(wide.n_fields_core >= 2, 0.10, 0) * (1 - core), 0, 1)

text_of = fac.set_index("facility_id").apply(lambda r: " | ".join(t for _, t in facility_items(r)), axis=1)
ftype_of = fac.set_index("facility_id").facility_type.fillna("").str.lower()
miss, mult = [], []
for fid, svc in zip(wide.facility_id, wide.service):
    txt = text_of.get(fid, "")
    m = [k for k, rx in COMPILED[svc]["prereq"].items() if not rx.search(txt)]
    w = 0.85 ** len(m)
    if re.search(r"pharmac|dentist|dental|laborator", ftype_of.get(fid, "")) and SERVICES[svc]["severity"] == "high":
        m.append(f"suitable facility type (listed as '{ftype_of.get(fid, '')}')"); w *= 0.5
    miss.append(m); mult.append(w)
wide["missing_prereqs"] = miss
wide["evidence"] = wide.evidence_raw * np.array(mult)
wide["tier"] = pd.cut(wide.evidence, [-0.01, 0.2, 0.5, 0.75, 1.01], labels=["Unverified", "Weak", "Moderate", "Strong"]).astype(str)
scores = wide[["facility_id", "service", "evidence", "tier", "n_fields_core", "n_neg_core", "missing_prereqs"]].copy()

tc = scores.groupby(["service", "tier"]).size().unstack(fill_value=0).reindex(columns=["Strong", "Moderate", "Weak", "Unverified"], fill_value=0)
tc["severity"] = [SERVICES[s]["severity"] for s in tc.index]
display(tc)
px.bar(tc.drop(columns="severity").reset_index().melt(id_vars="service"), x="service", y="value", color="tier",
       title="How many facilities can back up each service claim?",
       color_discrete_map={"Strong": "#1a7f37", "Moderate": "#9a6700", "Weak": "#d1242f", "Unverified": "#8c959f"}).update_layout(height=450).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Two independent witnesses: a facility-similarity graph and a masked-keyword text model
# MAGIC **5a. Graph propagation.** We build a k-nearest-neighbour graph of facilities by profile similarity (TF-IDF cosine, k = 10). For each service, `graph_score` is the similarity-weighted share of a facility's neighbours that have evidence for it: *"facilities that look like this one usually offer X"*. We check the homophily assumption before using it.
# MAGIC
# MAGIC **5b. Contextual plausibility.** A logistic regression trained on rule-derived labels, with the service's own keywords **masked out**, so it learns the *context* real providers share. Out-of-fold predictions only. Low plausibility on a strong claim triggers a "looks out of place" warning.

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.sparse import hstack, csr_matrix

full_text = fac.apply(lambda r: " ".join(t for _, t in facility_items(r)).lower(), axis=1)
X_all = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True, stop_words="english").fit_transform(full_text)
K = min(11, len(fac))
nn = NearestNeighbors(n_neighbors=K, metric="cosine").fit(X_all)
dist, idx = nn.kneighbors(X_all)
w_nb, idx_nb = np.clip(1 - dist[:, 1:], 0, 1), idx[:, 1:]          # drop self

fid_pos = pd.Series(np.arange(len(fac)), index=fac.facility_id)
graph_rows, homophily = [], []
for svc in SERVICES:
    e = np.zeros(len(fac))
    s = scores[scores.service == svc]
    e[fid_pos[s.facility_id].values] = s.evidence.values
    g = (w_nb * e[idx_nb]).sum(1) / (w_nb.sum(1) + 1e-9)
    graph_rows.append(pd.DataFrame({"facility_id": fac.facility_id, "service": svc, "graph_score": g}))
    has = e > 0.5
    homophily.append(dict(service=svc, mean_graph_if_strong_or_moderate=round(g[has].mean(), 3) if has.any() else None,
                          mean_graph_otherwise=round(g[~has].mean(), 3), corr=round(np.corrcoef(e, g)[0, 1], 3) if e.std() > 0 else None))
graph = pd.concat(graph_rows, ignore_index=True)
display(pd.DataFrame(homophily))

# COMMAND ----------

struct = np.column_stack([np.log1p(fac[c].fillna(0)) for c in ["n_doctors", "capacity", "n_claims", "desc_len", "n_specialties"]]
                         + [fac[c].notna().astype(float) for c in ["n_doctors", "capacity", "year_est"]])
struct = (struct - struct.mean(0)) / (struct.std(0) + 1e-9)
core_struct = evidence[(evidence.kind == "core") & ~evidence.negated & evidence.field.isin(["procedure", "equipment", "capability"])]
pos_sets = core_struct.groupby("service").facility_id.apply(set).to_dict()
any_hit = evidence.groupby("service").facility_id.apply(set).to_dict()
plaus_rows, model_report = [], []
for svc, d in SERVICES.items():
    masked = full_text.str.replace(alt(d["core"] + d["support"], r"\w*"), " ", regex=True)
    pos = fac.facility_id.isin(pos_sets.get(svc, set())).values
    neg = ~fac.facility_id.isin(any_hit.get(svc, set())).values
    train = pos | neg
    if pos.sum() < 15 or neg.sum() < 15:
        model_report.append(dict(service=svc, positives=int(pos.sum()), note="too few weak labels")); continue
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=40000, sublinear_tf=True, stop_words="english")
    Xt = vec.fit_transform(masked); X = hstack([Xt, csr_matrix(struct)]).tocsr(); y = pos[train].astype(int)
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    oof = cross_val_predict(clf, X[train], y, cv=StratifiedKFold(min(5, y.sum(), (1 - y).sum()), shuffle=True, random_state=0), method="predict_proba")[:, 1]
    clf.fit(X[train], y); pr = clf.predict_proba(X)[:, 1]; pr[np.where(train)[0]] = oof
    plaus_rows.append(pd.DataFrame({"facility_id": fac.facility_id, "service": svc, "plaus_pct": pd.Series(pr).rank(pct=True).values}))
    vocab = np.array(vec.get_feature_names_out()); co = clf.coef_[0][: Xt.shape[1]]
    model_report.append(dict(service=svc, positives=int(pos.sum()), negatives=int(neg.sum()), oof_auc=round(roc_auc_score(y, oof), 3),
                             oof_ap=round(average_precision_score(y, oof), 3), top_context_terms=", ".join(vocab[np.argsort(-co)[:10]])))
plaus = pd.concat(plaus_rows, ignore_index=True) if plaus_rows else pd.DataFrame(columns=["facility_id", "service", "plaus_pct"])
model_report = pd.DataFrame(model_report)
display(model_report)

scores = scores.merge(graph, on=["facility_id", "service"], how="left").merge(plaus, on=["facility_id", "service"], how="left")
scores["graph_score"] = scores.graph_score.fillna(0); scores["plaus_pct"] = scores.plaus_pct.fillna(0.5)
scores["flag_out_of_place"] = scores.tier.isin(["Strong", "Moderate"]) & (scores.plaus_pct < 0.15)
print("Strong/moderate claims whose context looks out of place:", int(scores.flag_out_of_place.sum()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Government registry verification (PM-JAY, state schemes, de-empanelled / suspended lists)
# MAGIC **Getting the lists.** PM-JAY publishes empanelled hospitals through its *Find Hospital* portal (hospitals.pmjay.gov.in) and public dashboard, and state scheme sites publish PDFs. Some states publish separate empanelled, de-empanelled, inactive and suspended lists (Haryana does). Convert them to CSV and upload to the Volume shown below. Files whose names contain `deempanel`, `de-empanel`, `suspend` or `inactive` are treated as **negative** lists; everything else as **positive** (empanelled). Columns are matched loosely (hospital name, state, district, pincode, phone, specialities, date).
# MAGIC
# MAGIC **Linking** is Fellegi–Sunter-style: Indian-name normalisation (strips generic words like *hospital / multispeciality / pvt ltd*, plus phonetic folding for transliteration: *ee→i, sh→s, w→v, bh→b*, doubled letters), then agreement weights on name similarity, PIN, district and phone combined in log-odds into a **match probability**. Weights are set by hand and documented; ≥ 0.8 = linked, 0.6–0.8 = possible match.
# MAGIC
# MAGIC Registries are **referees, not oracles** (the CAG has audited empanelment as imperfect): they raise or lower confidence, they never override evidence.

# COMMAND ----------

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.registries")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.exports")
    tmpl = os.path.join(REGISTRY_DIR, "_TEMPLATE_pmjay_empanelled.csv.txt")
    if not os.path.exists(tmpl):
        pd.DataFrame(columns=["hospital_id", "hospital_name", "state", "district", "pincode", "phone", "specialities", "hospital_type", "empanelment_date"]).to_csv(tmpl, index=False)
        pd.DataFrame(columns=["hospital_id", "hospital_name", "state", "district", "pincode", "action", "action_date", "reason"]).to_csv(
            os.path.join(REGISTRY_DIR, "_TEMPLATE_deempanelled.csv.txt"), index=False)
except Exception as e:
    print("Volume setup skipped:", e)

REG_FIELDS = {"reg_id": ["hospitalid", "hospid", "hospitalcode", "registration", "id"], "reg_name": ["hospitalname", "nameofhospital", "name", "hospital"],
              "reg_state": ["state"], "reg_district": ["district", "city"], "reg_pin": ["pincode", "pin", "postcode", "zip"],
              "reg_phone": ["phone", "mobile", "contact"], "reg_specialties": ["speciali", "specialt", "services", "packages"],
              "reg_date": ["actiondate", "date", "empanelmentdate"], "reg_action": ["action", "status"]}

def load_registry(path):
    df = pd.read_csv(path, dtype=str, encoding_errors="ignore")
    nm = {_nc(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for f, cands in REG_FIELDS.items():
        col = next((nm[k] for c in cands for k in nm if k == c), None) or next((nm[k] for c in cands for k in nm if c in k), None)
        out[f] = df[col] if col else None
    base = os.path.basename(path).lower()
    out["list_kind"] = ("suspended" if "suspend" in base else "inactive" if "inactive" in base
                        else "de-empanelled" if re.search(r"de.?empanel", base) else "empanelled")
    out["source_file"] = os.path.basename(path)
    return out

reg_files = [p for p in glob.glob(os.path.join(REGISTRY_DIR, "*.csv"))] if os.path.isdir(REGISTRY_DIR) else []
registry = pd.concat([load_registry(p) for p in reg_files], ignore_index=True) if reg_files else pd.DataFrame()
print(f"{len(reg_files)} registry files, {len(registry):,} rows")
if len(registry): display(registry.list_kind.value_counts().to_frame())

# COMMAND ----------

GENERIC_NAME = GENERIC | {"hosp", "pvt", "private", "ltd", "limited", "llp", "trust", "superspeciality", "multi", "speciality", "dr", "shri", "sri", "smt", "and", "co"}
FOLD = [("ee", "i"), ("oo", "u"), ("aa", "a"), ("sh", "s"), ("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"),
        ("gh", "g"), ("ck", "k"), ("w", "v"), ("z", "j"), ("q", "k"), ("y", "i")]
def name_key(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = " ".join(t for t in s.split() if t not in GENERIC_NAME)
    for a, b in FOLD: s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s).strip()
try:
    from rapidfuzz import fuzz
    sim = lambda a, b: fuzz.token_set_ratio(a, b) / 100.0
except ImportError:
    sim = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
digits = lambda v: set(re.findall(r"\d{10}", re.sub(r"\D", "", str(v))[-10:] if v else ""))

def match_prob(s, pin6, pin3, dist_ok, phone, short, n_same):
    # Fellegi-Sunter idea: agreeing on a *common* name is weak evidence, so the name weight shrinks with the
    # number of facilities in the same city sharing that name key (n_same).
    logit = -7 + 9 * s + 2.5 * pin6 + 1.0 * pin3 + 1.2 * dist_ok + 3.0 * phone - 1.5 * short - 2.0 * math.log2(max(n_same, 1))
    return 1 / (1 + math.exp(-logit))

fac["reg_status"], fac["reg_match_p"], fac["reg_id"], fac["reg_specialties"], fac["reg_action_date"] = "not checked", np.nan, None, None, None
link_log = []
if len(registry):
    registry["state_norm"] = registry.reg_state.fillna("").map(lambda s: (difflib.get_close_matches(re.sub(r"[^a-z ]", "", s.lower()).strip(),
                                                         fac.state_norm.dropna().unique(), n=1, cutoff=0.7) or [None])[0])
    registry["dist_norm"] = registry.reg_district.fillna("").str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
    registry["pin"] = registry.reg_pin.fillna("").str.replace(r"\D", "", regex=True).str[:6]
    registry["key"] = registry.reg_name.fillna("").map(name_key)
    registry["phones"] = registry.reg_phone.map(lambda v: {re.sub(r"\D", "", p)[-10:] for p in re.split(r"[,/;]", str(v)) if len(re.sub(r"\D", "", p)) >= 10})
    by_state = {s: g for s, g in registry.groupby("state_norm")}
    fac["name_key"] = fac["name"].map(name_key)
    key_freq = fac.groupby(["name_key", "city_key"]).facility_id.transform("size")
    fac["reg_status"] = "not found"
    pairs = []
    for i, r in fac.iterrows():
        g = by_state.get(r.state_norm)
        if g is None: continue
        pin = r.postcode if isinstance(r.postcode, str) else ""
        dn = r.district_norm if isinstance(r.district_norm, str) else None
        cn = r.city_norm if isinstance(r.city_norm, str) else None
        blk = g[(g.pin.str[:3] == pin[:3]) | (g.dist_norm == (dn or cn or "#"))] if pin or cn else g
        if blk.empty: continue
        k = r.name_key; fph = {re.sub(r"\D", "", p)[-10:] for p in r.phone if len(re.sub(r"\D", "", p)) >= 10}
        best = {}
        for j, x in blk.iterrows():
            p = match_prob(sim(k, x.key), float(bool(pin) and x.pin == pin), float(bool(pin) and x.pin[:3] == pin[:3]),
                           float(x.dist_norm in {dn, cn}), float(bool(fph & x.phones)), float(len(k) < 4), key_freq[i])
            if p >= 0.6:
                pairs.append((p, i, j, "pos" if x.list_kind == "empanelled" else "neg"))
    # One-to-one assignment: each registry row links to at most one facility (and vice versa), best matches first
    used_reg, used_fac = set(), set()
    for p, i, j, kind in sorted(pairs, reverse=True):
        if j in used_reg or (i, kind) in used_fac or (kind == "neg" and p < 0.8):
            continue
        used_reg.add(j); used_fac.add((i, kind)); x = registry.loc[j]
        if kind == "pos" and not str(fac.at[i, "reg_status"]).startswith(("de-", "susp", "inact")):
            fac.at[i, "reg_status"] = "empanelled" if p >= 0.8 else "empanelled (possible match)"
            fac.at[i, "reg_match_p"], fac.at[i, "reg_id"], fac.at[i, "reg_specialties"] = p, x.reg_id, x.reg_specialties
        elif kind == "neg":
            fac.at[i, "reg_status"], fac.at[i, "reg_match_p"], fac.at[i, "reg_action_date"] = x.list_kind, p, x.reg_date
        link_log.append(dict(kind=kind, match_p=p))
    display(fac.reg_status.value_counts().to_frame("facilities"))
    if link_log:
        px.histogram(pd.DataFrame(link_log), x="match_p", color="kind", nbins=40, title="Match probability of accepted links").show()

def years_since(d):
    try: return max(0.0, (pd.Timestamp.today() - pd.to_datetime(d, dayfirst=True)).days / 365.25)
    except Exception: return 0.0

def reg_factor(r):
    st = r.reg_status
    if st == "empanelled": return 0.92
    if st == "empanelled (possible match)": return 0.86
    if st in ("de-empanelled", "suspended", "inactive"):
        return 0.8 - 0.6 * math.exp(-years_since(r.reg_action_date) / 2.0)   # recent action weighs more
    if st == "not found": return 0.72
    return 0.80                                                               # not checked = neutral
fac["reg_factor"] = fac.apply(reg_factor, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Calibrated probability that a facility really offers the service, with a range
# MAGIC If enough facilities are **linked** to scheme lists that carry specialities, we train a calibrated logistic regression on *[evidence, graph score, plausibility, trust, cross-field corroboration, negation]* with label = *"registry lists the matching speciality"*, and report out-of-fold AUC, Brier score and a calibration curve. Otherwise we fall back to a documented heuristic. Linked registry specialities then update the odds (×4 if the speciality matches, ×0.6 if linked but the speciality is absent).
# MAGIC
# MAGIC Every estimate becomes a **Beta distribution** whose concentration grows with documentation depth and registry linkage, giving the 5–95% range shown to users ("likely, 72–88%").

# COMMAND ----------

from scipy.stats import beta as beta_dist
S = scores.merge(fac[["facility_id", "trust", "documentation", "reg_status", "reg_match_p", "reg_specialties"]], on="facility_id")
S["corrob"] = (S.n_fields_core >= 2).astype(float); S["has_neg"] = (S.n_neg_core > 0).astype(float)
S["pmjay_rx"] = S.service.map(lambda s: SERVICES[s].get("pmjay"))
linked = (S.reg_status == "empanelled") & (S.reg_match_p >= 0.8) & S.reg_specialties.notna() & S.pmjay_rx.notna()
S["reg_specialty_match"] = [bool(l and re.search(rx, str(sp), re.I)) for l, rx, sp in zip(linked, S.pmjay_rx, S.reg_specialties)]
FEATS = ["evidence", "graph_score", "plaus_pct", "trust", "corrob", "has_neg"]
heuristic = np.clip(0.02 + 0.75 * S.evidence + 0.12 * S.graph_score + 0.08 * S.plaus_pct - 0.1 * S.has_neg, 0.01, 0.97)
CALIBRATION = "heuristic (no registry labels loaded)"
if linked.sum() >= 60 and 20 <= S.loc[linked, "reg_specialty_match"].sum() <= linked.sum() - 20:
    Xl, yl = S.loc[linked, FEATS].values, S.loc[linked, "reg_specialty_match"].astype(int).values
    lr = LogisticRegression(max_iter=1000)
    oof = cross_val_predict(lr, Xl, yl, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    lr.fit(Xl, yl)
    p0 = lr.predict_proba(S[FEATS].values)[:, 1]; p0[np.where(linked)[0]] = oof
    auc_model, auc_raw = roc_auc_score(yl, oof), roc_auc_score(yl, S.loc[linked, "evidence"])
    USE_MODEL = auc_model >= 0.6 and auc_model >= auc_raw     # guard: never replace evidence with a weaker model
    CALIBRATION = ("" if USE_MODEL else "REJECTED (kept heuristic) — ") + f"logistic on {int(linked.sum())} registry-linked pairs: AUC {roc_auc_score(yl, oof):.3f}, Brier {brier_score_loss(yl, oof):.3f} (vs. raw evidence AUC {roc_auc_score(yl, S.loc[linked,'evidence']):.3f})"
    bins = pd.qcut(oof, 8, duplicates="drop")
    cal = pd.DataFrame({"pred": oof, "obs": yl}).groupby(bins).mean()
    fig = px.line(cal, x="pred", y="obs", markers=True, title="Calibration on held-out registry labels"); fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect"); fig.show()
    S["p_capable"] = p0 if USE_MODEL else heuristic
else:
    S["p_capable"] = heuristic
print("P(capable) model:", CALIBRATION)
odds = S.p_capable / (1 - S.p_capable)
odds = np.where(S.reg_specialty_match, odds * 4.0, np.where(linked & ~S.reg_specialty_match, odds * 0.6, odds))
S["p_capable"] = np.clip(odds / (1 + odds), 0.01, 0.99)
kappa = 4 + 20 * S.documentation + np.where(S.reg_status.str.startswith("empanelled"), 10, 0)
S["p_lo"] = beta_dist.ppf(0.05, S.p_capable * kappa, (1 - S.p_capable) * kappa)
S["p_hi"] = beta_dist.ppf(0.95, S.p_capable * kappa, (1 - S.p_capable) * kappa)
scores = S[["facility_id", "service", "evidence", "tier", "p_capable", "p_lo", "p_hi", "graph_score", "plaus_pct",
            "n_fields_core", "n_neg_core", "missing_prereqs", "flag_out_of_place", "reg_specialty_match"]].copy()
px.histogram(scores, x="p_capable", color="tier", nbins=50, title="P(facility really offers the service) by evidence tier").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Severity classifier
# MAGIC **Hard-coded clinical priors** set each service's default severity and clock (time window and benefit decay), stated openly as assumptions:
# MAGIC
# MAGIC | Service | Severity | Window | Basis |
# MAGIC |---|---|---|---|
# MAGIC | Trauma, ICU | high | 60 min | "golden hour" |
# MAGIC | Heart attack | high | 90 min | door-to-balloon target for angioplasty |
# MAGIC | Emergency C-section / PPH, NICU, burns, snakebite | high | 120 min | postpartum haemorrhage kills within ~2 h; anti-venom early |
# MAGIC | Stroke | high | 270 min | clot-busting window ≈ 4.5 h |
# MAGIC | Dialysis, MRI, CT, chemo, planned heart surgery | low | days–weeks | scheduled care |
# MAGIC
# MAGIC Free text is then scanned for **red flags** in English, Hindi and Hinglish (*behosh, bahut khoon, saans nahi, saanp ne kaata…*) and planned-care cues. Everything combines in log-odds, and the threshold comes from an **asymmetric cost**: missing an emergency is 6× worse than over-calling one, so the Bayes-optimal threshold is 1/(1+6) ≈ 0.14. A "routine dialysis" query with *"can't breathe"* escalates to an emergency.

# COMMAND ----------

display(pd.DataFrame([{"service": k, "label": v["label"], "severity": v["severity"],
                       "window_min": v.get("clock", {}).get("window"), "tau_min": v.get("clock", {}).get("tau")}
                      for k, v in SERVICES.items() if not v.get("hidden")]))

TRIAGE_TESTS = [  # hand-written safety test set: (query, expected severity)
    ("accident hua hai, behosh hai", "high"), ("road accident near highway, leg broken, bleeding", "high"),
    ("bahut khoon beh raha hai delivery ke baad", "high"), ("pregnant woman having fits", "high"),
    ("labour pain started, water broke", "high"), ("newborn baby not breathing properly", "high"),
    ("premature baby needs nicu", "high"), ("chest pain and sweating for 30 minutes", "high"),
    ("seene mein dard ho raha hai", "high"), ("father suddenly cannot speak, face drooping", "high"),
    ("lakwa maar gaya", "high"), ("saanp ne kaat liya", "high"), ("snake bite on farm", "high"),
    ("child drank pesticide", "high"), ("zeher kha liya", "high"), ("burned with hot oil, large area", "high"),
    ("jal gaya hai haath", "high"), ("unconscious after fall", "high"), ("can't breathe, lips blue", "high"),
    ("seizure won't stop", "high"), ("dialysis patient now breathless and confused", "high"),
    ("heavy bleeding after miscarriage", "high"), ("need icu bed urgently", "high"), ("trauma care needed", "high"),
    ("dialysis twice a week near Gaya", "low"), ("where can I get an MRI next week", "low"),
    ("routine kidney checkup", "low"), ("chemotherapy for my mother, planned", "low"), ("ct scan for headache report", "low"),
    ("planned bypass surgery cost", "low"), ("need dialysis centre near Patna", "low"), ("MRI for knee", "low"),
    ("second opinion for cancer treatment", "low"), ("follow-up dialysis appointment", "low"),
    ("which hospital does radiotherapy", "low"), ("heart surgery planned next month", "low"),
]
tt = pd.DataFrame(TRIAGE_TESTS, columns=["query", "expected"])
res = tt["query"].map(engine.triage)
tt["predicted"] = res.map(lambda r: r["severity"]); tt["p_high"] = res.map(lambda r: round(r["p_high"], 2))
tt["service"] = res.map(lambda r: r["service"]); tt["why"] = res.map(lambda r: "; ".join(r["reasons"]))
missed = ((tt.expected == "high") & (tt.predicted == "low")).sum(); over = ((tt.expected == "low") & (tt.predicted == "high")).sum()
triage_eval = pd.DataFrame([dict(test_queries=len(tt), emergencies=int((tt.expected == "high").sum()),
                                 missed_emergencies=int(missed), missed_emergency_rate=round(missed / (tt.expected == "high").sum(), 3),
                                 over_calls=int(over), over_call_rate=round(over / (tt.expected == "low").sum(), 3))])
display(triage_eval); display(tt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Low-severity branch: best facility for the service
# MAGIC Ranking: `score = P(capable)^0.45 × trust^0.20 × access^0.25 × registry^0.10`, with `access = exp(−road km / D_service)`. A weighted geometric mean, so a facility can't win on one dimension alone. We re-rank 300× with weights jittered ±40% to report stability, and draw the **Pareto frontier** (no other facility is both closer *and* more likely to offer the service), which makes the distance-vs-certainty trade-off visible.

# COMMAND ----------

fac_gold = fac[["facility_id", "name", "address", "city", "district", "state", "postcode", "lat", "lon", "sigma_km", "coord_source",
                "geo_ok", "trust", "documentation", "flag_geo_state", "flag_geo_city", "flag_pin_state", "flag_merge", "flag_overclaim",
                "dup_group", "dup_size", "capacity", "n_doctors", "year_est", "facility_type", "phone", "source_url",
                "city_norm", "district_norm", "state_norm", "reg_status", "reg_match_p", "reg_id", "reg_specialties",
                "reg_action_date", "reg_factor"]].copy()

def low_demo(service, place, k=3):
    loc = engine.resolve_location(place, fac_gold)
    out = engine.rank_low(service, loc, fac_gold, scores, k=k)
    r = out["results"]
    print(f"{SERVICES[service]['label']} near {loc['label']} ({loc['precision']})")
    display(r[["name", "city", "tier", "p_capable", "p_lo", "p_hi", "trust", "reg_status", "road_km", "score", "p_top1"]].round(2))
    c = out["candidates"].copy(); c["pick"] = c.facility_id.isin(r.facility_id)
    fig = px.scatter(c, x="road_km", y="p_capable", color="tier", symbol="pick", hover_name="name", error_y=c.p_hi - c.p_capable,
                     error_y_minus=c.p_capable - c.p_lo, title=f"Distance vs. likelihood of {SERVICES[service]['label']} (Pareto frontier in black)",
                     color_discrete_map={"Strong": "#1a7f37", "Moderate": "#9a6700", "Weak": "#d1242f"})
    p = out["pareto"].sort_values("road_km"); fig.add_scatter(x=p.road_km, y=p.p_capable, mode="lines", line=dict(color="black", dash="dot"), name="Pareto frontier")
    fig.update_layout(height=450); fig.show()
    for _, row in r.head(1).iterrows():
        print("Why:", [q["snippet"] for q in engine.quotes(evidence, row.facility_id, service)])
        print("Registry:", engine.registry_line(row)[1]); print("Caveats:", engine.caveats(row))
    return out

_ = low_demo("dialysis", "Gaya, Bihar")
_ = low_demo("mri", "Jaipur")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. High-severity branch: Monte Carlo over uncertain capability, location and travel time
# MAGIC The decision is *"which facility should the patient go to first?"*. In each of 1,000 simulated worlds we sample:
# MAGIC - whether each candidate **really** can give definitive care (`P(capable)`) or at least **stabilise** (24/7 emergency evidence), and whether it's operating (from trust);
# MAGIC - where it really is (repaired location ± `sigma_km`), and where the patient is (location-resolution precision);
# MAGIC - the road detour factor and travel speed (log-normal).
# MAGIC
# MAGIC Outcome per world uses the condition's clock (benefit `exp(−t/τ)`):
# MAGIC - **can treat** → `exp(−t/τ)`;
# MAGIC - **can only stabilise** → stabilised on arrival, then the clock runs 4× slower during handover and transfer to the nearest capable facility. This is the stabilise-then-transfer plan (consistent with PM-RAHAT, under which PM-JAY hospitals provide stabilisation to road-accident victims);
# MAGIC - **neither** → turned away after 10 minutes and redirected unstabilised.
# MAGIC
# MAGIC We report expected benefit, **how often each choice is the best in hindsight**, and the chance of reaching competent care within the window ("about 8 in 10"). The baseline is the naive *nearest facility that mentions the service*.

# COMMAND ----------

def high_demo(service, place):
    loc = engine.resolve_location(place, fac_gold)
    out = engine.simulate_emergency(service, loc, fac_gold, scores, n_worlds=1000)
    p = out["plans"]
    print(f"{SERVICES[service]['label']} near {loc['label']} — window {out['window']} min")
    display(p[["name", "role", "p_def", "p_stab", "minutes_median", "exp_benefit", "win_pct", "p_first_in_window",
               "p_definitive_in_window", "typical_transfer"]].head(6).round(2))
    if out["baseline"]: print("Naive nearest-mention baseline:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out["baseline"].items()})
    fig = go.Figure()
    fig.add_histogram(x=out["t_first_best"], name="first competent care", opacity=0.6)
    fig.add_histogram(x=out["t_defin_best"], name="definitive care", opacity=0.6)
    fig.add_vline(x=out["window"], line_dash="dash", annotation_text="clinical window")
    fig.update_layout(barmode="overlay", title=f"Minutes to care if going first to {p.name.iloc[0]} (1,000 simulated trips)", height=380); fig.show()
    return out

_ = high_demo("trauma", "Lucknow")
_ = high_demo("emergency_csection", "Nashik, Maharashtra")
_ = high_demo("snakebite", "Gaya")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Validation: what does each branch do better than a naive lookup?
# MAGIC Simulated queries from city centroids across India. **Low severity:** how often the naive *nearest facility that mentions the service* is weak-evidence or red-flagged, vs. our pick. **High severity:** expected time-adjusted benefit and P(competent care within the window) for our Monte Carlo pick vs. the naive nearest-mention facility, **evaluated in the same simulated worlds**.

# COMMAND ----------

ref_pts = fac_gold[fac_gold.geo_ok].groupby(["city_norm", "state_norm"])[["lat", "lon"]].median().reset_index()
pts = ref_pts.sample(min(60, len(ref_pts)), random_state=3)
low_rows, high_rows = [], []
for svc in ["dialysis", "mri", "cancer_chemo"]:
    for _, q in pts.iterrows():
        loc = dict(lat=q.lat, lon=q.lon, sigma_km=5)
        out = engine.rank_low(svc, loc, fac_gold, scores, k=1, n_sim=20)
        c = out["candidates"]
        if len(c) == 0: continue
        naive, ours = c.sort_values("km").iloc[0], out["results"].iloc[0]
        bad = lambda r: r.tier == "Weak" or r.trust < 0.6 or bool(r.flag_out_of_place) or r.reg_status in ("de-empanelled", "suspended")
        low_rows.append(dict(service=svc, naive_bad=bad(naive), ours_bad=bad(ours), extra_km=ours.road_km - naive.road_km))
for svc in ["trauma", "emergency_csection", "heart_attack"]:
    for _, q in pts.head(30).iterrows():
        out = engine.simulate_emergency(svc, dict(lat=q.lat, lon=q.lon, sigma_km=5), fac_gold, scores, n_worlds=400)
        if out is None or out["baseline"] is None: continue
        b = out["plans"].iloc[0]
        high_rows.append(dict(service=svc, ours_benefit=b.exp_benefit, naive_benefit=out["baseline"]["exp_benefit"],
                              ours_in_window=b.p_first_in_window, naive_in_window=out["baseline"]["p_first_in_window"]))
low_abl = pd.DataFrame(low_rows).groupby("service").agg(queries=("naive_bad", "size"), naive_low_confidence_pct=("naive_bad", lambda s: round(100 * s.mean(), 1)),
                                                        ours_low_confidence_pct=("ours_bad", lambda s: round(100 * s.mean(), 1)),
                                                        median_extra_km=("extra_km", "median")).reset_index()
high_abl = pd.DataFrame(high_rows).groupby("service").mean().round(3).reset_index()
display(low_abl); display(high_abl)
fig = px.bar(high_abl.melt(id_vars="service", value_vars=["naive_in_window", "ours_in_window"]), x="service", y="value", color="variable",
             barmode="group", title="P(competent care within the clinical window): naive nearest-mention vs CareCompass")
fig.update_layout(yaxis_tickformat=".0%"); fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Impact
# MAGIC ### 12a. Golden-hour coverage: stabilising vs definitive care
# MAGIC For each city, estimated drive time to the nearest facility likely (`P ≥ 0.5`) to **stabilise** vs. to give **definitive** care. The gap shows where a stabilise-then-transfer protocol (and upgrading stabilising hospitals) changes outcomes most.

# COMMAND ----------

def nearest_minutes(points, sub):
    if sub.empty: return np.full(len(points), np.nan)
    A, B = np.radians(points[["lat", "lon"]].values), np.radians(sub[["lat", "lon"]].values)
    best = np.full(len(A), np.inf)
    for i in range(0, len(B), 2000):
        b = B[i:i + 2000]
        a = np.sin((A[:, None, 0] - b[None, :, 0]) / 2) ** 2 + np.cos(A[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin((A[:, None, 1] - b[None, :, 1]) / 2) ** 2
        best = np.minimum(best, (6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).min(1))
    return best * engine.ROAD_FACTOR / engine.EMERGENCY_SPEED_KMH * 60

care_gaps = ref_pts.copy()
cap = scores.merge(fac_gold[["facility_id", "lat", "lon", "coord_source"]], on="facility_id")
cap = cap[cap.coord_source != "none"]
stab = cap[(cap.service == "emergency_24x7") & (cap.p_capable >= 0.5)]
care_gaps["min_to_stabilise"] = nearest_minutes(care_gaps, stab)
cov_rows = []
for svc in [s for s in SERVICES if SERVICES[s]["severity"] == "high" and not SERVICES[s].get("hidden")]:
    care_gaps[f"min_to_{svc}"] = nearest_minutes(care_gaps, cap[(cap.service == svc) & (cap.p_capable >= 0.5)])
    w = SERVICES[svc]["clock"]["window"]
    cov_rows.append(dict(service=SERVICES[svc]["label"], window_min=w,
                         cities_with_stabilisation_in_window=round(100 * (care_gaps.min_to_stabilise <= w).mean(), 1),
                         cities_with_definitive_in_window=round(100 * (care_gaps[f"min_to_{svc}"] <= w).mean(), 1)))
for svc in [s for s in SERVICES if SERVICES[s]["severity"] == "low"]:
    care_gaps[f"min_to_{svc}"] = nearest_minutes(care_gaps, cap[(cap.service == svc) & (cap.p_capable >= 0.5)])
coverage = pd.DataFrame(cov_rows)
display(coverage)
px.bar(coverage.melt(id_vars=["service", "window_min"]), x="service", y="value", color="variable", barmode="group",
       title="% of cities reaching care within the clinical window").update_layout(height=430, yaxis_title="% of cities").show()
fig = px.scatter_geo(care_gaps, lat="lat", lon="lon", color=care_gaps.min_to_trauma.clip(upper=240), hover_name="city_norm",
                     color_continuous_scale="RdYlGn_r", title="Drive minutes to likely definitive trauma care")
fig.update_geos(fitbounds="locations", showcountries=True); fig.update_layout(height=620); fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 12b. Verification priorities: whose phone call shrinks a care desert most?
# MAGIC A facility with moderate evidence in a place where the nearest likely provider is far away is the highest-value thing to verify.

# COMMAND ----------

def verification_priorities(service, top=15):
    s = scores[(scores.service == service) & scores.tier.isin(["Weak", "Moderate"])].merge(fac_gold, on="facility_id")
    s = s[s.geo_ok]
    strong = scores[(scores.service == service) & (scores.p_capable >= 0.7)].merge(fac_gold[["facility_id", "lat", "lon"]], on="facility_id")
    if s.empty or strong.empty: return pd.DataFrame()
    s["km_to_likely_provider"] = [haversine_km(r.lat, r.lon, strong.lat.values, strong.lon.values).min() * engine.ROAD_FACTOR for r in s.itertuples()]
    s["priority"] = s.km_to_likely_provider * s.p_capable * (0.5 + s.plaus_pct)
    return s.sort_values("priority", ascending=False)[["name", "city", "state", "tier", "p_capable", "reg_status", "km_to_likely_provider"]].head(top).round(2)
for svc in ["dialysis", "emergency_csection", "trauma"]:
    print(SERVICES[svc]["label"]); display(verification_priorities(svc))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 12c. Policy recommendations (fill in numbers from the cells above)
# MAGIC 1. **Route emergencies by time-to-competent-hands, not by name or size.** In our Monte Carlo ablation, the naive "nearest facility that mentions it" reaches competent care within the window less often than our pick. Referral apps used by ASHAs should default to stabilise-then-transfer plans.
# MAGIC 2. **Fund stabilisation where definitive care is far.** The coverage gap (12a) shows many cities with stabilising care in the window but no definitive care. Low-cost upgrades there (blood storage, trained emergency staff, PM-RAHAT stabilisation empanelment) buy the most time.
# MAGIC 3. **Verify where it matters.** A volunteer phone-bank working the 12b priority list closes care deserts faster than re-scraping 10,000 records.
# MAGIC 4. **Publish a machine-readable registry.** Linking would be near-perfect if PM-JAY / state lists were CSVs with hospital IDs, PIN codes and phone numbers. The de-empanelled and suspended lists in particular should be in one national file.
# MAGIC
# MAGIC **Limitations.** City centroids stand in for population; there's no road network (fixed detour factor + speed distribution); severity rules are a decision-support aid, not a clinical triage system, and every emergency answer starts with "call 108"; the plausibility and calibration models learn from rule-derived or registry labels and inherit their biases.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Gold tables + export for the dashboard

# COMMAND ----------

def to_storable(df):
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, (list, np.ndarray, set))).any():
            out[c] = out[c].map(lambda v: json.dumps(list(v)) if isinstance(v, (list, np.ndarray, set)) else v)
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))
    return out

ev_gold = (evidence.assign(o=(evidence.kind != "core").astype(int) * 2 + evidence.negated.astype(int))
           .sort_values("o").groupby(["facility_id", "service"]).head(6).drop(columns="o"))
GOLD = {"facilities": (fac_gold, "One row per facility: repaired location, uncertainty radius, trust, flags, registry status"),
        "facility_service_scores": (scores, "Facility x service: evidence tier, calibrated P(capable) with 5-95% range, graph and plausibility scores"),
        "service_evidence": (ev_gold, "Citations: text snippets supporting each facility-service claim"),
        "care_gaps": (care_gaps, "Per city: estimated drive minutes to likely stabilising and definitive care per service"),
        "triage_eval": (tt, "Severity classifier safety test set with predictions"),
        "model_report": (model_report, "Plausibility model out-of-fold metrics"),
        "coverage_summary": (coverage, "% of cities with stabilising vs definitive care within the clinical window")}
for name, (df, comment) in GOLD.items():
    sdf = to_storable(df)
    try:
        t = f"{OUTPUT_CATALOG}.{OUTPUT_SCHEMA}.{name}"
        spark.createDataFrame(sdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(t)
        spark.sql(f"COMMENT ON TABLE {t} IS '{comment}'")
        print("table ", t, len(sdf))
    except Exception as e:
        print("table write failed:", name, e)
    try:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        sdf.to_parquet(os.path.join(EXPORT_DIR, f"{name}.parquet"), index=False)
    except Exception as e:
        print("export failed:", name, e)
print("Parquet snapshot for local hosting:", EXPORT_DIR)
