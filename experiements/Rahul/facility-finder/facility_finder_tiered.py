# Databricks notebook source
# MAGIC %md
# MAGIC # Facility Finder: criticality-tiered, evidence-based facility recommendations
# MAGIC **CMU Datathon Fall 2026 · Virtue Foundation India dataset · runs on Databricks Free Edition**
# MAGIC
# MAGIC **Input:** a *service* (free text, e.g. "emergency c-section", "dialysis", "cancer care") and a *location*
# MAGIC (city, district, village or post-office name, 6-digit PIN code, or `lat,lon`).
# MAGIC **Output:** the best few facility options. Each answers the two required questions in plain language:
# MAGIC * **WHY this facility?** Evidence that it exists, that it offers the service, and that it suits this location.
# MAGIC * **HOW did you decide?** The model, the numbers, the weights, and the trade-offs.
# MAGIC
# MAGIC ---
# MAGIC ## The plan in one picture
# MAGIC ```
# MAGIC  service text ──► [0] TRIAGE: criticality tier (High / Medium / Low) from clinical time-windows
# MAGIC  location text ─► [A] LOCATION: cleaned India Post PIN layer → origin as a cloud of possible points
# MAGIC                         │
# MAGIC  facilities ──► [B] EVIDENCE extraction (quotes) ──► [1] CLAIM COURT: Bayesian P(offers service)  ◄── [5] REGISTRY
# MAGIC                         │                                     (used by every tier)                (official lists,
# MAGIC                         ▼                                                                          used everywhere)
# MAGIC        ┌────────────────┼──────────────────────────────┐
# MAGIC   HIGH tier          MEDIUM tier                    LOW tier
# MAGIC   Monte-Carlo        Access & need model (E2SFCA     Care-desert context + LLM EVIDENCE COURT
# MAGIC   Plan-A/Plan-B      + NFHS need + Huff congestion   (prosecutor / defender / fact-checker /
# MAGIC   router: expected   + multi-criteria score +        judge), then a bounded Bayesian update
# MAGIC   outcome vs time    weight-robustness + Pareto)     and ranking
# MAGIC        └────────────────┴──────────────────────────────┘
# MAGIC                         ▼
# MAGIC        WHY / HOW explanation cards (HTML) + structured results table
# MAGIC ```
# MAGIC
# MAGIC ## Datasets
# MAGIC | Dataset | Role | Required? |
# MAGIC |---|---|---|
# MAGIC | `facilities` (Virtue Foundation, ~10k rows) | the claims we verify and rank | **yes** |
# MAGIC | `india_post_pincode_directory` | geography: PIN → points, district, rural/urban proxy | **yes** |
# MAGIC | `nfhs_5_district_health_indicators` | district need / context, priors | **yes** |
# MAGIC | `workspace.default.registry_facilities` | LaQshya certified labour rooms & maternity OTs: credibility layer (Idea 5) | optional; the notebook runs without it and says so |
# MAGIC | `workspace.default.district_population` | demand weights for the access model | optional; falls back to a post-office-count proxy |
# MAGIC
# MAGIC **How to run:** attach **Serverless** compute → **Run all**. Change the two widgets at the top → re-run the section **"11. Run a query"**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Configuration

# COMMAND ----------

CATALOG_SCHEMA = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
FACILITIES_TABLE = f"{CATALOG_SCHEMA}.facilities"
PIN_TABLE = f"{CATALOG_SCHEMA}.india_post_pincode_directory"
NFHS_TABLE = f"{CATALOG_SCHEMA}.nfhs_5_district_health_indicators"

# Optional tables you upload yourself (see the instructions that came with this notebook). Missing = skipped gracefully.
REGISTRY_TABLE = "workspace.default.registry_facilities"
POPULATION_TABLE = "workspace.default.district_population"
OUTPUT_SCHEMA = "workspace.default"

# LLMs for the low-criticality Evidence Court. The first one that answers a probe is used; if none answers,
# a transparent rule-based court is used instead (the notebook never fails because of the LLM).
LLM_CANDIDATES = [
    "databricks-meta-llama-3-3-70b-instruct", "system.ai.meta-llama-3-3-70b-instruct",
    "databricks-qwen3-next-80b-a3b-instruct", "system.ai.qwen3-next-80b-a3b-instruct",
    "databricks-gpt-oss-120b", "system.ai.gpt-oss-120b",
]
USE_LLM = True          # set False to force the rule-based court
MC_SIMULATIONS = 3000   # Monte-Carlo draws for the high-criticality router
RANDOM_SEED = 7

# COMMAND ----------

import re, json, math, html, difflib, time
import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", 120)
RNG = np.random.default_rng(RANDOM_SEED)

def table_exists(name):
    try:
        spark.table(name).limit(1).collect()
        return True
    except Exception:
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Small helpers (text cleaning, distances)
# MAGIC * In the facilities table every column is text and **missing values are the literal word `"null"`**. `to_text` and `to_list` treat those as empty.
# MAGIC * List fields are JSON text with repeats (`'["pediatrics","pediatrics"]'`), so `to_list` parses them and removes duplicates.
# MAGIC * `haversine_km` is the great-circle distance on a sphere (Earth radius 6371 km). Road distance is longer, and the models account for that.

# COMMAND ----------

NULLS = {"", "null", "none", "nan", "na", "n/a", "[]", "{}"}

def is_blank(v):
    return v is None or (isinstance(v, float) and v != v) or str(v).strip().lower() in NULLS

def to_text(v):
    return "" if is_blank(v) else str(v).strip()

def _dedupe(items):
    seen, out = set(), []
    for x in items:
        k = x.lower().strip()
        if k and k not in seen and k not in NULLS:
            seen.add(k); out.append(x.strip())
    return out

def to_list(v):
    if v is None or (isinstance(v, float) and v != v): return []
    if isinstance(v, (list, tuple, np.ndarray)): return _dedupe([str(x) for x in v if x is not None])
    s = str(v).strip()
    if s.lower() in NULLS: return []
    if s.startswith("["):
        try: return _dedupe([str(x) for x in json.loads(s) if x is not None])
        except Exception: s = s.strip("[]")
    return _dedupe([p.strip().strip('"\'') for p in re.split(r'",\s*"|;|\|', s)])

def to_num(v):
    try:
        x = float(str(v).replace(",", "").strip())
        return x if x == x and x > 0 else np.nan
    except Exception:
        return np.nan

def uncamel(s): return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)

def norm_name(s):
    s = to_text(s).lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()

STATE_ALIASES = {"nctofdelhi": "delhi", "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
                 "jammuandkashmir": "jammuandkashmir", "dadraandnagarhavelianddamananddiu": "dnhdd",
                 "dadraandnagarhaveli": "dnhdd", "damananddiu": "dnhdd", "dnhanddd": "dnhdd", "dnhavelianddamananddiu": "dnhdd",
                 "andamanandnicobarisland": "andamanandnicobar", "anislands": "andamanandnicobar", "andamanandnicobarislands": "andamanandnicobar",
                 "andamanandnicobar": "andamanandnicobar"}
def norm_state(s):
    k = re.sub(r"[^a-z]", "", to_text(s).lower().replace("&", "and"))
    return STATE_ALIASES.get(k, k)

def norm_district(s):
    k = norm_name(s)
    k = re.sub(r"\b(district|dist|dt)\b", " ", k)
    return re.sub(r"[^a-z0-9]", "", k)

def first_url(*values):
    for v in values:
        for it in (to_list(v) or [to_text(v)]):
            m = re.search(r"https?://[^\s\"\],]+", it or "")
            if m: return m.group(0)
            if re.fullmatch(r"[\w.-]+\.[a-z]{2,}(/\S*)?", it or ""): return "https://" + it
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def in_india(lat, lon):
    return (lat >= 6.0) & (lat <= 37.6) & (lon >= 68.0) & (lon <= 97.5)

def jitter(lat, lon, sigma_km, n, rng):
    dlat = rng.normal(0, sigma_km / 111.0, n)
    dlon = rng.normal(0, sigma_km / (111.0 * max(math.cos(math.radians(lat)), 0.2)), n)
    return lat + dlat, lon + dlon

def sv(x):
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. [A] Geography layer: cleaning the India Post PIN directory
# MAGIC Inspecting the table showed coordinates stored as **text**, some `"NA"`, some with **latitude and longitude swapped**
# MAGIC (e.g. `79.0, 17.0`), and some far outside India (e.g. `35.9, 61.3`). We clean it in four steps:
# MAGIC 1. Parse to numbers. If a point falls outside India but the *swapped* pair falls inside, swap it (counted as `swapped`).
# MAGIC 2. Drop points still outside India.
# MAGIC 3. Per PIN, take the **median** point, drop offices > 40 km from it (bad points), then recompute. The median ignores a few bad points.
# MAGIC 4. Per PIN, record: `radius_km` (80th-percentile spread, i.e. how big the area is), `bo_share` (share of *Branch Offices*,
# MAGIC    which are mostly rural) and `nondelivery_share`. Both serve as a **rural/urban proxy** for travel speed and road detours.
# MAGIC
# MAGIC The cleaned offices also let users type **village / post-office names** as locations.

# COMMAND ----------

po = spark.table(PIN_TABLE).toPandas()
po["pincode"] = po["pincode"].map(lambda v: re.sub(r"\D", "", str(v))[:6].zfill(6) if not is_blank(v) else "")
po["lat"] = pd.to_numeric(po["latitude"], errors="coerce")
po["lon"] = pd.to_numeric(po["longitude"], errors="coerce")
ok = in_india(po["lat"], po["lon"])
swap_ok = ~ok & in_india(po["lon"], po["lat"])
po.loc[swap_ok, ["lat", "lon"]] = po.loc[swap_ok, ["lon", "lat"]].values
po["coord_fix"] = np.where(ok, "ok", np.where(swap_ok, "swapped", "invalid"))
po.loc[po["coord_fix"] == "invalid", ["lat", "lon"]] = np.nan
po["district_n"] = po["district"].map(norm_district)
po["state_n"] = po["statename"].map(norm_state)
po["office_n"] = po["officename"].map(lambda s: norm_name(re.sub(r"\b(b\.?o|s\.?o|h\.?o|g\.?p\.?o)\b\.?", " ", to_text(s), flags=re.I)))
po["is_bo"] = po["officetype"].astype(str).str.upper().str.strip().eq("BO")
po["is_nondel"] = po["delivery"].astype(str).str.lower().str.contains("non")

po_clean = po.dropna(subset=["lat", "lon"]).copy()
for _ in range(2):   # vectorised: drop offices > 40 km from their PIN's median, twice
    mlat = po_clean.groupby("pincode")["lat"].transform("median"); mlon = po_clean.groupby("pincode")["lon"].transform("median")
    po_clean = po_clean[haversine_km(po_clean["lat"], po_clean["lon"], mlat, mlon) <= 40]
po_clean = po_clean.reset_index(drop=True)
def fast_mode(df, key, col):
    """Most common non-empty value of `col` per `key`, as one grouped count (fast) instead of a Python call per group (slow)."""
    d = df[df[col].astype(str).str.len() > 0]
    c = d.groupby([key, col]).size().reset_index(name="n").sort_values([key, "n"], ascending=[True, False])
    return c.drop_duplicates(key).set_index(key)[col]

pin_table = po.groupby("pincode").agg(n_offices=("officename", "size"), bo_share=("is_bo", "mean"),
                                      nondelivery_share=("is_nondel", "mean"))
for _out, _in in [("district_n", "district_n"), ("state_n", "state_n"), ("district", "district"), ("state", "statename")]:
    pin_table[_out] = fast_mode(po, "pincode", _in)
pin_table = pin_table.fillna({"district_n": "", "state_n": "", "district": "", "state": ""}).reset_index()
cent = po_clean.groupby("pincode").agg(lat=("lat", "median"), lon=("lon", "median")).reset_index()
pin_table = pin_table.merge(cent, on="pincode", how="left")
_spread = po_clean.merge(cent, on="pincode", suffixes=("", "_c"))
_spread["d"] = haversine_km(_spread["lat"], _spread["lon"], _spread["lat_c"], _spread["lon_c"])
pin_table = pin_table.merge(_spread.groupby("pincode")["d"].quantile(0.8).rename("radius_km").reset_index(), on="pincode", how="left")
pin_table["radius_km"] = pin_table["radius_km"].fillna(5).clip(1, 30)
PIN_IDX = pin_table.set_index("pincode")
pin_geo = pin_table.dropna(subset=["lat", "lon"]).reset_index(drop=True)

print(f"post offices: {len(po):,} | coordinates ok: {(po.coord_fix=='ok').mean():.1%} | swapped & fixed: {(po.coord_fix=='swapped').sum():,} "
      f"| invalid: {(po.coord_fix=='invalid').sum():,} | PINs: {len(pin_table):,} (with a usable centre: {len(pin_geo):,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Facilities: load, clean, place on the map, and check location validity
# MAGIC Location validity asks whether the facility's map pin is where its address says it is. We compare the stated coordinates
# MAGIC with the centre of the facility's **own PIN code** from India Post, scaled by that PIN's size:
# MAGIC
# MAGIC | distance from PIN centre | P(location correct) |
# MAGIC |---|---|
# MAGIC | ≤ 1× PIN radius | 0.95 |
# MAGIC | ≤ 3× radius | 0.75 |
# MAGIC | ≤ 10× radius | 0.35 |
# MAGIC | further | 0.10 |
# MAGIC
# MAGIC With no usable PIN, we fall back to the distance from the median of other facilities in the same city.
# MAGIC With no valid coordinates, the facility is placed at its PIN centre, or failing that its city centre, and P = 0 for the stated pin.
# MAGIC The high-criticality Monte-Carlo **samples** the facility's true position from this.

# COMMAND ----------

raw = spark.table(FACILITIES_TABLE).toPandas()
def col(name): return raw[name] if name in raw.columns else pd.Series([None] * len(raw))

fac = pd.DataFrame({
    "fid": col("unique_id").map(to_text),
    "name": col("name").map(to_text),
    "org_type": col("organization_type").map(lambda v: to_text(v).lower()),
    "ftype": col("facilityTypeId").map(lambda v: to_text(v).lower()),
    "operator": col("operatorTypeId").map(lambda v: to_text(v).lower()),
    "address": [", ".join(x for x in t if x) for t in zip(col("address_line1").map(to_text), col("address_line2").map(to_text), col("address_line3").map(to_text))],
    "city": col("address_city").map(to_text),
    "region": col("address_stateOrRegion").map(to_text),
    "postcode": col("address_zipOrPostcode").map(lambda v: re.sub(r"\D", "", to_text(v))[:6]),
    "lat": pd.to_numeric(col("latitude"), errors="coerce"),
    "lon": pd.to_numeric(col("longitude"), errors="coerce"),
    "specialties": col("specialties").map(lambda v: [uncamel(x) for x in to_list(v)]),
    "procedure": col("procedure").map(to_list),
    "equipment": col("equipment").map(to_list),
    "capability": col("capability").map(to_list),
    "description": col("description").map(to_text),
    "doctors": col("numberDoctors").map(to_num),
    "beds": col("capacity").map(to_num),
    "phone": col("officialPhone").map(to_text),
    "website": col("officialWebsite").map(lambda v: first_url(v)),
    "source_url": [first_url(a, b) for a, b in zip(col("source_urls"), col("officialWebsite"))],
    "cluster": col("cluster_id").map(to_text),
    "last_post": col("post_metrics_most_recent_social_media_post_date").map(to_text),
})
fac = fac[fac["org_type"] != "ngo"].reset_index(drop=True)          # NGOs are not treatment sites
fac["city_n"] = fac["city"].map(norm_name)
fac["name_n"] = fac["name"].map(norm_name)
fac["coords_ok"] = in_india(fac["lat"], fac["lon"]) & ~((fac["lat"] == 0) & (fac["lon"] == 0))
fac.loc[~fac["coords_ok"], ["lat", "lon"]] = np.nan

# PIN attributes of the facility's own postcode
pinfo = fac["postcode"].map(lambda p: PIN_IDX.loc[p] if p in PIN_IDX.index else None)
fac["pin_lat"] = [r["lat"] if r is not None else np.nan for r in pinfo]
fac["pin_lon"] = [r["lon"] if r is not None else np.nan for r in pinfo]
fac["pin_radius"] = [r["radius_km"] if r is not None else np.nan for r in pinfo]
fac["district_n"] = [r["district_n"] if r is not None else "" for r in pinfo]
fac["state_n"] = [r["state_n"] if r is not None else norm_state(s) for r, s in zip(pinfo, fac["region"])]
fac["district"] = [r["district"] if r is not None else "" for r in pinfo]

# City centres from facilities (fallback geography)
cc = fac[fac["coords_ok"] & (fac["city_n"] != "")].groupby("city_n").agg(c_lat=("lat", "median"), c_lon=("lon", "median"), c_n=("lat", "size"))
fac = fac.join(cc, on="city_n")

# Facilities with no usable PIN: assign district from the nearest PIN centre (chunked nearest-neighbour)
def nearest_pin(lat, lon, max_km=30):
    out_idx, out_d = np.full(len(lat), -1), np.full(len(lat), np.inf)
    P = pin_geo[["lat", "lon"]].to_numpy()
    for s in range(0, len(lat), 400):
        la, lo = lat[s:s+400, None], lon[s:s+400, None]
        d = haversine_km(la, lo, P[None, :, 0], P[None, :, 1])
        j = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
        out_idx[s:s+400] = j; out_d[s:s+400] = d[np.arange(len(j)), j]
    out_idx[out_d > max_km] = -1
    return out_idx
need = (fac["district_n"] == "") & fac["coords_ok"]
if need.any() and len(pin_geo):
    idx = nearest_pin(fac.loc[need, "lat"].to_numpy(), fac.loc[need, "lon"].to_numpy())
    fac.loc[need, "district_n"] = [pin_geo.loc[i, "district_n"] if i >= 0 else "" for i in idx]
    fac.loc[need, "state_n"] = [pin_geo.loc[i, "state_n"] if i >= 0 else s for i, s in zip(idx, fac.loc[need, "state_n"])]
    fac.loc[need, "district"] = [pin_geo.loc[i, "district"] if i >= 0 else "" for i in idx]

# Location validity
d_pin = haversine_km(fac["lat"], fac["lon"], fac["pin_lat"], fac["pin_lon"])
r = fac["pin_radius"].fillna(5)
p_loc = np.select([d_pin <= r, d_pin <= 3 * r, d_pin <= 10 * r, d_pin > 10 * r], [0.95, 0.75, 0.35, 0.10], default=np.nan)
d_city = haversine_km(fac["lat"], fac["lon"], fac["c_lat"], fac["c_lon"])
p_loc_city = np.where((fac["c_n"] >= 3) & (d_city > 60), 0.30, 0.85)
fac["p_loc"] = np.where(~fac["coords_ok"], 0.0, np.where(np.isnan(p_loc), p_loc_city, p_loc))
fac["km_from_pin"] = d_pin
# Best-guess position used for searching
fac["plat"] = fac["lat"].fillna(fac["pin_lat"]).fillna(fac["c_lat"])
fac["plon"] = fac["lon"].fillna(fac["pin_lon"]).fillna(fac["c_lon"])
fac["pos_source"] = np.where(fac["coords_ok"], "stated coordinates", np.where(fac["pin_lat"].notna(), "PIN-code centre", np.where(fac["c_lat"].notna(), "city centre", "unknown")))
fac["dup_key"] = np.where(fac["cluster"] != "", "c:" + fac["cluster"], "n:" + fac["name_n"] + "|" + fac["city_n"])
fac["dup_count"] = fac.groupby("dup_key")["fid"].transform("count")
fac["maybe_merged"] = fac["description"].str.contains(r"\b(?:branches|chain of|group of hospitals|network of hospitals|multiple locations|units across)\b", case=False, regex=True)
fac["all_text"] = fac.apply(lambda r_: " | ".join([" ; ".join(r_["procedure"]), " ; ".join(r_["equipment"]), " ; ".join(r_["capability"]), r_["description"], " ; ".join(r_["specialties"])]), axis=1)
fac = fac.reset_index(drop=True)
print(f"facilities: {len(fac):,} | valid coords: {fac.coords_ok.mean():.1%} | with district: {(fac.district_n!='').mean():.1%} "
      f"| location validity ≥0.75: {(fac.p_loc>=0.75).mean():.1%} | no position at all: {(fac.pos_source=='unknown').sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. NFHS-5: parsing, uncertainty flags, and the district bridge
# MAGIC About 40 NFHS columns are stored as **text** because NFHS marks small samples: `(45.2)` means based on 25–49 unweighted cases,
# MAGIC and `*` means suppressed (fewer than 25 cases). We keep the number **and a `small_sample` flag**, so explanations can say "(small sample)"
# MAGIC instead of presenting a shaky number as fact.
# MAGIC
# MAGIC **District bridge.** NFHS district names and India Post district names differ (renamed and new districts, spelling, case).
# MAGIC We match on (state, district): exact normalised match first, then a small alias list, then fuzzy matching within the same state. The match rate is printed.
# MAGIC Unmatched districts simply get no NFHS context, and the explanation says so.

# COMMAND ----------

nf_raw = spark.table(NFHS_TABLE).toPandas()
NF_SMALL = pd.DataFrame(index=nf_raw.index)
nf = pd.DataFrame({"district_name": nf_raw["district_name"], "state_ut": nf_raw["state_ut"]})
for c in nf_raw.columns:
    if c in ("district_name", "state_ut"): continue
    s = nf_raw[c]
    if s.dtype == object:
        txt = s.astype(str)
        NF_SMALL[c] = txt.str.contains(r"\(") | txt.str.contains(r"\*")
        nf[c] = pd.to_numeric(txt.str.replace(r"[()*,%\s]", "", regex=True), errors="coerce")
    else:
        NF_SMALL[c] = False
        nf[c] = pd.to_numeric(s, errors="coerce")
nf["key"] = nf["state_ut"].map(norm_state) + "|" + nf["district_name"].map(norm_district)
NF_SMALL["key"] = nf["key"]
nf = nf.drop_duplicates("key").set_index("key")
NF_SMALL = NF_SMALL.drop_duplicates("key").set_index("key")

DISTRICT_ALIASES = {"gurugram": "gurgaon", "gurgaon": "gurugram", "nuh": "mewat", "prayagraj": "allahabad", "ayodhya": "faizabad",
                    "hanumakonda": "warangalurban", "warangal": "warangalrural", "jangoan": "jangaon", "bengaluruurban": "bangalore",
                    "bengalururural": "bangalorerural", "mysuru": "mysore", "belagavi": "belgaum", "kalaburagi": "gulbarga",
                    "vijayapura": "bijapur", "shivamogga": "shimoga", "tumakuru": "tumkur", "chikkamagaluru": "chikmagalur"}
_nf_by_state = {}
for k in nf.index:
    st, ds = k.split("|", 1); _nf_by_state.setdefault(st, []).append(ds)

def nfhs_key(state_n, district_n):
    if not district_n: return None
    for d in [district_n, DISTRICT_ALIASES.get(district_n, "")]:
        if d and f"{state_n}|{d}" in nf.index: return f"{state_n}|{d}"
    cands = _nf_by_state.get(state_n, [])
    m = difflib.get_close_matches(district_n, cands, n=1, cutoff=0.82)
    if m: return f"{state_n}|{m[0]}"
    for st, ds in _nf_by_state.items():          # state-name mismatch (e.g. after reorganisation): unique exact district
        if district_n in ds: return f"{st}|{district_n}"
    return None

_pairs = pin_table[["state_n", "district_n"]].drop_duplicates()
_pairs["nf_key"] = [nfhs_key(s, d) for s, d in zip(_pairs["state_n"], _pairs["district_n"])]
BRIDGE = {(s, d): k for s, d, k in _pairs.itertuples(index=False)}
pin_table["nf_key"] = [BRIDGE.get((s, d)) for s, d in zip(pin_table["state_n"], pin_table["district_n"])]
PIN_IDX = pin_table.set_index("pincode"); pin_geo = pin_table.dropna(subset=["lat", "lon"]).reset_index(drop=True)
fac["nf_key"] = [BRIDGE.get((s, d)) or nfhs_key(s, d) for s, d in zip(fac["state_n"], fac["district_n"])]
print(f"NFHS districts: {len(nf)} | India-Post districts matched to NFHS: {_pairs['nf_key'].notna().mean():.1%} "
      f"| facilities with NFHS context: {fac['nf_key'].notna().mean():.1%}")

def nf_col(*must):
    """Find an NFHS column whose name contains all the given fragments (robust to long column names)."""
    for c in nf.columns:
        if all(m in c for m in must): return c
    return None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. [0] Triage: classifying a service into High / Medium / Low criticality
# MAGIC **Definition (yours):** criticality = how time-critical the need is.
# MAGIC
# MAGIC **Method.** No Indian dataset labels services by time-criticality, and asking an LLM to guess would be neither reproducible nor citable.
# MAGIC So we use the quantity clinicians themselves use: the **maximum safe delay before treatment (the time window)**, taken from published clinical
# MAGIC guidance. The tier follows from the window:
# MAGIC
# MAGIC | Tier | Time window | Typical services |
# MAGIC |---|---|---|
# MAGIC | **High** | ≤ 6 hours | trauma/accident, emergency C-section & unexpected childbirth, sick newborn (NICU), burns, heart attack, stroke, emergency dialysis, ICU |
# MAGIC
# MAGIC Every catalog entry records its source and how strong that source is (`basis_type`): **guideline** (a published clinical guideline target),
# MAGIC **evidence** (outcome studies), **convention** (widely used but weakly evidenced, e.g. the trauma "golden hour"), or **assumption** (our own
# MAGIC judgement: high-risk delivery, CT, cardiac surgery, MRI, and the 6 h emergency-dialysis limit). The source and its type are shown in every triage explanation.
# MAGIC | **Medium** | 6 hours – 7 days | maintenance dialysis, planned high-risk delivery, CT scan, (non-emergency) cardiac surgery |
# MAGIC | **Low** | > 7 days | cancer care / chemotherapy, MRI |
# MAGIC
# MAGIC **Context modifiers from the user's text.** Words like *accident, bleeding, unconscious, labour pain, chest pain, emergency* **upgrade** the
# MAGIC request. Where an emergency variant exists we switch to it (dialysis → emergency dialysis, cardiac → heart attack). Words like *scheduled,
# MAGIC follow-up, routine, planned* can **downgrade Medium → Low**. A High request is never downgraded by such words (the one exception: an explicitly
# MAGIC *scheduled* C-section, delivery or dialysis with no urgency words is mapped to its planned counterpart, and the output says so), because under-triage
# MAGIC (treating an emergency as routine) costs far more than over-triage. That asymmetry is deliberate and stated in the output.
# MAGIC
# MAGIC **Unknown services** are fuzzy-matched to the catalog. If that fails, an optional LLM estimates the time window (bounded and labelled as an estimate).
# MAGIC If that also fails, the request defaults to **Medium** with a warning.
# MAGIC
# MAGIC Each catalog entry also carries: evidence patterns (strong = direct claim, support = plausible), a **capability bundle** (what a facility needs
# MAGIC to do this well, which gives the quality score), `tau` (how fast the benefit of treatment decays with delay), `visits_per_month` (recurring care makes distance matter more),
# MAGIC the NFHS **need** indicators, and which official **registries** confirm it.

# COMMAND ----------

SERVICE_CATALOG = {
 # ---------------- HIGH ----------------
 "trauma care": dict(window_min=60, basis="‘Golden hour’ convention of trauma systems; a teaching principle, its exact 60-min cut-off is weakly supported by outcome data", basis_type="convention", source="https://pmc.ncbi.nlm.nih.gov/articles/PMC3008652", visits=0,
     aliases=["trauma", "accident", "road accident", "injury", "polytrauma", "fracture emergency", "head injury", "trauma care"],
     strong=[r"trauma (?:centre|center|care|unit|surgery|ward)", r"polytrauma", r"casualty", r"accident (?:and|&) emergency", r"emergency surgery"],
     support=[r"emergency", r"orthop(?:a)?edic", r"neurosurg", r"\bicu\b", r"ambulance"],
     bundle={"CT scan": r"\bct\b|computed tomography", "blood bank": r"blood bank|blood storage", "operation theatre": r"operation theat|\bot\b|surgery",
             "orthopaedics": r"orthop", "neurosurgery": r"neurosurg", "ICU": r"\bicu\b|intensive care", "24×7": r"24\s*[x×/]\s*7|24 hours|round the clock"},
     family="trauma", needs_hospital=True, registries=["PMJAY:emergency", "PMJAY:polytrauma", "PMJAY:orthopaedics", "TRAUMA"]),
 "emergency c-section": dict(window_min=60, basis="Category-1 caesarean: decision-to-delivery target ≤30 min (NICE NG192); we allow ~1 h including transfer", basis_type="guideline", source="https://www.nice.org.uk/guidance/ng192", visits=0,
     aliases=["c-section", "c section", "caesarean", "cesarean", "emergency c-section", "lscs", "emergency caesarean"],
     strong=[r"c[\s-]?section", r"ca?esarean", r"\blscs\b", r"\bcemonc\b", r"comprehensive emergency obstetric", r"emergency obstetric"],
     support=[r"obstetric", r"gyn(?:a)?ecolog", r"maternity", r"labou?r room", r"anaesthe|anesthe", r"blood bank"],
     bundle={"operation theatre": r"operation theat|\bot\b|maternity ot", "anaesthesia": r"anaesthe|anesthe", "blood bank": r"blood bank|blood storage",
             "newborn care/NICU": r"nicu|sncu|neonat", "obstetrician": r"obstetric|gyn", "24×7": r"24\s*[x×/]\s*7|24 hours|round the clock"},
     family="obstetric", needs_hospital=True, registries=["LAQSHYA_MOT", "FRU", "PMJAY:obstetrics"]),
 "emergency childbirth": dict(window_min=120, basis="Postpartum haemorrhage, the leading cause of maternal death (WHO), can kill within ~2 h of onset if untreated", basis_type="evidence", source="https://www.who.int/publications/i/item/9789240081802", visits=0,
     aliases=["unexpected childbirth", "labour", "labor", "in labour", "labour pain", "delivery now", "childbirth", "giving birth", "pregnancy emergency", "bleeding in pregnancy"],
     strong=[r"labou?r room", r"emergency obstetric", r"\b(?:b|c)emonc\b", r"deliver(?:y|ies)", r"maternity (?:ward|unit|care)", r"c[\s-]?section|ca?esarean"],
     support=[r"obstetric", r"gyn(?:a)?ecolog", r"maternity", r"midwi", r"neonat"],
     bundle={"labour room": r"labou?r room", "operation theatre": r"operation theat|\bot\b", "blood bank": r"blood bank", "newborn care": r"nicu|sncu|neonat|newborn",
             "24×7": r"24\s*[x×/]\s*7|24 hours"},
     family="obstetric", needs_hospital=False, registries=["LAQSHYA_LR", "LAQSHYA_MOT", "FRU", "PMJAY:obstetrics"]),
 "nicu": dict(window_min=60, basis="Neonatal ‘Golden Hour’: stabilise high-risk newborns within the first 60 min of life (strongest evidence for preterm babies)", basis_type="evidence", source="https://pmc.ncbi.nlm.nih.gov/articles/PMC5604187", visits=0,
     aliases=["nicu", "neonatal icu", "newborn icu", "sncu", "sick newborn", "premature baby", "neonatal intensive care"],
     strong=[r"\bnicu\b", r"neonatal intensive", r"\bsncu\b", r"special newborn care", r"neonatal ventilat", r"level\s*(?:ii|iii|2|3) nicu"],
     support=[r"neonat", r"paediatric|pediatric", r"incubator", r"phototherapy", r"radiant warmer"],
     bundle={"ventilator/CPAP": r"ventilat|cpap", "incubator/warmer": r"incubator|radiant warmer", "phototherapy": r"phototherapy", "paediatrician": r"paediatric|pediatric|neonatolog",
             "24×7": r"24\s*[x×/]\s*7|24 hours"},
     family="neonatal", needs_hospital=True, registries=["SNCU", "PMJAY:neonatal"]),
 "burn care": dict(window_min=120, basis="Mortality rises if burn fluid resuscitation is delayed beyond 2 h after injury", basis_type="evidence", source="https://pmc.ncbi.nlm.nih.gov/articles/PMC4057496", visits=0,
     aliases=["burn", "burns", "burn care", "burn unit", "scald", "fire injury"],
     strong=[r"burns? (?:unit|ward|care|centre|center|icu)", r"burn management", r"skin graft"],
     support=[r"plastic surg", r"reconstructive", r"\bicu\b", r"emergency"],
     bundle={"burn unit": r"burns? (?:unit|ward|icu)", "ICU": r"\bicu\b|intensive care", "plastic surgery": r"plastic surg", "blood bank": r"blood bank", "24×7": r"24\s*[x×/]\s*7|24 hours"},
     family="burn", needs_hospital=True, registries=["BURN_UNIT", "PMJAY:burns"]),
 "heart attack": dict(window_min=90, basis="STEMI: first-medical-contact-to-device ≤90 min, or ≤120 min if transferred (ACC/AHA)", basis_type="guideline", source="https://professional.heart.org/en/science-news/2013-guideline-for-the-management-of-st-elevation-myocardial-infarction/commentary", visits=0,
     aliases=["heart attack", "chest pain", "stemi", "cardiac emergency", "angioplasty", "mi", "myocardial infarction"],
     strong=[r"cath(?:eteri[sz]ation)? lab", r"angioplasty", r"\bpci\b|ptca", r"primary angioplasty", r"\bccu\b|coronary care", r"thrombolysis"],
     support=[r"cardiolog", r"\becg\b", r"\bicu\b", r"echo"],
     bundle={"cath lab": r"cath(?:eteri[sz]ation)? lab", "CCU/ICU": r"\bccu\b|coronary care|\bicu\b", "cardiologist": r"cardiolog", "24×7": r"24\s*[x×/]\s*7|24 hours"},
     family="cardiac", needs_hospital=True, registries=["PMJAY:cardiology"]),
 "stroke": dict(window_min=270, basis="IV thrombolysis within 4.5 h of symptom onset (AHA/ASA acute ischaemic stroke guideline)", basis_type="guideline", source="https://stroke.org/-/media/Stroke-Files/Ischemic-Stroke-Professional-Materials/AIS-Toolkit/Guidelines-for-Mangaging-Patients-with-AIS-2019-Update-to-2018-Guidelines.pdf", visits=0,
     aliases=["stroke", "paralysis", "brain attack", "facial droop"],
     strong=[r"stroke (?:unit|care|centre|center)", r"thrombolysis", r"thrombectomy"],
     support=[r"neurolog", r"\bct\b", r"\bmri\b", r"\bicu\b"],
     bundle={"CT scan": r"\bct\b|computed tomography", "neurologist": r"neurolog", "ICU": r"\bicu\b", "24×7": r"24\s*[x×/]\s*7|24 hours"},
     family="cardiac", needs_hospital=True, registries=["PMJAY:neurology"]),
 "emergency dialysis": dict(window_min=360, basis="KDIGO AKI 5.1.1: start dialysis ‘emergently’ for life-threatening fluid/electrolyte/acid-base changes; KDIGO gives no hour limit, the 6 h figure is our assumption", basis_type="guideline + assumption", source="https://pmc.ncbi.nlm.nih.gov/articles/PMC4089702", visits=0,
     aliases=["emergency dialysis", "acute dialysis", "urgent dialysis", "kidney failure emergency"],
     strong=[r"h(?:a)?emodialysis", r"dialysis", r"\bcrrt\b", r"\bsled\b"],
     support=[r"nephrolog", r"kidney", r"renal", r"\bicu\b"],
     bundle={"dialysis unit": r"dialysis", "ICU": r"\bicu\b", "nephrologist": r"nephrolog", "24×7": r"24\s*[x×/]\s*7|24 hours"},
     family="renal", needs_hospital=True, registries=["PMNDP", "PMJAY:nephrology"]),
 "icu": dict(window_min=240, basis="Delays to ICU admission are linked to higher mortality: ≥6 h (Chalfin 2007, Crit Care Med) and >4 h for ventilated patients", basis_type="evidence", source="https://pmc.ncbi.nlm.nih.gov/articles/PMC4175615", visits=0,
     aliases=["icu", "intensive care", "critical care", "ventilator", "on ventilator"],
     strong=[r"\bicu\b", r"intensive care", r"critical care", r"ventilator"],
     support=[r"emergency", r"anaesthe|anesthe", r"monitor"],
     bundle={"ventilator": r"ventilat", "ICU beds": r"\bicu\b|intensive care", "24×7": r"24\s*[x×/]\s*7|24 hours", "blood bank": r"blood bank"},
     family="critical", needs_hospital=True, registries=["PMJAY:critical care"]),
 # ---------------- MEDIUM ----------------
 "dialysis": dict(window_min=2 * 1440, basis="Thrice-weekly haemodialysis: deaths are higher on the day after the 2-day gap (Foley et al., NEJM 2011), so a missed session is dangerous within days", basis_type="evidence", source="https://pubmed.ncbi.nlm.nih.gov/21992122/", visits=12,
     aliases=["dialysis", "kidney dialysis", "haemodialysis", "hemodialysis", "renal dialysis", "maintenance dialysis"],
     strong=[r"h(?:a)?emodialysis", r"dialysis", r"\bcrrt\b", r"peritoneal dialysis"],
     support=[r"nephrolog", r"kidney", r"renal"],
     bundle={"dialysis machines": r"dialysis machine|dialysis unit|dialysis", "nephrologist": r"nephrolog", "ICU backup": r"\bicu\b"},
     family="renal", needs_hospital=False, registries=["PMNDP", "PMJAY:nephrology"]),
 "high-risk delivery": dict(window_min=7 * 1440, basis="Planned referral of a high-risk pregnancy before term; the 7-day window is our assumption", basis_type="assumption", source="", visits=2,
     aliases=["high risk delivery", "high-risk pregnancy", "high risk pregnancy", "planned delivery", "maternity"],
     strong=[r"high[\s-]?risk (?:pregnancy|delivery|obstetric)", r"maternal[\s-]f(?:o)?etal", r"f(?:o)?etal medicine", r"\bnicu\b", r"c[\s-]?section", r"ca?esarean", r"\bcemonc\b"],
     support=[r"obstetric", r"gyn(?:a)?ecolog", r"maternity", r"labou?r room", r"neonat", r"blood bank"],
     bundle={"operation theatre": r"operation theat|\bot\b", "NICU": r"nicu|sncu", "blood bank": r"blood bank", "obstetrician": r"obstetric|gyn", "anaesthesia": r"anaesthe|anesthe"},
     family="obstetric", needs_hospital=True, registries=["LAQSHYA_MOT", "LAQSHYA_LR", "FRU", "PMJAY:obstetrics"]),
 "ct scan": dict(window_min=1440, basis="Urgent (non-emergency) imaging, same or next day; our assumption (emergency CT is covered by trauma/stroke)", basis_type="assumption", source="", visits=0,
     aliases=["ct", "ct scan", "cat scan", "computed tomography"],
     strong=[r"\bct\b(?! ?valve)", r"ct scan", r"computed tomography", r"\bmdct\b", r"\d+[\s-]slice"],
     support=[r"radiolog", r"imaging", r"diagnostic"],
     bundle={"CT": r"\bct\b|computed tomography", "radiologist": r"radiolog"},
     family="imaging", needs_hospital=False, registries=["PMJAY:radiology"]),
 "cardiac surgery": dict(window_min=5 * 1440, basis="Urgent (non-emergency) cardiac surgery scheduled within days; our assumption (heart attack is a separate High service)", basis_type="assumption", source="", visits=1,
     aliases=["cardiac surgery", "heart surgery", "bypass", "cabg", "open heart", "valve replacement"],
     strong=[r"cardiac surg", r"cardiothoracic", r"\bcabg\b", r"bypass surgery", r"open[\s-]heart", r"valve replacement", r"\bctvs\b"],
     support=[r"cardiolog", r"cath(?:eteri[sz]ation)? lab", r"angioplasty", r"heart[\s-]lung machine", r"\bccu\b"],
     bundle={"cardiac OT": r"cardiac (?:ot|theat)|cardiothoracic", "heart-lung machine": r"heart[\s-]lung|perfusion", "CCU/ICU": r"\bccu\b|\bicu\b", "cath lab": r"cath"},
     family="cardiac", needs_hospital=True, registries=["PMJAY:cardio-thoracic", "PMJAY:cardiothoracic"]),
 # ---------------- LOW ----------------
 "cancer care": dict(window_min=30 * 1440, basis="Each 4-week delay in cancer treatment raises mortality by ~6–13% (Hanna et al., BMJ 2020): urgent in weeks, not hours", basis_type="evidence", source="https://www.bmj.com/content/371/bmj.m4087", visits=3,
     aliases=["cancer", "oncology", "chemotherapy", "chemo", "radiotherapy", "tumour", "tumor", "malignancy"],
     strong=[r"oncolog", r"chemotherap", r"radiotherap|radiation therap", r"linear accelerator|linac", r"cancer (?:care|centre|center|hospital|surgery)"],
     support=[r"cancer", r"tumou?r", r"biopsy", r"\bpet\b"],
     bundle={"chemotherapy": r"chemotherap", "radiotherapy": r"radiotherap|linac|linear accelerator", "surgical oncology": r"surgical oncolog|onco[\s-]?surg", "pathology": r"patholog|biopsy"},
     family="oncology", needs_hospital=True, registries=["PMJAY:medical oncology", "PMJAY:radiation oncology", "PMJAY:surgical oncology"]),
 "mri": dict(window_min=14 * 1440, basis="Elective diagnostic MRI, days to weeks; our assumption", basis_type="assumption", source="", visits=0,
     aliases=["mri", "mri scan", "magnetic resonance"],
     strong=[r"\bmri\b", r"magnetic resonance", r"\b1\.5\s*t(?:esla)?\b", r"\b3\s*t(?:esla)?\b"],
     support=[r"radiolog", r"imaging", r"diagnostic"],
     bundle={"MRI": r"\bmri\b|magnetic resonance", "radiologist": r"radiolog"},
     family="imaging", needs_hospital=False, registries=["PMJAY:radiology"]),
}
PLANNED_VARIANT = {"emergency c-section": "high-risk delivery", "emergency childbirth": "high-risk delivery", "emergency dialysis": "dialysis"}
EMERGENCY_VARIANT = {"dialysis": "emergency dialysis", "cardiac surgery": "heart attack", "high-risk delivery": "emergency childbirth", "ct scan": "trauma care"}

def tier_of(window_min):
    return "High" if window_min <= 360 else ("Medium" if window_min <= 7 * 1440 else "Low")
for k, s in SERVICE_CATALOG.items():
    s["key"] = k; s["tier"] = tier_of(s["window_min"])
    s["tau"] = {"High": s["window_min"] * 0.75, "Medium": 180.0, "Low": 480.0}[s["tier"]]   # minutes; outcome-decay constant

ONCO_WORDS = r"\b\w*(?:sarcoma|carcinoma|lymphoma|leuka?emia|myeloma|melanoma|blastoma|glioma|mesothelioma|seminoma|neoplasm|metasta\w*)\b"
HIGH_WORDS = r"\b(emergency|accident|crash|bleeding|haemorrhage|hemorrhage|unconscious|collapsed|not breathing|can.?t breathe|labou?r pain|in labou?r|water broke|chest pain|heart attack|stroke|seizure|fitting|burnt|burned|severe|critical|urgent|right now|immediately)\b"
LOW_WORDS = r"\b(scheduled|follow[- ]?up|routine|check[- ]?up|second opinion|planned|elective|next month)\b"

def triage(service_text):
    t = service_text.strip().lower()
    reasons, key, how = [], None, ""
    # exact alias first (longest alias wins), then fuzzy
    best = None
    for k, s in SERVICE_CATALOG.items():
        for a in s["aliases"] + [k]:
            if re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", t) and (best is None or len(a) > best[1]): best = (k, len(a))
    if best: key, how = best[0], "matched catalog wording"
    elif re.search(ONCO_WORDS, t):
        key, how = "cancer care", f"'{re.search(ONCO_WORDS, t).group(0)}' is a cancer diagnosis (medical suffix rule)"
    else:
        all_alias = {a: k for k, s in SERVICE_CATALOG.items() for a in s["aliases"] + [k]}
        m = difflib.get_close_matches(t, list(all_alias), n=1, cutoff=0.75)
        if m: key, how = all_alias[m[0]], f"fuzzy-matched '{t}' to '{m[0]}'"
    if key is None:
        est = llm_estimate_window(t) if USE_LLM else None
        if est:
            return dict(key=None, label=t, tier=tier_of(est["window_min"]), window_min=est["window_min"],
                        reasons=[f"'{t}' is not in the service catalog; an LLM estimated a time window of {est['window_min']/60:.0f} h ({est['why']}). Treat this tier as an estimate."],
                        unknown=True)
        return dict(key=None, label=t, tier="Medium", window_min=1440, unknown=True,
                    reasons=[f"'{t}' is not in the service catalog. Defaulted to Medium criticality. If this is an emergency, call 108/112."])
    spec = SERVICE_CATALOG[key]
    reasons.append(f"Service recognised as **{key}** ({how}).")
    reasons.append(f"Clinical time window ≈ {fmt_window(spec['window_min'])}: {spec['basis']} [{spec['basis_type']}"
                   + (f"; source: {spec['source']}" if spec['source'] else "") + "]. Window ≤ 6 h → High; ≤ 7 days → Medium; otherwise Low.")
    tier = spec["tier"]
    if re.search(HIGH_WORDS, t) and tier != "High":
        if key in EMERGENCY_VARIANT:
            new = EMERGENCY_VARIANT[key]
            reasons.append(f"Urgency words in the request ('{re.search(HIGH_WORDS, t).group(0)}') → switched to the emergency variant **{new}**.")
            key, spec, tier = new, SERVICE_CATALOG[new], SERVICE_CATALOG[new]["tier"]
        else:
            reasons.append(f"Urgency words in the request ('{re.search(HIGH_WORDS, t).group(0)}') → upgraded to High.")
            tier = "High"
    elif re.search(LOW_WORDS, t) and tier == "Medium":
        reasons.append(f"Planning words in the request ('{re.search(LOW_WORDS, t).group(0)}') → downgraded Medium → Low.")
        tier = "Low"
    elif re.search(LOW_WORDS, t) and tier == "High" and key in PLANNED_VARIANT and not re.search(HIGH_WORDS, t):
        new = PLANNED_VARIANT[key]
        reasons.append(f"The request is explicitly planned ('{re.search(LOW_WORDS, t).group(0)}') with no urgency words → treated as the planned service "
                       f"**{new}**. If this is happening now, search for it as an emergency instead.")
        key, spec, tier = new, SERVICE_CATALOG[new], SERVICE_CATALOG[new]["tier"]
    elif re.search(LOW_WORDS, t) and tier == "High":
        reasons.append("Planning words were found, but a High request is never downgraded: under-triage is costlier than over-triage.")
    return dict(key=key, label=key, tier=tier, window_min=spec["window_min"], reasons=reasons, unknown=False)

def fmt_window(m):
    return f"{m:.0f} min" if m < 120 else (f"{m/60:.1f} h" if m < 2880 else f"{m/1440:.0f} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. [5] Registry triangulation (credibility layer, used by every tier)
# MAGIC Official government lists are an **independent** source. They aren't scraped from the facility's own website, so a match is strong evidence.
# MAGIC We match facilities to registry rows with **record linkage**:
# MAGIC * **Blocking:** only compare within the same PIN, or the same (state, district), to keep it fast and avoid false matches.
# MAGIC * **Similarity:** the maximum of (a) token-set Jaccard on cleaned names, with generic words like "hospital"/"govt" removed, and (b) `difflib` ratio.
# MAGIC   Plus a rule: two names that are both "district/civil/general hospital" in the same district match.
# MAGIC * **Threshold** 0.72. Every match records the matched name and score, so it can be shown and audited.
# MAGIC
# MAGIC Expected upload schema (one row per facility per registry), `workspace.default.registry_facilities`:
# MAGIC `registry, facility_name, state, district, pincode, services, valid_to, source_url, retrieved_on`.
# MAGIC `services` holds catalog keys separated by `;` (e.g. `emergency c-section;high-risk delivery`) **or** registry tags like `PMJAY:nephrology`.
# MAGIC **Caveat shown to users:** *absence* from a registry is **not** evidence against a facility, especially private ones. Only presence counts.

# COMMAND ----------

GENERIC = {"hospital", "hospitals", "govt", "government", "the", "and", "of", "pvt", "ltd", "private", "limited", "centre", "center",
           "medical", "health", "care", "clinic", "nursing", "home", "multispeciality", "multi", "speciality", "specialty", "super", "dr", "sri", "shri"}
ABBREV = {"dh": "district hospital", "sdh": "sub divisional hospital", "chc": "community health centre", "phc": "primary health centre",
          "mc": "medical college", "gh": "general hospital", "ch": "civil hospital", "rh": "rural hospital", "dwh": "district women hospital",
          "mch": "mother child hospital", "govt": "government"}
def expand(s): return " ".join(ABBREV.get(t, t) for t in norm_name(s).split())
def name_tokens(s): return {t for t in expand(s).split() if t not in GENERIC and len(t) > 1}
PUBLIC_GENERIC = re.compile(r"\b(district|civil|general|sadar|headquarters?|dhh|area|sub[- ]?divisional|community health|chc|phc|medical college)\b")

PLACE_WORDS = {"district", "sub", "divisional", "community", "primary", "general", "civil", "rural", "women", "mother", "child", "sadar", "area", "taluk", "college"}
def name_sim(a, b, place_tokens=frozenset()):
    """Similarity of two facility names using only their *core* words (generic words like 'hospital'/'district' and the
    district/town name removed), so 'Civil Hospital Nagpur' does not match 'DH Nagpur' just because they share 'hospital nagpur'."""
    ea, eb = expand(a), expand(b)
    ca = name_tokens(a) - place_tokens - PLACE_WORDS
    cb = name_tokens(b) - place_tokens - PLACE_WORDS
    sim = 0.0
    if ca and cb:
        sim = max(len(ca & cb) / len(ca | cb), difflib.SequenceMatcher(None, " ".join(sorted(ca)), " ".join(sorted(cb))).ratio())
        small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        if small <= big and any(len(t) >= 4 for t in small): sim = max(sim, 0.8)      # 'DH Daga' ⊂ 'Daga Memorial Women Hospital'
    if PUBLIC_GENERIC.search(ea) and PUBLIC_GENERIC.search(eb) and ("district" in ea) == ("district" in eb) and ("sub divisional" in ea) == ("sub divisional" in eb):
        if not ca or not cb or (ca & cb): sim = max(sim, 0.75)                          # 'District Hospital Nagpur' ~ 'DH Nagpur'
    return sim

REG = None
if table_exists(REGISTRY_TABLE):
    REG = spark.table(REGISTRY_TABLE).toPandas()
    REG.columns = [c.strip().lower() for c in REG.columns]
    for c in ["registry", "facility_name", "state", "district", "pincode", "services", "valid_to", "status", "source_url", "source_ref", "retrieved_on"]:
        if c not in REG.columns: REG[c] = ""
    REG = REG.fillna("")
    REG = REG[~REG["facility_name"].astype(str).str.upper().str.startswith("EXAMPLE")].reset_index(drop=True)   # template example rows
    REG["state_n"] = REG["state"].map(norm_state); REG["district_n"] = REG["district"].map(norm_district)
    REG["pincode"] = REG["pincode"].map(lambda v: re.sub(r"\D", "", to_text(v))[:6])
    REG["svc"] = REG["services"].map(lambda s: [x.strip().lower() for x in to_text(s).split(";") if x.strip()])
    # Official PDFs often truncate district names ("Ramanathapur"); map each to the India Post district of the same state
    _pd = pin_table.groupby("state_n")["district_n"].apply(lambda x: sorted(set(x) - {""})).to_dict()
    def _canon(st, d):
        ds = _pd.get(st, [])
        if not d or d in ds: return d
        if DISTRICT_ALIASES.get(d) in ds: return DISTRICT_ALIASES[d]
        pre = [x for x in ds if len(d) >= 6 and (x.startswith(d) or d.startswith(x))]
        if len(pre) == 1: return pre[0]
        m = difflib.get_close_matches(d, ds, n=1, cutoff=0.8)
        return m[0] if m else d
    REG["district_n"] = [_canon(st, d) for st, d in zip(REG["state_n"], REG["district_n"])]
    REG["district_alias"] = REG["district_n"].map(lambda d: DISTRICT_ALIASES.get(d, ""))
    _today = pd.Timestamp.today().strftime("%Y-%m-%d")
    REG["current"] = [(not to_text(v)) or to_text(v) >= _today for v in REG["valid_to"]]
    REG["strong"] = REG["current"] & (REG["status"].astype(str).str.lower() != "conditional")
    print(f"registry rows: {len(REG):,} from {REG['registry'].nunique()} registries: {REG['registry'].value_counts().to_dict()}")
else:
    print(f"Registry table {REGISTRY_TABLE} not found → credibility layer OFF (the notebook still runs; explanations will say so).")

REG_MATCH_CACHE = {}
def registry_matches(row):
    """Registry rows matching this facility (cached)."""
    if REG is None: return []
    if row["fid"] in REG_MATCH_CACHE: return REG_MATCH_CACHE[row["fid"]]
    block = REG[(REG["pincode"] == row["postcode"]) & (REG["pincode"] != "")]
    if row["district_n"]:
        block = pd.concat([block, REG[(REG["district_n"] == row["district_n"]) | (REG["district_alias"] == row["district_n"])]])
    out = []
    for _, g in block[~block.index.duplicated()].iterrows():
        s = name_sim(row["name"], g["facility_name"], frozenset(norm_name(row["district"]).split()) | frozenset(norm_name(row["city"]).split()))
        if s >= 0.72:
            out.append(dict(registry=g["registry"], name=g["facility_name"], score=round(s, 2), services=g["svc"], strong=bool(g["strong"]),
                            current=bool(g["current"]), status=to_text(g["status"]), valid_to=to_text(g["valid_to"]),
                            source_url=to_text(g["source_url"]), source_ref=to_text(g["source_ref"]), retrieved_on=to_text(g["retrieved_on"])))
    REG_MATCH_CACHE[row["fid"]] = out
    return out

def registry_confirms(match, spec):
    tags = set(match["services"]) | {match["registry"].lower()}
    wanted = {spec["key"]} | {r.lower() for r in spec["registries"]}
    return bool(tags & wanted) or any(t.split(":")[0] == w.split(":")[0] and ":" in t and t.split(":")[1] in w for t in tags for w in wanted)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. [B] + [1] Evidence extraction and the **Claim Court** (used by every tier)
# MAGIC The hypothesis on trial is *"facility f offers service s"*. We start from a **prior** probability and let each piece of evidence (each "witness")
# MAGIC multiply the odds by a **likelihood ratio (LR)**. In log-odds this is just addition, so each step is visible:
# MAGIC
# MAGIC $$\log\frac{p}{1-p} = \log\frac{p_0}{1-p_0} + \sum_i \log LR_i$$
# MAGIC
# MAGIC | Witness | LR | Why that size |
# MAGIC |---|---|---|
# MAGIC | prior: hospital / clinic / unknown | p₀ = 0.25 / 0.05 / 0.12 (clinic claiming hospital-level care: 0.02) | facility type is the base rate |
# MAGIC | strong claim in **procedure** | ×6 | a concrete claim about work performed |
# MAGIC | strong claim in **equipment** | ×5 | equipment is hard to claim casually |
# MAGIC | strong claim in **capability** | ×5 | |
# MAGIC | strong claim in **description** | ×2.5 | marketing text, easily inflated |
# MAGIC | strong claim in **specialties** | ×1.8 | a broad category |
# MAGIC | supporting evidence only (best one) | ×1.5 | plausible, not proof |
# MAGIC | corroboration across ≥2 independent fields | ×1.5 | agreement is the best signal without labels |
# MAGIC | ≥ half of the capability bundle present | ×1.5 | can actually deliver it |
# MAGIC | **official registry confirms this service** (current, unconditional) | ×10 | independent, government-checked |
# MAGIC | official registry confirms, but expired or conditional | ×4 | the unit existed and was assessed; status may have lapsed |
# MAGIC | in a registry for something else | ×1.3 | at least the facility exists |
# MAGIC | district context (NFHS) for surgical obstetrics | ×0.8–1.25 | private C-section rate in the district |
# MAGIC | negation ("no dialysis", "referred to") | ×0.25 | |
# MAGIC | clinic claiming hospital-level care | ×0.4 | |
# MAGIC | possible merged/group record | ×0.7 | the service may be at another branch |
# MAGIC
# MAGIC These are **expert-set** LRs, and that is stated in the output. If a registry is uploaded, section 12 measures how well they agree with it.
# MAGIC The output is P(offers service), plus an uncertainty band from a Beta distribution whose strength grows with the number of witnesses.

# COMMAND ----------

FIELD_LR = {"procedure": 6.0, "equipment": 5.0, "capability": 5.0, "description": 2.5, "specialties": 1.8}
STRUCTURED = {"procedure", "equipment", "capability"}
NEG_PAT = re.compile(r"(\bno\b|\bnot\b|\bwithout\b|\bunavailable\b|\brefer(?:red|s)? (?:to|out)\b|\bcoming soon\b|\bplanned\b|\bunder construction\b)[^.;]{0,25}$", re.I)

def _snip(text, m, w=90):
    a, b = max(0, m.start() - w), min(len(text), m.end() + w)
    return ("…" if a else "") + text[a:b].replace("\n", " ") + ("…" if b < len(text) else "")

def find_hits(value, patterns):
    items = value if isinstance(value, list) else [value]
    hits, neg = [], []
    for it in items:
        if not it: continue
        for p in patterns:
            m = re.search(p, it, re.I)
            if m:
                q = it if len(it) <= 220 else _snip(it, m)
                (neg if NEG_PAT.search(it[max(0, m.start() - 40): m.start()]) else hits).append((m.group(0), q))
    seen, out = set(), []
    for t, q in hits:
        if q not in seen: seen.add(q); out.append((t, q))
    return out, neg

def bundle_check(row, spec):
    found, missing = {}, []
    for label, pat in spec["bundle"].items():
        m = re.search(pat, row["all_text"], re.I)
        if m: found[label] = m.group(0)
        else: missing.append(label)
    frac = len(found) / max(1, len(spec["bundle"]))
    return found, missing, 0.5 + 0.5 * frac     # quality q in [0.5, 1]

def nfhs_value(key, *frag):
    c = nf_col(*frag)
    if key is None or c is None or key not in nf.index: return None, False
    v = nf.loc[key, c]; small = bool(NF_SMALL.loc[key, c]) if c in NF_SMALL.columns else False
    return (None if pd.isna(v) else float(v)), small

def claim_court(row, spec, use_registry=True):
    ftype = row["ftype"] or "unknown"
    p0 = {"hospital": 0.25, "clinic": 0.05}.get(ftype, 0.12)
    if spec["needs_hospital"] and ftype in ("clinic", "pharmacy", "doctor", "dentist"): p0 = 0.02
    lo = math.log(p0 / (1 - p0))
    steps = [dict(label=f"Prior for a {ftype} (base rate)", lr=None, p=p0, quote=None, field=None)]
    evidence, strong_fields, negs = [], set(), []
    def add(label, lr, quote=None, field=None):
        nonlocal lo
        lo += math.log(lr); p = 1 / (1 + math.exp(-lo))
        steps.append(dict(label=label, lr=lr, p=p, quote=quote, field=field))
    best_support = None
    for field, lr in FIELD_LR.items():
        h, n = find_hits(row[field], spec["strong"])
        negs += [(field, q) for _, q in n]
        if h:
            strong_fields.add(field)
            add(f"Strong claim in {field}: “{h[0][0]}”", lr, h[0][1], field)
            evidence += [dict(field=field, kind="direct", quote=q) for _, q in h[:2]]
        else:
            hs, _ = find_hits(row[field], spec["support"])
            if hs and best_support is None:
                best_support = (field, hs[0])
    if best_support and not strong_fields:
        add(f"Supporting evidence only, in {best_support[0]}: “{best_support[1][0]}”", 1.5, best_support[1][1], best_support[0])
        evidence.append(dict(field=best_support[0], kind="indirect", quote=best_support[1][1]))
    if len(strong_fields) >= 2: add(f"Corroborated by {len(strong_fields)} independent fields", 1.5)
    found, missing, q = bundle_check(row, spec)
    if q >= 0.75 and strong_fields: add(f"Capability bundle mostly present ({len(found)}/{len(spec['bundle'])})", 1.5)
    for field, qt in negs[:1]: add(f"Negation in {field}", 0.25, qt, field)
    if spec["needs_hospital"] and ftype in ("clinic", "pharmacy", "doctor", "dentist"): add(f"Listed as a {ftype}; this service needs a hospital", 0.4)
    if row["maybe_merged"]: add("Record may describe a hospital group, not this site", 0.7)
    regs = registry_matches(row) if use_registry else []
    conf = [m for m in regs if registry_confirms(m, spec)]
    if conf:
        conf = sorted(conf, key=lambda m: (not m.get("strong", True), -m["score"]))
        m0 = conf[0]
        if m0.get("strong", True):
            add(f"Official registry confirms: {m0['registry']} (“{m0['name']}”, match {m0['score']}, valid)", 10.0, None, "registry")
        else:
            why_weak = "certification expired " + m0["valid_to"] if not m0.get("current", True) else "conditional certification"
            add(f"Official registry confirms (weaker: {why_weak}): {m0['registry']} (“{m0['name']}”, match {m0['score']})", 4.0, None, "registry")
    elif regs: add(f"Listed in an official registry ({regs[0]['registry']}) for other services: it exists", 1.3, None, "registry")
    if spec["family"] == "obstetric" and spec["needs_hospital"] and row["nf_key"]:
        v, small = nfhs_value(row["nf_key"], "private_fac", "csection")
        if v is not None and row["operator"] == "private":
            lr = float(np.clip(1 + (v - 45) / 100, 0.8, 1.25))
            add(f"District context: {v:.0f}% of private-facility births are C-sections (NFHS-5{', small sample' if small else ''})", lr)
    p = 1 / (1 + math.exp(-lo))
    n_wit = len(steps) - 1
    n_eff = 2 + 3 * n_wit
    a, b = p * n_eff, (1 - p) * n_eff
    lo_ci, hi_ci = beta_interval(a, b)
    return dict(p=p, p_lo=lo_ci, p_hi=hi_ci, steps=steps, evidence=evidence, strong_fields=sorted(strong_fields),
                has_strong=bool(strong_fields), negations=negs, bundle_found=found, bundle_missing=missing, q=q,
                registry=regs, registry_confirms=conf, n_witnesses=n_wit)

def beta_interval(a, b, n=4000):
    s = RNG.beta(max(a, 1e-3), max(b, 1e-3), n)
    return float(np.quantile(s, 0.1)), float(np.quantile(s, 0.9))

def p_open(row):
    if re.search(r"24\s*[x×/]\s*7|24 hours|round the clock", row["all_text"], re.I): return 0.95
    return {"hospital": 0.85, "clinic": 0.5}.get(row["ftype"], 0.75)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Understanding the location (origin as a cloud of possible points)
# MAGIC The input can be vague: "Jaipur" covers hundreds of km², and a PIN covers a few. So the origin is a **set of possible points**, not a single point:
# MAGIC
# MAGIC | input | how it's resolved | sampled origin points |
# MAGIC |---|---|---|
# MAGIC | `lat,lon` | exact | ±0.2 km |
# MAGIC | 6-digit PIN | India Post offices of that PIN (cleaned) | the offices of that PIN |
# MAGIC | district | India Post offices in the district | offices in the district |
# MAGIC | city / town | facilities and post offices with that name | their points |
# MAGIC | village / post office | that office | ±1 km |
# MAGIC
# MAGIC Add `, State` to disambiguate (e.g. `Aurangabad, Bihar`). When a name exists in several states we pick the one with the most data
# MAGIC **and say so**, listing the alternatives.

# COMMAND ----------

STATE_DISPLAY = {k: v.title() for k, v in po.groupby("state_n")["statename"].agg(lambda x: x.mode().iloc[0]).items()}
def _st(gg): return STATE_DISPLAY.get(gg["state_n"].iloc[0], gg["state_n"].iloc[0].title())

def resolve_location(text):
    t = text.strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", t)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not in_india(lat, lon) and in_india(lon, lat): lat, lon = lon, lat
        if not in_india(lat, lon): raise ValueError("Those coordinates are outside India.")
        pts = np.column_stack(jitter(lat, lon, 0.2, 50, RNG))
        return _origin(f"coordinates {lat:.4f}, {lon:.4f}", "exact point", pts, lat, lon)
    digits = re.sub(r"\D", "", t)
    if len(digits) == 6 and len(t.replace(" ", "")) <= 7:
        g = po_clean[po_clean["pincode"] == digits]
        if len(g):
            return _origin(f"PIN {digits}", f"{len(g)} post offices of PIN {digits}", g[["lat", "lon"]].to_numpy(), pin_code=digits)
        g = po_clean[po_clean["pincode"].str[:3] == digits[:3]]
        if len(g):
            return _origin(f"PIN area {digits[:3]}xxx", f"approximate: PIN {digits} not found, used the {digits[:3]}xxx postal area ({len(g)} offices)", g[["lat", "lon"]].to_numpy())
        raise ValueError(f"PIN {digits} not found.")
    name, state = (t.rsplit(",", 1) + [""])[:2] if "," in t else (t, "")
    nn, st = norm_name(name), norm_state(state)
    dn = norm_district(name)
    candidates = []
    g = po_clean[po_clean["district_n"].isin({dn, DISTRICT_ALIASES.get(dn, "")} - {""})]
    if st: g = g[g["state_n"] == st]
    for s_, gg in g.groupby("state_n"): candidates.append(("district", _st(gg), gg[["lat", "lon"]].to_numpy(), len(gg)))
    f = fac[(fac["city_n"] == nn) & fac["coords_ok"]]
    if st: f = f[f["state_n"] == st]
    for s_, gg in f.groupby("state_n"): candidates.append(("city", STATE_DISPLAY.get(s_, s_.title()), gg[["lat", "lon"]].to_numpy(), len(gg)))
    o = po_clean[po_clean["office_n"] == nn]
    if st: o = o[o["state_n"] == st]
    for s_, gg in o.groupby("state_n"): candidates.append(("post office / village", _st(gg), gg[["lat", "lon"]].to_numpy(), len(gg)))
    if not candidates and norm_state(name) in set(po_clean["state_n"]):      # a whole state / UT, e.g. "Delhi", "Kerala"
        gg = po_clean[po_clean["state_n"] == norm_state(name)]
        candidates.append(("state / UT", _st(gg), gg[["lat", "lon"]].to_numpy(), len(gg)))
    if not candidates:
        names = list(set(po_clean["district_n"]) | set(fac["city_n"]) | set(po_clean["office_n"]))
        close = difflib.get_close_matches(nn.replace(" ", ""), [x.replace(" ", "") for x in names], n=1, cutoff=0.85)
        if close:
            back = {x.replace(" ", ""): x for x in names}[close[0]]
            res = resolve_location(back + ("," + state if state else ""))
            res["precision"] += f" (interpreted '{name.strip()}' as '{back}')"
            return res
        raise ValueError(f"Couldn't find '{t}'. Try 'Town, State', a 6-digit PIN code, or 'lat,lon'.")
    candidates.sort(key=lambda c: (c[0] == "district", c[3]), reverse=True)
    kind, s_, pts, n = candidates[0]
    others = sorted({c[1] for c in candidates[1:] if c[1] != s_})
    note = f"{kind} '{name.strip().title()}' in {s_}, {n} reference points" + (f". Also found in: {', '.join(others)}. Add ', State' to choose." if others else "")
    if kind == "post office / village": pts = np.vstack([np.column_stack(jitter(p[0], p[1], 1.0, 10, RNG)) for p in pts])
    return _origin(f"{name.strip().title()} ({s_})", note, pts)

def _origin(label, precision, pts, lat=None, lon=None, pin_code=None):
    pts = np.asarray(pts, dtype=float)
    lat = float(np.median(pts[:, 0])) if lat is None else lat
    lon = float(np.median(pts[:, 1])) if lon is None else lon
    spread = float(np.quantile(haversine_km(pts[:, 0], pts[:, 1], lat, lon), 0.8)) if len(pts) > 1 else 0.2
    # rural proxy + district of the origin from the nearest PIN centre
    i = nearest_pin(np.array([lat]), np.array([lon]), max_km=60)[0] if len(pin_geo) else -1
    pr = pin_geo.loc[i] if i >= 0 else None
    return dict(label=label, precision=precision, lat=lat, lon=lon, points=pts, spread_km=spread,
                rural=float(pr["bo_share"]) if pr is not None else 0.5,
                district=(pr["district"] if pr is not None else ""), state=(pr["state"] if pr is not None else ""),
                nf_key=(pr["nf_key"] if pr is not None else None), pin_code=pin_code)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. The three tier models
# MAGIC ### 9a. Shared: candidate search and NFHS need context
# MAGIC **Search radius by tier.** High: 120 km, widened to 250 if empty (an emergency transfer beyond ~4 h by road is rarely useful).
# MAGIC Medium: 200 km, widened to 400. Low: 500 km, widened to 1,000 (people travel far for planned cancer care).
# MAGIC
# MAGIC **District need index (NFHS-5).** For the service's family we take a few indicators, turn each into a z-score across all districts
# MAGIC (sign-adjusted so that higher = more need), average them, and report the district's **percentile**. Every indicator used is listed in the output,
# MAGIC with a small-sample flag where it applies.

# COMMAND ----------

NEED_INDICATORS = {
    "obstetric": [(("institutional_birth_5y",), -1, "institutional births"), (("at_least_4_anc",), -1, "mothers with ≥4 antenatal visits"),
                  (("w15_19", "mothers_or_pregnant"), +1, "teenage pregnancy"), (("births_delivered_by_csection",), -1, "C-section rate (very low = unmet need)")],
    "neonatal": [(("institutional_birth_5y",), -1, "institutional births"), (("child_6_59m", "anaemic"), +1, "child anaemia"), (("stunted",), +1, "child stunting")],
    "renal": [(("w15_plus", "high_or_very_high", "sugar"), +1, "high blood sugar (women)"), (("m15_plus", "high_or_very_high", "sugar"), +1, "high blood sugar (men)"),
              (("blood_pressure",), +1, "raised blood pressure"), (("health_insurance",), -1, "health-insurance coverage")],
    "cardiac": [(("w15_plus", "high_or_very_high", "sugar"), +1, "high blood sugar (women)"), (("overweight_obese",), +1, "overweight/obese women"),
                (("blood_pressure",), +1, "raised blood pressure"), (("health_insurance",), -1, "health-insurance coverage")],
    "trauma": [(("health_insurance",), -1, "health-insurance coverage"), (("hh_electricity",), -1, "household electricity (development proxy)")],
    "burn": [(("clean_fuel",), -1, "households using clean cooking fuel (open-fire cooking raises burn risk)"), (("health_insurance",), -1, "health-insurance coverage")],
    "critical": [(("health_insurance",), -1, "health-insurance coverage")],
    "oncology": [(("tobacco",), +1, "tobacco use"), (("health_insurance",), -1, "health-insurance coverage")],
    "imaging": [(("health_insurance",), -1, "health-insurance coverage")],
}
_NEED_CACHE = {}
def need_index(family, key):
    if key is None or key not in nf.index: return None
    if family not in _NEED_CACHE:
        cols, zs = [], []
        for frags, sign, label in NEED_INDICATORS.get(family, []):
            c = nf_col(*frags)
            if c is None: continue
            x = nf[c]
            if x.notna().sum() < 20: continue
            z = sign * (x - x.mean()) / (x.std() or 1)
            cols.append((c, sign, label)); zs.append(z)
        if not zs: _NEED_CACHE[family] = None
        else:
            comp = pd.concat(zs, axis=1).mean(axis=1)
            _NEED_CACHE[family] = (cols, comp.rank(pct=True))
    if _NEED_CACHE[family] is None: return None
    cols, pct = _NEED_CACHE[family]
    if key not in pct.index or pd.isna(pct.loc[key]): return None
    detail = []
    for c, sign, label in cols:
        v = nf.loc[key, c]
        if pd.notna(v): detail.append(f"{label}: {v:.1f}%" + (" (small sample)" if bool(NF_SMALL.loc[key, c]) else ""))
    return dict(percentile=float(pct.loc[key]), detail=detail, district=nf.loc[key, "district_name"], state=nf.loc[key, "state_ut"])

RADIUS = {"High": (120, 250), "Medium": (200, 400), "Low": (500, 1000)}

def find_candidates(spec, origin, tier):
    notes = []
    d = pd.Series(haversine_km(origin["lat"], origin["lon"], fac["plat"], fac["plon"]), index=fac.index)
    for rad in RADIUS[tier]:
        near = fac[(d <= rad) & fac["plat"].notna()]
        rows = []
        for i, r_ in near.iterrows():
            if not re.search("|".join(spec["strong"] + spec["support"]), r_["all_text"], re.I) and not registry_matches(r_): continue
            cc_ = claim_court(r_, spec)
            if not cc_["evidence"] and not cc_["registry_confirms"]: continue
            rows.append({**r_.to_dict(), **{f"cc_{k}": v for k, v in cc_.items()}, "dist_km": float(d[i])})
        cand = pd.DataFrame(rows)
        if len(cand):
            cand = cand.sort_values("cc_p", ascending=False).drop_duplicates("dup_key").reset_index(drop=True)
            if (cand["cc_has_strong"] | cand["cc_registry_confirms"].map(bool)).any() or rad == RADIUS[tier][-1]:
                return cand, notes
        notes.append(f"No well-evidenced facility within {rad} km, so the search was widened.")
    return pd.DataFrame(), notes

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9b. HIGH criticality: Monte-Carlo Plan-A / Plan-B router
# MAGIC **Goal:** maximise the **expected outcome** for a time-critical patient. That is not the same as going to the nearest place that *claims* the service.
# MAGIC
# MAGIC Each of the `MC_SIMULATIONS` simulated "worlds" draws:
# MAGIC 1. **Where the patient is.** One of the origin points from section 8.
# MAGIC 2. **Where each facility really is (location validity).** With probability `p_loc` at its stated coordinates (±0.3 km). Otherwise at its PIN-code
# MAGIC    centre (± half the PIN radius), or with wide noise if there is no PIN. A facility whose map pin contradicts its PIN code is therefore *uncertain*, not just penalised.
# MAGIC 3. **Road and traffic.** Road distance = straight-line × a detour factor (lognormal, median 1.3 urban / 1.5 rural, using the PIN's branch-office share
# MAGIC    as the rural proxy). Speed is lognormal (median 30 km/h urban, 40 rural). Plus 5–20 min to set off.
# MAGIC 4. **Whether each facility can treat the patient on arrival.** Bernoulli(P(offers service) from the Claim Court × P(open/available)), where
# MAGIC    P(open) is 0.95 with 24×7 evidence, 0.85 for a hospital, 0.5 for a clinic.
# MAGIC
# MAGIC A **plan** is "go to A; if A can't treat, transfer to B (+10 min handover); if B can't either, a further ~90 min referral".
# MAGIC Outcome for a world = **quality × e^(−T/τ)**, where T is the time to definitive care, τ = 0.75 × the clinical window, and quality = 0.5–1.0 from how much
# MAGIC of the **capability bundle** is evidenced (e.g. for C-section: OT, anaesthesia, blood bank, newborn care, obstetrician, 24×7).
# MAGIC This is why a **slightly farther but better-equipped, better-evidenced** facility can win: its higher quality and lower chance of a
# MAGIC wasted trip outweigh a few extra minutes.
# MAGIC
# MAGIC We evaluate all ordered pairs among the top 8 candidates and report, for each recommended first stop: expected outcome,
# MAGIC P(definitive care within the clinical window), median and 90th-percentile time. We also compare against the naive **"nearest claimed facility first"** plan.

# COMMAND ----------

def _rad(r_): return float(r_["pin_radius"]) if pd.notna(r_["pin_radius"]) else 5.0

def mc_router(cand, origin, spec, tri, n=MC_SIMULATIONS, top_m=8):
    window, tau = spec["window_min"], spec["tau"]
    c = cand.copy()
    c["p_treat"] = c["cc_p"] * c.apply(p_open, axis=1)
    c["t_est"] = 10 + 60 * 1.4 * c["dist_km"] / 35
    c["h"] = c["p_treat"] * c["cc_q"] * np.exp(-c["t_est"] / tau)
    c = c.sort_values("h", ascending=False).head(top_m).reset_index(drop=True)
    M = len(c)
    rng = np.random.default_rng(RANDOM_SEED)
    o = origin["points"][rng.integers(0, len(origin["points"]), n)]
    # facility positions
    F = np.zeros((n, M, 2))
    for j, r_ in c.iterrows():
        if r_["coords_ok"]:
            at_stated = rng.random(n) < r_["p_loc"]
            sl, so = jitter(r_["lat"], r_["lon"], 0.3, n, rng)
            if pd.notna(r_["pin_lat"]):
                al, ao = jitter(r_["pin_lat"], r_["pin_lon"], _rad(r_) / 2, n, rng)
            else:
                al, ao = jitter(r_["lat"], r_["lon"], 5.0, n, rng)
            F[:, j, 0] = np.where(at_stated, sl, al); F[:, j, 1] = np.where(at_stated, so, ao)
        else:
            sig = _rad(r_) / 2 if pd.notna(r_["pin_lat"]) else 8.0
            F[:, j, 0], F[:, j, 1] = jitter(r_["plat"], r_["plon"], sig, n, rng)
    rural = origin["rural"]
    circ = np.exp(rng.normal(np.log(1.3 + 0.2 * rural), 0.12, (n, 1)))
    speed = np.exp(rng.normal(np.log(30 + 10 * rural), 0.25, (n, 1)))
    setoff = rng.uniform(5, 20, (n, 1))
    d_o = haversine_km(o[:, [0]], o[:, [1]], F[:, :, 0], F[:, :, 1])
    t_o = setoff + 60 * circ * d_o / speed
    treat = rng.random((n, M)) < c["p_treat"].to_numpy()[None, :]
    q = c["cc_q"].to_numpy()
    def plan(a, b):
        d_ab = haversine_km(F[:, a, 0], F[:, a, 1], F[:, b, 0], F[:, b, 1])
        t_ab = 10 + 60 * circ[:, 0] * d_ab / speed[:, 0]
        T = np.where(treat[:, a], t_o[:, a], np.where(treat[:, b], t_o[:, a] + t_ab, t_o[:, a] + t_ab + 10 + 90))
        Q = np.where(treat[:, a], q[a], np.where(treat[:, b], q[b], 0.6))
        U = Q * np.exp(-T / tau)
        within = (T <= window) & (treat[:, a] | treat[:, b])
        return dict(a=a, b=b, EU=float(U.mean()), p_window=float(within.mean()), t_med=float(np.median(T)), t_p90=float(np.quantile(T, 0.9)),
                    p_a=float(treat[:, a].mean()), t_a_med=float(np.median(t_o[:, a])))
    if M == 1:
        b0 = plan(0, 0); plans = [b0]
    else:
        plans = [plan(a, b) for a in range(M) for b in range(M) if a != b]
    best_by_a = {}
    for p_ in plans:
        if p_["a"] not in best_by_a or p_["EU"] > best_by_a[p_["a"]]["EU"]: best_by_a[p_["a"]] = p_
    ranked = sorted(best_by_a.values(), key=lambda x: -x["EU"])
    # naive baseline: nearest facility with a strong claim first, then the next nearest
    strong = [j for j in np.argsort(np.median(t_o, axis=0)) if c.loc[j, "cc_has_strong"]] or list(np.argsort(np.median(t_o, axis=0)))
    base = plan(strong[0], strong[1]) if len(strong) > 1 else plan(strong[0], strong[0])
    return c, ranked, base

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9c. MEDIUM criticality: access, need and congestion model (extended Idea 3)
# MAGIC Time still matters (days, not minutes), but now **access, capacity and recurring travel** dominate. The model has five parts:
# MAGIC 1. **Demand points:** India Post PIN centres within 120 km of the user. Weight = district population (if uploaded) split across the district's PINs by office count,
# MAGIC    otherwise the number of post offices (a settlement-count proxy, stated in the output).
# MAGIC 2. **Supply:** each candidate's capacity (beds; if missing, imputed as hospital 50 / clinic 6 / other 15, flagged) × P(offers service).
# MAGIC 3. **Enhanced two-step floating catchment (E2SFCA)** with Gaussian distance decay inside a 60 km catchment. Step 1: each facility's supply-to-demand
# MAGIC    ratio. Step 2: each place sums the ratios it can reach. This gives the user's **access score**, compared with the area around them.
# MAGIC 4. **Huff gravity model for congestion:** P(patient at i uses j) ∝ supply_j · e^(−d_ij/30). The expected load on j divided by its capacity gives **pressure**
# MAGIC    (how crowded it will be relative to its size).
# MAGIC 5. **Multi-criteria score** with explicit weights: validity 0.30 · travel 0.20 (raised for recurring care) · quality 0.20 · credibility 0.10 · low pressure 0.10 · affordability 0.10
# MAGIC    (PM-JAY empanelment for this specialty, when the registry is uploaded).
# MAGIC
# MAGIC **Going further:** (a) **weight-robustness:** 1,000 random weight sets drawn from a Dirichlet distribution around the defaults; we report how often each facility stays #1 or top-3,
# MAGIC so the user sees whether the ranking depends on our weight choice. (b) **Pareto check:** is any other option better on *every* criterion? If not, the option is
# MAGIC "non-dominated", a choice that only involves trade-offs. (c) **District need (NFHS)** and **access percentile**, to flag care deserts.

# COMMAND ----------

def load_population():
    if not table_exists(POPULATION_TABLE): return None
    p = spark.table(POPULATION_TABLE).toPandas(); p.columns = [c.lower() for c in p.columns]
    p["k"] = p["state"].map(norm_state) + "|" + p["district"].map(norm_district)
    return p.set_index("k")["population"].astype(float).to_dict()
POP = load_population()
print("district population table:", "loaded" if POP else "not found → post-office-count proxy for demand")

def e2sfca_and_huff(cand, origin, d0=60.0, beta_km=30.0):
    dp = pin_geo[haversine_km(origin["lat"], origin["lon"], pin_geo["lat"], pin_geo["lon"]) <= 120].copy()
    if dp.empty: dp = pin_geo.iloc[[nearest_pin(np.array([origin['lat']]), np.array([origin['lon']]), 500)[0]]].copy()
    if POP:
        tot = dp.groupby(["state_n", "district_n"])["n_offices"].transform("sum")
        dp["w"] = [POP.get(f"{s}|{d}", np.nan) for s, d in zip(dp["state_n"], dp["district_n"])] * dp["n_offices"] / tot
        dp["w"] = dp["w"].fillna(dp["n_offices"] * 1000)
        demand_note = "district population (uploaded) split across PINs by post-office count"
    else:
        dp["w"] = dp["n_offices"].astype(float); demand_note = "post-office count per PIN (proxy; upload district population to improve)"
    cap = cand["beds"].copy(); imputed = cap.isna()
    cap[imputed] = cand.loc[imputed, "ftype"].map({"hospital": 50, "clinic": 6}).fillna(15)
    S = cap.to_numpy() * cand["cc_p"].to_numpy()
    D = haversine_km(dp["lat"].to_numpy()[:, None], dp["lon"].to_numpy()[:, None], cand["plat"].to_numpy()[None, :], cand["plon"].to_numpy()[None, :])
    W = np.where(D <= d0, (np.exp(-0.5 * (D / d0) ** 2) - np.exp(-0.5)) / (1 - np.exp(-0.5)), 0.0)
    R = S / np.maximum((dp["w"].to_numpy()[:, None] * W).sum(axis=0), 1e-9)
    A = (W * R[None, :]).sum(axis=1)
    d_or = haversine_km(origin["lat"], origin["lon"], cand["plat"].to_numpy(), cand["plon"].to_numpy())
    W0 = np.where(d_or <= d0, (np.exp(-0.5 * (d_or / d0) ** 2) - np.exp(-0.5)) / (1 - np.exp(-0.5)), 0.0)
    A0 = float((W0 * R).sum())
    attr = S[None, :] * np.exp(-D / beta_km)
    P = attr / np.maximum(attr.sum(axis=1, keepdims=True), 1e-12)
    load = (dp["w"].to_numpy()[:, None] * P).sum(axis=0)
    pressure = load / np.maximum(cap.to_numpy(), 1)
    pressure = pressure / max(float(np.mean(pressure)), 1e-12)          # relative to the local average (1.0 = average)
    access_pct = float((A <= A0).mean()) if len(A) else np.nan
    return dict(access=A0, access_pct=access_pct, pressure=pressure, cap=cap.to_numpy(), cap_imputed=imputed.to_numpy(), demand_note=demand_note, n_points=len(dp))

def mcda(cand, origin, spec, tier, tau_travel, n_draws=1000):
    c = cand.copy()
    visits = spec["visits"]
    c["travel_min"] = 10 + 60 * (1.3 + 0.2 * origin["rural"]) * c["dist_km"] / (30 + 10 * origin["rural"])
    c["burden_min"] = c["travel_min"] * (1 + visits / 4)
    c["s_travel"] = np.exp(-c["burden_min"] / tau_travel)
    c["s_valid"] = c["cc_p"]
    c["s_quality"] = (c["cc_q"] - 0.5) / 0.5
    c["s_cred"] = c["cc_registry_confirms"].map(lambda x: 1.0 if x else 0.0) if REG is not None else 0.5
    pm = c["cc_registry"].map(lambda regs: any(m["registry"].upper().startswith("PMJAY") for m in regs))
    c["s_afford"] = np.where(pm, 1.0, 0.5)
    ctx = e2sfca_and_huff(c, origin)
    pr = pd.Series(ctx["pressure"]); c["pressure"] = pr.values; c["cap_used"] = ctx["cap"]; c["cap_imputed"] = ctx["cap_imputed"]
    c["s_pressure"] = 1 - pr.rank(pct=True).values if len(pr) > 1 else 0.5
    crit = ["s_valid", "s_travel", "s_quality", "s_cred", "s_pressure", "s_afford"]
    w = np.array([0.30, 0.20 + 0.05 * min(visits, 12) / 4, 0.20, 0.10, 0.10, 0.10]); w = w / w.sum()
    X = c[crit].to_numpy(dtype=float)
    c["score"] = X @ w
    draws = RNG.dirichlet(w * 20, n_draws)
    S = X @ draws.T
    order = np.argsort(-S, axis=0)
    c["top1_share"] = [(order[0] == i).mean() for i in range(len(c))]
    c["top3_share"] = [(order[:3] == i).any(axis=0).mean() for i in range(len(c))]
    dominated = []
    for i in range(len(c)):
        dominated.append(any((X[j] >= X[i]).all() and (X[j] > X[i]).any() for j in range(len(c)) if j != i))
    c["pareto"] = ~np.array(dominated)
    c = c.sort_values("score", ascending=False).reset_index(drop=True)
    return c, dict(weights=dict(zip(["validity", "travel", "quality", "credibility", "low pressure", "affordability"], w.round(3))), **ctx)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9d. LOW criticality: care-desert context + LLM **Evidence Court**
# MAGIC With weeks to decide, the risk is choosing a facility whose claim is false (e.g. "cancer hospital" that only does screening). So we spend effort on
# MAGIC **verifying claims**, not on minutes.
# MAGIC 1. Shortlist with the same multi-criteria model as Medium (wide radius, distance weighted by visits per month: chemo cycles mean repeat travel).
# MAGIC 2. **Evidence Court** for the top 5. For each facility:
# MAGIC    * **Prosecutor** (LLM) argues the claim is false or overstated. Every point must quote the record.
# MAGIC    * **Defender** (LLM) argues it is true. Every point must quote the record.
# MAGIC    * **Fact-checker** (plain code, not an LLM) keeps only points whose quote **appears word for word** in the record. Invented evidence is removed and counted.
# MAGIC    * **Judge** (LLM) sees only verified points and returns `supported / refuted / insufficient` + confidence c.
# MAGIC 3. The verdict updates the Claim Court probability as **one bounded witness**: supported → LR = 1+2c (≤3), refuted → 1/(1+2c), insufficient → 1.
# MAGIC    The LLM can shift the decision but **cannot dominate** the evidence, and every step is logged.
# MAGIC 4. If no LLM endpoint answers, a **rule-based court** builds the same prosecutor and defender arguments from the Claim Court's own witnesses, so the output format is identical.

# COMMAND ----------

LLM_ENDPOINT = None
def probe_llm():
    global LLM_ENDPOINT
    if not USE_LLM: return None
    for ep in LLM_CANDIDATES:
        try:
            r = spark.sql(f"SELECT ai_query('{ep}', 'Reply with the single word OK') AS r").collect()[0]["r"]
            if r: LLM_ENDPOINT = ep; print("LLM endpoint:", ep); return ep
        except Exception as e:
            print(f"  {ep}: not available ({str(e)[:80]})")
    print("No LLM endpoint answered → rule-based Evidence Court will be used."); return None

def llm_batch(prompts):
    """Run many prompts in one Spark job through ai_query. Returns a list of strings (None on failure)."""
    if not LLM_ENDPOINT or not prompts: return [None] * len(prompts)
    try:
        df = spark.createDataFrame(pd.DataFrame({"i": list(range(len(prompts))), "prompt": prompts}))
        out = df.selectExpr("i", f"ai_query('{LLM_ENDPOINT}', prompt) AS r").toPandas().sort_values("i")
        return [to_text(x) or None for x in out["r"]]
    except Exception as e:
        print("LLM batch failed:", str(e)[:200]); return [None] * len(prompts)

def parse_json(s):
    if not s: return None
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
    if not m: return None
    try: return json.loads(m.group(1))
    except Exception:
        try: return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(1)))
        except Exception: return None

def llm_estimate_window(service_text):
    if not LLM_ENDPOINT: return None
    out = llm_batch([f"How time-critical is the medical service '{service_text}'? Reply ONLY JSON: "
                     '{"window_hours": <maximum safe delay before treatment, number>, "why": "<10 words>"}'])[0]
    j = parse_json(out)
    try:
        h = float(j["window_hours"]); h = min(max(h, 0.5), 24 * 90)
        return dict(window_min=h * 60, why=str(j.get("why", ""))[:80])
    except Exception: return None

def record_text(r_):
    return (f"NAME: {r_['name']}\nTYPE: {r_['ftype'] or 'unknown'}\nPROCEDURES: {' ; '.join(r_['procedure'])}\nEQUIPMENT: {' ; '.join(r_['equipment'])}\n"
            f"CAPABILITIES: {' ; '.join(r_['capability'])}\nSPECIALTIES: {' ; '.join(r_['specialties'])}\nDESCRIPTION: {r_['description']}")[:3500]

def _verify(points, rec):
    ok, bad = [], 0
    low = re.sub(r"\s+", " ", rec.lower())
    for p_ in points or []:
        q = re.sub(r"\s+", " ", to_text(p_.get("quote", "")).lower()).strip(' "\'')
        if len(q) >= 4 and q in low: ok.append(dict(point=to_text(p_.get("point", ""))[:240], quote=to_text(p_.get("quote", ""))[:240]))
        else: bad += 1
    return ok, bad

def evidence_court(short, spec):
    recs = [record_text(r_) for _, r_ in short.iterrows()]
    svc = spec["key"]
    pro_prompts = [f"You are a PROSECUTOR auditing a hospital record. Argue that this facility does NOT really provide {svc} (or that the claim is weak/overstated). "
                   "Use ONLY the record. Reply ONLY JSON: {\"points\": [{\"point\": \"...\", \"quote\": \"<exact words copied from the record>\"}]} (max 4 points).\nRECORD:\n" + rc for rc in recs]
    def_prompts = [f"You are a DEFENDER. Argue that this facility DOES provide {svc}. Use ONLY the record. "
                   "Reply ONLY JSON: {\"points\": [{\"point\": \"...\", \"quote\": \"<exact words copied from the record>\"}]} (max 4 points).\nRECORD:\n" + rc for rc in recs]
    pro_raw = llm_batch(pro_prompts) if LLM_ENDPOINT else [None] * len(recs)
    def_raw = llm_batch(def_prompts) if LLM_ENDPOINT else [None] * len(recs)
    sessions, judge_prompts = [], []
    for k, (_, r_) in enumerate(short.iterrows()):
        jp, jd = parse_json(pro_raw[k]), parse_json(def_raw[k])
        mode = "LLM" if (jp is not None and jd is not None) else "rules"
        if mode == "LLM":
            pro, pbad = _verify(jp.get("points", []) if isinstance(jp, dict) else [], recs[k])
            dfn, dbad = _verify(jd.get("points", []) if isinstance(jd, dict) else [], recs[k])
        else:
            pro, dfn, pbad, dbad = rule_arguments(r_, spec)
        sessions.append(dict(mode=mode, prosecution=pro, defence=dfn, rejected=pbad + dbad))
        judge_prompts.append(f"You are a neutral JUDGE. Question: does this facility really provide {svc}? Consider ONLY these verified arguments.\n"
                             f"PROSECUTION: {json.dumps(pro)}\nDEFENCE: {json.dumps(dfn)}\n"
                             "Reply ONLY JSON: {\"verdict\": \"supported\"|\"refuted\"|\"insufficient\", \"confidence\": <0..1>, \"reason\": \"<25 words>\"}")
    jraw = llm_batch(judge_prompts) if LLM_ENDPOINT else [None] * len(recs)
    for k, (_, r_) in enumerate(short.iterrows()):
        j = parse_json(jraw[k]) if sessions[k]["mode"] == "LLM" else None
        if isinstance(j, dict) and j.get("verdict") in ("supported", "refuted", "insufficient"):
            v, cf, why = j["verdict"], float(np.clip(float(j.get("confidence", 0.5) or 0.5), 0, 1)), to_text(j.get("reason"))[:200]
        else:
            v, cf, why = rule_judge(r_)
            if sessions[k]["mode"] == "LLM": sessions[k]["mode"] = "LLM arguments + rule judge"
        lr = 1 + 2 * cf if v == "supported" else (1 / (1 + 2 * cf) if v == "refuted" else 1.0)
        p0 = r_["cc_p"]; lo = math.log(p0 / (1 - p0)) + math.log(lr); p1 = 1 / (1 + math.exp(-lo))
        sessions[k].update(verdict=v, confidence=cf, reason=why, lr=lr, p_before=p0, p_after=p1)
    return sessions

def rule_arguments(r_, spec):
    pro, dfn = [], []
    for st in r_["cc_steps"][1:]:
        item = dict(point=st["label"], quote=st["quote"] or "")
        (dfn if (st["lr"] or 1) > 1 else pro).append(item)
    if r_["cc_bundle_missing"]: pro.append(dict(point=f"No evidence of: {', '.join(r_['cc_bundle_missing'])}", quote=""))
    if r_["p_loc"] < 0.5: pro.append(dict(point=f"Map location disagrees with its PIN code (location validity {r_['p_loc']:.0%})", quote=""))
    if REG is not None and not r_["cc_registry_confirms"]: pro.append(dict(point="Not confirmed by any uploaded official registry", quote=""))
    return pro[:5], dfn[:5], 0, 0

def rule_judge(r_):
    p = r_["cc_p"]; structured = bool(set(r_["cc_strong_fields"]) & STRUCTURED)
    if p >= 0.6 and structured: return "supported", min(1.0, 2 * abs(p - 0.5)), "Specific structured evidence outweighs the objections (rule-based judge)."
    if p < 0.25 or r_["cc_negations"]: return "refuted", min(1.0, 2 * abs(p - 0.5)), "Evidence is weak or contradicted (rule-based judge)."
    return "insufficient", 0.3, "Mixed or thin evidence (rule-based judge)."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Orchestration and explanation cards (WHY / HOW)
# MAGIC `recommend(service, location)` runs triage → location → candidates (+ Claim Court + registry) → the tier model → explanations.
# MAGIC Every option carries a **WHY** block (exists · offers the service · fits the location · credibility, with quotes) and a **HOW** block
# MAGIC (the tier model's numbers and the trade-off made). Results are also returned as a table.
# MAGIC
# MAGIC **Scenarios handled:** unknown service; ambiguous or unknown location; no candidates (radius widened, then advice); only weak candidates (shown as
# MAGIC "unverified leads"); facilities with bad or missing coordinates; duplicates collapsed; no NFHS match; registry or population tables missing; LLM unavailable.
# MAGIC High tier always shows the emergency banner (call 108 / 112).

# COMMAND ----------

def _exists_lines(r_):
    out = []
    loc = {"stated coordinates": f"map pin {'agrees with' if r_['p_loc']>=0.75 else 'is questionable versus'} its PIN-code area (location validity {r_['p_loc']:.0%}"
                                  + (f", {r_['km_from_pin']:.1f} km from PIN centre" if pd.notna(r_['km_from_pin']) else "") + ")",
           "PIN-code centre": "no valid map pin; position estimated from its PIN code",
           "city centre": "no valid map pin or PIN; position estimated from its city centre"}.get(r_["pos_source"], "position unknown")
    out.append(f"Exists: {r_['name']} ({r_['ftype'] or 'type unknown'}), {', '.join(x for x in [r_['address'], r_['city'], r_['district'] or r_['region'], r_['postcode']] if x)}; {loc}.")
    extra = [f"phone {r_['phone']}" if r_['phone'] else "", f"{int(r_['beds'])} beds" if pd.notna(r_['beds']) else "",
             f"{int(r_['doctors'])} doctors" if pd.notna(r_['doctors']) else "", f"last social post {r_['last_post']}" if r_['last_post'] else ""]
    extra = [x for x in extra if x]
    if extra: out.append("Signs of an operating facility: " + ", ".join(extra) + ".")
    if r_["dup_count"] > 1: out.append(f"Note: appears {int(r_['dup_count'])} times in the dataset (duplicates merged).")
    return out

def _offers_lines(r_, spec):
    out = [f"Offers {spec['key']}: P = {r_['cc_p']:.0%} (80% range {r_['cc_p_lo']:.0%}–{r_['cc_p_hi']:.0%}), from {r_['cc_n_witnesses']} piece{'s' if r_['cc_n_witnesses'] != 1 else ''} of evidence:"]
    for e in r_["cc_evidence"][:4]:
        out.append(f"  • [{e['field']}, {e['kind']}] “{e['quote'][:200]}”")
    if r_["cc_bundle_found"]: out.append(f"  • Capability bundle present: {', '.join(r_['cc_bundle_found'])}" + (f"; not evidenced: {', '.join(r_['cc_bundle_missing'])}" if r_["cc_bundle_missing"] else ""))
    for f, qt in r_["cc_negations"][:1]: out.append(f"  • Caution, possible negation in {f}: “{qt[:160]}”")
    return out

def _cred_lines(r_):
    if REG is None: return ["Credibility: no official registry uploaded, so the claim rests on the facility's web-derived record only."]
    if r_["cc_registry_confirms"]:
        m = r_["cc_registry_confirms"][0]
        state_ = ("valid to " + m["valid_to"]) if m.get("current", True) else ("certification EXPIRED " + m["valid_to"] + ", counted as weaker evidence")
        if m.get("status") == "conditional": state_ += ", conditional"
        return [f"Credibility: confirmed by official list **{m['registry']}** as “{m['name']}” (name match {m['score']}; {state_}"
                + (f"; {m['source_ref']}" if m.get('source_ref') else "") + (f"; source {m['source_url']}" if m['source_url'] else "") + ")."]
    if r_["cc_registry"]:
        return [f"Credibility: found in {r_['cc_registry'][0]['registry']} (so it exists), but that list does not confirm this service."]
    return ["Credibility: not found in the uploaded official lists. Absence is not evidence against it, especially for private facilities."]

def _waterfall(r_):
    return " → ".join([f"{s['p']:.0%}" if s["lr"] is None else f"×{s['lr']:.2g} {s['p']:.0%}" for s in r_["cc_steps"]])

def recommend(service_text, location_text, top_k=3):
    t0 = time.time()
    tri = triage(service_text)
    res = dict(query=dict(service=service_text, location=location_text), triage=tri, options=[], leads=[], notes=[], context={})
    try:
        origin = resolve_location(location_text)
    except ValueError as e:
        res["notes"].append(str(e)); return res
    res["origin"] = {k: v for k, v in origin.items() if k != "points"}
    if tri["key"] is None:
        res["notes"].append("Unknown service: no evidence patterns exist for it, so facilities can't be verified. Try a listed service name.")
        return res
    spec = SERVICE_CATALOG[tri["key"]]; tier = tri["tier"]
    cand, notes = find_candidates(spec, origin, tier); res["notes"] += notes
    need = need_index(spec["family"], origin["nf_key"])
    res["context"]["need"] = need
    if not origin["nf_key"]: res["notes"].append("Your district could not be matched to NFHS-5, so there is no district health context.")
    if cand.empty:
        res["notes"].append(f"No facility within {RADIUS[tier][-1]} km has any evidence of {spec['key']}."
                            + (" Call 108 (ambulance) / 112 and go to the nearest district hospital." if tier == "High" else ""))
        return res
    strong = cand[cand["cc_has_strong"] | cand["cc_registry_confirms"].map(bool)]
    weak = cand[~cand.index.isin(strong.index)]
    if strong.empty:
        res["notes"].append("Only weak evidence was found nearby. The options below are unverified leads; phone before travelling.")
        strong = weak; weak = weak.iloc[0:0]
    if tier == "High":
        c, ranked, base = mc_router(strong, origin, spec, tri)
        for p_ in ranked[:top_k]:
            r_ = c.loc[p_["a"]]; rb = c.loc[p_["b"]]
            how = [f"Model: Monte-Carlo Plan-A/Plan-B router ({MC_SIMULATIONS:,} simulated journeys). Criticality High because the clinical window is {fmt_window(spec['window_min'])}.",
                   f"Plan: go here first; if they cannot treat, transfer to **{rb['name']}**.",
                   f"Expected outcome score {p_['EU']:.2f} (1.0 = full-capability care at time 0). P(definitive care within {fmt_window(spec['window_min'])}) = {p_['p_window']:.0%}.",
                   f"Travel time to here: median {p_['t_a_med']:.0f} min. Time to definitive care: median {p_['t_med']:.0f} min, 90th percentile {p_['t_p90']:.0f} min.",
                   f"P(this facility can treat on arrival) = {p_['p_a']:.0%} (= P(offers service) {r_['cc_p']:.0%} × P(open/available) {p_open(r_):.0%}). Quality factor {r_['cc_q']:.2f} from the capability bundle.",
                   f"Uncertainty modelled: your exact position ({origin['precision']}), this facility's true location (validity {r_['p_loc']:.0%}), road detours and speed (rural share {origin['rural']:.0%}), and whether each facility can actually treat."]
            if p_["a"] != base["a"]:
                rn = c.loc[base["a"]]
                how.append(f"Trade-off: the nearest claimed facility, **{rn['name']}** (median {base['t_a_med']:.0f} min), scores only {base['EU']:.2f}: "
                           f"P(can treat) {base['p_a']:.0%}, quality {rn['cc_q']:.2f}. This option is {p_['t_a_med']-base['t_a_med']:+.0f} min further but "
                           f"{'more likely to treat' if p_['p_a']>base['p_a'] else 'better equipped'}, so its expected outcome is {p_['EU']-base['EU']:+.2f} higher.")
            else:
                how.append("This is also the nearest claimed facility, so speed and evidence agree.")
            res["options"].append(_option(r_, spec, how, dict(EU=p_["EU"], p_window=p_["p_window"], t_med=p_["t_med"], backup=rb["name"])))
        res["context"]["baseline"] = dict(name=c.loc[base["a"]]["name"], EU=base["EU"])
    elif tier == "Medium":
        c, ctx = mcda(strong, origin, spec, tier, tau_travel=180)
        res["context"]["access"] = ctx
        for _, r_ in c.head(top_k).iterrows():
            how = [f"Model: access & need model + multi-criteria score. Criticality Medium (window {fmt_window(spec['window_min'])}): hours to days matter less than reliability, capacity and recurring travel.",
                   f"Score {r_['score']:.2f} = " + " + ".join(f"{w:.2f}×{lbl}" for lbl, w in ctx["weights"].items()) +
                   f" with validity {r_['s_valid']:.2f}, travel {r_['s_travel']:.2f} (≈{r_['travel_min']:.0f} min each way" + (f", ×{1+spec['visits']/4:.1f} for {spec['visits']} visits/month" if spec['visits'] else "") +
                   f"), quality {r_['s_quality']:.2f}, credibility {r_['s_cred']:.2f}, low pressure {r_['s_pressure']:.2f}, affordability {r_['s_afford']:.2f}.",
                   f"Congestion (Huff model): expected patients per bed ≈ {r_['pressure']:.1f}× the average of nearby options" + (" (capacity imputed: no bed count listed)" if r_["cap_imputed"] else f" ({int(r_['cap_used'])} beds)") + ".",
                   f"Robustness: stays #1 in {r_['top1_share']:.0%} and top-3 in {r_['top3_share']:.0%} of 1,000 random weightings. " + ("Pareto-optimal: no other option is better on every criterion." if r_["pareto"] else "Dominated by another option on all criteria (shown for completeness)."),
                   f"Area access (E2SFCA, 60 km Gaussian catchment): your location is at the {ctx['access_pct']:.0%} percentile of {ctx['n_points']} nearby places. Demand weights: {ctx['demand_note']}."]
            res["options"].append(_option(r_, spec, how, dict(score=r_["score"], top1=r_["top1_share"], pareto=bool(r_["pareto"]))))
    else:
        c, ctx = mcda(strong, origin, spec, tier, tau_travel=480)
        res["context"]["access"] = ctx
        short = c.head(5).reset_index(drop=True)
        sessions = evidence_court(short, spec)
        short["court_p"] = [s["p_after"] for s in sessions]
        wv = ctx["weights"]["validity"]
        short["final"] = short["score"] - wv * short["s_valid"] + wv * short["court_p"]
        short["session"] = sessions
        short = short.sort_values("final", ascending=False)
        for _, r_ in short.head(top_k).iterrows():
            s = r_["session"]
            how = [f"Model: care-desert context + Evidence Court ({s['mode']}). Criticality Low (window {fmt_window(spec['window_min'])}): weeks to decide, so effort goes into verifying the claim.",
                   f"Court: {len(s['prosecution'])} prosecution and {len(s['defence'])} defence points survived fact-checking" + (f"; {s['rejected']} invented or unquotable points were thrown out" if s['rejected'] else "") + ".",
                   *[f"  – Prosecution: {a['point']}" + (f" (“{a['quote'][:120]}”)" if a['quote'] else "") for a in s["prosecution"][:2]],
                   *[f"  + Defence: {a['point']}" + (f" (“{a['quote'][:120]}”)" if a['quote'] else "") for a in s["defence"][:2]],
                   f"Judge: **{s['verdict']}** (confidence {s['confidence']:.2f}). {s['reason']} → bounded update ×{s['lr']:.2f}: P {s['p_before']:.0%} → {s['p_after']:.0%}.",
                   f"Final score {r_['final']:.2f} = multi-criteria score with the court-updated validity (travel ≈ {r_['travel_min']:.0f} min each way" + (f", {spec['visits']} visits/month" if spec['visits'] else "") + f"; quality {r_['s_quality']:.2f}; credibility {r_['s_cred']:.2f})."]
            res["options"].append(_option(r_, spec, how, dict(final=r_["final"], verdict=s["verdict"], court_p=r_["court_p"])))
    for _, r_ in weak.head(2).iterrows():
        res["leads"].append(dict(name=r_["name"], city=r_["city"], dist_km=round(r_["dist_km"], 1), p=round(r_["cc_p"], 2), phone=r_["phone"]))
    res["runtime_s"] = round(time.time() - t0, 1)
    return res

def _option(r_, spec, how, metrics):
    why = _exists_lines(r_) + _offers_lines(r_, spec) + _cred_lines(r_) + [f"Fits the location: ≈{r_['dist_km']:.0f} km straight-line (road distance is usually 30–50% more)."]
    how = how + [f"Claim Court path (Bayesian, expert-set likelihood ratios): {_waterfall(r_)}."]
    return dict(fid=r_["fid"], name=r_["name"], city=sv(r_["city"]), district=sv(r_["district"]), phone=sv(r_["phone"]), source_url=sv(r_["source_url"]),
                dist_km=round(float(r_["dist_km"]), 1), p_offers=round(float(r_["cc_p"]), 3), p_loc=round(float(r_["p_loc"]), 2),
                quality=round(float(r_["cc_q"]), 2), registry=[m["registry"] for m in r_["cc_registry_confirms"]], why=why, how=how, metrics=metrics)

# COMMAND ----------

TIER_COLOR = {"High": "#cf222e", "Medium": "#9a6700", "Low": "#1a7f37"}
def _md(s):
    s = html.escape(s)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

def render(res):
    tri = res["triage"]; col_ = TIER_COLOR.get(tri["tier"], "#555")
    h = [f"<div style='font-family:system-ui,sans-serif;max-width:980px'>"]
    if tri["tier"] == "High":
        h.append("<div style='background:#ffebe9;border:1px solid #cf222e;padding:8px;border-radius:6px'><b>Emergency:</b> call <b>108</b> (ambulance) or <b>112</b> now. "
                 "This tool helps choose where to go; it does not replace emergency services. Phone the facility on the way if you can.</div>")
    h.append(f"<h2 style='margin:8px 0'>{_md(tri['label'].title())} near {_md(res.get('origin',{}).get('label', res['query']['location']))} "
             f"<span style='background:{col_};color:#fff;border-radius:10px;padding:2px 10px;font-size:14px'>{tri['tier']} criticality</span></h2>")
    h.append("<div style='color:#555;font-size:13px'><b>Triage:</b> " + " ".join(_md(x) for x in tri["reasons"]) + "</div>")
    if "origin" in res: h.append(f"<div style='color:#555;font-size:13px'><b>Location used:</b> {_md(res['origin']['precision'])}</div>")
    need = res["context"].get("need")
    if need:
        h.append(f"<div style='color:#555;font-size:13px'><b>District context (NFHS-5, {_md(need['district'])}):</b> need for this kind of care is higher than in "
                 f"{need['percentile']:.0%} of districts. {_md('; '.join(need['detail']))}.</div>")
    for n in res["notes"]: h.append(f"<div style='color:#9a6700'><b>Note:</b> {_md(n)}</div>")
    for i, o in enumerate(res["options"], 1):
        h.append(f"<div style='border:1px solid #d0d7de;border-left:6px solid {col_};border-radius:6px;padding:10px;margin:10px 0'>"
                 f"<div style='font-size:17px'><b>{i}. {_md(o['name'])}</b> · {_md(o['city'])} · {o['dist_km']} km · "
                 f"P(offers) {o['p_offers']:.0%}" + (f" · <a href='{html.escape(o['source_url'])}' target='_blank'>source</a>" if o['source_url'] else "") + "</div>"
                 "<div><b>WHY this facility?</b><ul style='margin:4px 0'>" + "".join(f"<li>{_md(x)}</li>" for x in o["why"]) + "</ul></div>"
                 "<div><b>HOW did we decide?</b><ul style='margin:4px 0'>" + "".join(f"<li>{_md(x)}</li>" for x in o["how"]) + "</ul></div></div>")
    if res["leads"]:
        h.append("<div><b>Unverified leads</b> (related evidence only; phone first): " + "; ".join(f"{_md(l['name'])} ({l['dist_km']} km, P {l['p']:.0%})" for l in res["leads"]) + "</div>")
    h.append("</div>")
    return "".join(h)

def as_text(res):
    lines = [f"{res['triage']['label']} | {res['triage']['tier']} | {res.get('origin',{}).get('label','?')}"] + [f"NOTE: {n}" for n in res["notes"]]
    for i, o in enumerate(res["options"], 1):
        lines += [f"\n#{i} {o['name']} ({o['city']}, {o['dist_km']} km)", "WHY:"] + [f"  - {x}" for x in o["why"]] + ["HOW:"] + [f"  - {x}" for x in o["how"]]
    return "\n".join(lines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Run a query
# MAGIC Change the widgets at the top of the notebook and re-run this cell. Free-text service is fine, e.g. *"my wife is in labour and bleeding"*,
# MAGIC *"dialysis 3 times a week"*, *"chemotherapy for my father"*.

# COMMAND ----------

probe_llm()
dbutils.widgets.text("service", "emergency c-section", "Service (free text)")
dbutils.widgets.text("location", "Lucknow", "Location (city, district, PIN, village, or lat,lon)")
result = recommend(dbutils.widgets.get("service"), dbutils.widgets.get("location"))
displayHTML(render(result))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Scenario tests and validation
# MAGIC One query per tier plus the edge cases. Check the printed summary. Then, if a registry is uploaded, **validation**: for facilities that an official
# MAGIC registry confirms (independent ground truth for *presence*), what share does the text-only Claim Court already rate ≥ 50% (**recall**)?
# MAGIC And how do P-values for confirmed versus unconfirmed facilities compare? Precision can't be measured honestly from presence-only lists, and we say so.

# COMMAND ----------

SCENARIOS = [("trauma", "Pune"), ("my wife is in labour and bleeding", "Lucknow"), ("NICU", "226001"), ("dialysis", "Jaipur"),
             ("emergency dialysis", "Jaipur"), ("high-risk delivery", "Nagpur"), ("cardiac surgery", "Ahmedabad"), ("cancer care", "Patna"),
             ("mri", "Aurangabad"), ("burn care", "26.85, 80.95"), ("acupuncture", "Delhi"), ("dialysis", "Xyzzyville")]
summary = []
for s_, l_ in SCENARIOS:
    try:
        r_ = recommend(s_, l_)
        summary.append(dict(service=s_, location=l_, tier=r_["triage"]["tier"], resolved=r_["triage"]["label"], options=len(r_["options"]),
                            top=r_["options"][0]["name"] if r_["options"] else "-", notes=" | ".join(r_["notes"])[:150], secs=r_.get("runtime_s")))
    except Exception as e:
        summary.append(dict(service=s_, location=l_, tier="ERROR", resolved="", options=0, top="", notes=str(e)[:150], secs=None))
display(pd.DataFrame(summary))

# COMMAND ----------

if REG is not None:
    rows = []
    for key, spec in SERVICE_CATALOG.items():
        for _, r_ in fac.iterrows():
            regs = registry_matches(r_)
            if not regs: continue
            conf = any(registry_confirms(m, spec) for m in regs)
            spec_txt = re.search("|".join(spec["strong"]), r_["all_text"], re.I)
            if not conf and not spec_txt: continue
            p_text = claim_court(r_, spec, use_registry=False)["p"]          # text-only probability (registry witness switched off)
            rows.append(dict(service=key, confirmed=conf, p_text=p_text))
    v = pd.DataFrame(rows)
    if len(v):
        out = v.groupby(["service", "confirmed"])["p_text"].agg(["count", "mean", lambda x: (x >= 0.5).mean()]).rename(columns={"<lambda_0>": "share_p≥0.5"})
        display(out.reset_index())
        print("Read: for confirmed=True rows, share_p≥0.5 is the text model's recall against the official list.")
else:
    print("No registry uploaded → validation skipped. Upload registry_facilities to measure recall against official lists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. (Optional) Save results for a Dashboard or Genie space
# MAGIC Writes the latest query's options, with their WHY/HOW text, to `workspace.default.facility_recommendations`.

# COMMAND ----------

def save_result(res):
    if not res["options"]: print("Nothing to save."); return
    df = pd.DataFrame([dict(service=res["triage"]["label"], tier=res["triage"]["tier"], location=res["query"]["location"], rank=i + 1, name=o["name"],
                            city=o["city"], dist_km=o["dist_km"], p_offers=o["p_offers"], p_location=o["p_loc"], quality=o["quality"],
                            registry=", ".join(o["registry"]), why="\n".join(o["why"]), how="\n".join(o["how"]), source_url=o["source_url"] or "")
                       for i, o in enumerate(res["options"])])
    spark.createDataFrame(df).write.mode("append").option("mergeSchema", "true").saveAsTable(f"{OUTPUT_SCHEMA}.facility_recommendations")
    print(f"saved {len(df)} rows")
save_result(result)
