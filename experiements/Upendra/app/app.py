"""CareCompass dashboard. Runs as a Databricks App (reads gold tables via a SQL warehouse)
or locally / on any Streamlit host (reads the parquet snapshot in ./data)."""
import os, html, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import engine
from engine import SERVICES, VISIBLE_SERVICES

st.set_page_config(page_title="CareCompass: find the right care", page_icon="🧭", layout="wide")

INK, PAPER, MIST, TEAL, RED, AMBER, GREEN, GREY = "#1E2B33", "#FFFFFF", "#EEF3F2", "#0B6E69", "#B3261E", "#A15C00", "#2E7D32", "#5B6B73"
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Noto+Sans+Devanagari:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, .stSelectbox, button {{
  font-family: 'Atkinson Hyperlegible', 'Noto Sans Devanagari', system-ui, sans-serif !important; color: {INK}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1180px; }}
h1.cc-title {{ font-size: 2.1rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }}
p.cc-sub {{ font-size: 1.05rem; color: {GREY}; margin: .2rem 0 1rem; max-width: 62ch; }}
.cc-triage {{ border-left: 6px solid {TEAL}; background: {MIST}; padding: .7rem 1rem; margin: .6rem 0 1rem; font-size: 1rem; }}
.cc-triage.high {{ border-color: {RED}; background: #FBEDEC; }}
.cc-alert {{ background: {RED}; color: white; padding: 1rem 1.2rem; font-size: 1.45rem; font-weight: 700; border-radius: 6px; }}
.cc-alert span {{ display: block; font-size: 1rem; font-weight: 400; margin-top: .2rem; }}
.cc-plan {{ border: 2px solid {INK}; border-radius: 6px; padding: 1.1rem 1.3rem; margin-top: .9rem; }}
.cc-plan .lead {{ color: {GREY}; font-size: .95rem; }}
.cc-plan h2 {{ margin: .1rem 0 .3rem; font-size: 1.7rem; }}
.cc-plan .meta {{ font-size: 1.02rem; }}
.cc-dots {{ display: flex; gap: 6px; margin: .8rem 0 .3rem; }}
.cc-dot {{ width: 22px; height: 22px; border-radius: 50%; border: 2px solid {TEAL}; }}
.cc-dot.on {{ background: {TEAL}; }}
.cc-odds {{ font-size: 1.05rem; }}
.cc-card {{ border: 1px solid #C9D3D6; border-radius: 6px; padding: .9rem 1rem; height: 100%; }}
.cc-card h3 {{ font-size: 1.2rem; margin: 0 0 .2rem; }}
.cc-like {{ font-weight: 700; }}
.cc-badge {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: .82rem; margin: .25rem 0; }}
.cc-badge.ok {{ background: #E3F1E4; color: {GREEN}; }} .cc-badge.maybe {{ background: #FFF3DC; color: {AMBER}; }}
.cc-badge.warn {{ background: #FBEDEC; color: {RED}; }} .cc-badge.none {{ background: {MIST}; color: {GREY}; }}
.cc-quote {{ font-size: .93rem; margin: .25rem 0; }}
.cc-quote small {{ color: {GREY}; display: block; }}
.cc-unsure {{ font-size: .9rem; color: {AMBER}; margin: .15rem 0; }}
mark {{ background: #D7EFEC; padding: 0 2px; }}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
TABLES = ["facilities", "facility_service_scores", "service_evidence", "care_gaps", "triage_eval", "coverage_summary"]
DATA_DIR = Path(__file__).parent / "data"

def _read_databricks():
    from databricks import sql
    from databricks.sdk.core import Config
    cfg = Config()
    prefix = os.getenv("CARECOMPASS_TABLES", "workspace.care_compass")
    wh = os.environ["DATABRICKS_WAREHOUSE_ID"]
    out = {}
    with sql.connect(server_hostname=cfg.host, http_path=f"/sql/1.0/warehouses/{wh}",
                     credentials_provider=lambda: cfg.authenticate) as conn:
        with conn.cursor() as cur:
            for t in TABLES:
                try:
                    cur.execute(f"SELECT * FROM {prefix}.{t}")
                    out[t] = cur.fetchall_arrow().to_pandas()
                except Exception:
                    out[t] = pd.DataFrame()
    return out

@st.cache_resource(show_spinner="Loading facility data…")
def load_data():
    src = os.getenv("CARECOMPASS_SOURCE", "auto")
    if src == "local" or (src == "auto" and (DATA_DIR / "facilities.parquet").exists()):
        d = {t: (pd.read_parquet(DATA_DIR / f"{t}.parquet") if (DATA_DIR / f"{t}.parquet").exists() else pd.DataFrame()) for t in TABLES}
        d["_source"] = "local snapshot"
    else:
        d = _read_databricks(); d["_source"] = "Databricks gold tables"
    f = d["facilities"]
    for c in ["lat", "lon", "sigma_km", "trust", "documentation", "reg_match_p", "reg_factor", "capacity", "n_doctors", "year_est", "dup_size"]:
        if c in f: f[c] = pd.to_numeric(f[c], errors="coerce")
    for c in ["geo_ok", "flag_geo_state", "flag_geo_city", "flag_pin_state", "flag_merge", "flag_overclaim"]:
        if c in f: f[c] = f[c].astype(str).str.lower().isin(["true", "1"])
    f["reg_factor"] = f.get("reg_factor", pd.Series(0.8, index=f.index)).fillna(0.8)
    s = d["facility_service_scores"]
    for c in ["evidence", "p_capable", "p_lo", "p_hi", "graph_score", "plaus_pct", "n_fields_core", "n_neg_core"]:
        if c in s: s[c] = pd.to_numeric(s[c], errors="coerce")
    for c in ["flag_out_of_place", "reg_specialty_match"]:
        if c in s: s[c] = s[c].astype(str).str.lower().isin(["true", "1"])
    e = d["service_evidence"]
    if len(e): e["negated"] = e["negated"].astype(str).str.lower().isin(["true", "1"])
    return d

try:
    D = load_data()
except Exception as ex:
    st.error(f"Couldn't load data: {ex}. Run the notebook first, then either set DATABRICKS_WAREHOUSE_ID "
             "(Databricks App) or copy the parquet export into app/data/.")
    st.stop()
FAC, SC, EV = D["facilities"], D["facility_service_scores"], D["service_evidence"]

# ------------------------------------------------------------------------------------------------
# Small render helpers
# ------------------------------------------------------------------------------------------------
esc = lambda s: html.escape(str(s)) if s is not None else ""

def html_block(s):
    """Flatten multi-line HTML: Streamlit's markdown ends an HTML block at the first blank line."""
    import re as _re
    return _re.sub(r"\n\s*", " ", s.strip())

def mark(snip, term):
    s = esc(snip)
    t = esc(term)
    i = s.lower().find(t.lower()) if t else -1
    return s if i < 0 else s[:i] + f"<mark>{s[i:i+len(t)]}</mark>" + s[i+len(t):]

def quotes_html(fid, service, n=3):
    out = []
    for q in engine.quotes(EV, fid, service, n):
        note = " (mentioned as planned/referred/unavailable)" if q["negated"] else (" (related)" if q["supporting"] else "")
        out.append(f"<div class='cc-quote'>“{mark(q['snippet'], q['matched'])}”<small>from the {esc(q['field'])}{note}</small></div>")
    return "".join(out) or "<div class='cc-quote'>No direct quote available.</div>"

def badge(row):
    kind, text = engine.registry_line(row)
    return f"<span class='cc-badge {kind}'>{esc(text)}</span>"

def phone_of(row):
    ph = engine.as_list(row.get("phone"))
    return ph[0] if ph else None

def address_of(row):
    return ", ".join(str(x) for x in [row.get("address"), row.get("city"), row.get("state"), row.get("postcode")]
                     if isinstance(x, str) and x and x != "None")

def dots(p):
    k = engine.in_ten(p)
    return "<div class='cc-dots' aria-hidden='true'>" + "".join(f"<div class='cc-dot{' on' if i < k else ''}'></div>" for i in range(10)) + "</div>"

def map_fig(points, loc, height=380):
    pts = points.copy()
    pts["kind"] = pts.get("kind", "option")
    you = pd.DataFrame([dict(lat=loc["lat"], lon=loc["lon"], name="Patient location", kind="patient")])
    allp = pd.concat([pts[["lat", "lon", "name", "kind"]], you], ignore_index=True)
    colors = {"patient": INK, "first choice": RED, "option": TEAL, "backup": AMBER, "transfer": GREY}
    kw = dict(lat="lat", lon="lon", hover_name="name", color="kind", color_discrete_map=colors, zoom=8, height=height)
    try:
        fig = px.scatter_map(allp, **kw); fig.update_layout(map_style="open-street-map")
    except Exception:
        fig = px.scatter_mapbox(allp, **kw); fig.update_layout(mapbox_style="open-street-map")
    fig.update_traces(marker=dict(size=13))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=1.02, x=0, title=None))
    return fig

# ------------------------------------------------------------------------------------------------
# Header + tabs
# ------------------------------------------------------------------------------------------------
st.markdown("<h1 class='cc-title'>CareCompass</h1><p class='cc-sub'>Where should this patient go? Describe the need in any "
            "language. Every answer shows the evidence behind it and how sure we are.</p>", unsafe_allow_html=True)
tab_find, tab_how, tab_data = st.tabs(["Find care", "How it decides", "Data health"])

EXAMPLES = [("accident hua hai, behosh hai", "Lucknow"), ("bahut khoon beh raha hai delivery ke baad", "Nashik, Maharashtra"),
            ("saanp ne kaat liya", "Gaya"), ("dialysis twice a week", "Gaya, Bihar"), ("MRI for knee next week", "Jaipur")]

def _use_example(i):
    st.session_state["need"], st.session_state["place"] = EXAMPLES[i]
    st.session_state["run"] = True

with tab_find:
    st.caption("Try an example")
    ex_cols = st.columns(len(EXAMPLES))
    for i, (q, l) in enumerate(EXAMPLES):
        ex_cols[i].button(f"{q} · {l}", key=f"ex{i}", on_click=_use_example, args=(i,), use_container_width=True)
    with st.form("query"):
        c1, c2, c3 = st.columns([3, 1.4, 1.4])
        need = c1.text_area("What does the patient need?", key="need", height=80,
                            placeholder="e.g. “accident hua hai, behosh hai” or “dialysis twice a week”")
        place = c2.text_input("Where is the patient?", key="place", placeholder="City, PIN code, or lat, lon")
        svc_choice = c3.selectbox("Service (optional)", ["Detect from description"] + [SERVICES[s]["label"] for s in VISIBLE_SERVICES])
        submitted = st.form_submit_button("Find care", type="primary")
    if submitted:
        st.session_state["run"] = True

    if st.session_state.get("run") and st.session_state.get("need", "").strip() and st.session_state.get("place", "").strip():
        need, place = st.session_state["need"], st.session_state["place"]
        override = None if svc_choice == "Detect from description" else next(s for s in VISIBLE_SERVICES if SERVICES[s]["label"] == svc_choice)
        tri = engine.triage(need, override)
        loc = engine.resolve_location(place, FAC)
        if tri["service"] is None:
            st.warning("We couldn't tell which service is needed. Pick one from “Service (optional)” and run again.")
            st.stop()
        if loc is None:
            st.warning(f"We couldn't place “{place}”. Try a nearby city, a 6-digit PIN code, or coordinates like 25.6, 85.1.")
            st.stop()
        high = tri["severity"] == "high"
        why = "; ".join(tri["reasons"]) or "no danger signs mentioned"
        if high:
            svc = tri["emergency_service"]
            extra = (f" The patient may also need {SERVICES[tri['service']]['label'].lower()}; stabilising comes first."
                     if tri["escalated"] else "")
            st.markdown(f"<div class='cc-triage high'><b>Treated as an emergency</b> ({SERVICES[svc]['label']}). "
                        f"Why: {esc(why)}. Chance this is an emergency: {tri['p_high']:.0%}; we treat anything above "
                        f"{tri['threshold']:.0%} as one, because missing an emergency is far worse than over-calling.{esc(extra)}</div>",
                        unsafe_allow_html=True)
        else:
            svc = tri["service"]
            st.markdown(f"<div class='cc-triage'><b>Planned care:</b> {SERVICES[svc]['label']}. Why: {esc(why)}. "
                        "If the patient gets worse (unconscious, heavy bleeding, can't breathe), describe that and run again.</div>",
                        unsafe_allow_html=True)
        st.caption(f"Patient location: {loc['label']} — {loc['precision']}"
                   + (f" from {loc['n']} facility records" if loc.get("n") else "") + ".")

        # ---------------------------------- HIGH severity ----------------------------------
        if high:
            line102 = "<span>Pregnancy or newborn emergency: 102 for free transport.</span>" if svc in ("emergency_csection", "nicu") else \
                      "<span>Keep the patient still and warm while you wait. Share the hospital below with the ambulance crew.</span>"
            st.markdown(f"<div class='cc-alert'>Call 108 for an ambulance now.{line102}</div>", unsafe_allow_html=True)
            with st.spinner("Simulating 1,000 possible trips…"):
                out = engine.simulate_emergency(svc, loc, FAC, SC, n_worlds=1000, seed=1)
            if out is None or out["plans"].empty:
                st.error("No facilities with usable locations were found nearby. Ask the ambulance service (108) for the nearest emergency department.")
                st.stop()
            plans, win = out["plans"], out["window"]
            b = plans.iloc[0]
            role_txt = ("Records suggest it can treat this." if b.role == "Can likely treat" else
                        f"Records suggest it can stabilise the patient; a transfer is likely needed"
                        + (f", most often to {esc(b.typical_transfer)}." if b.typical_transfer else ".") if b.role.startswith("Can likely stabilise")
                        else "Its capability is uncertain, but it's the best bet given travel time. Call ahead.")
            ph = phone_of(b)
            st.markdown(html_block(f"""
<div class='cc-plan'>
  <div class='lead'>Go first to</div>
  <h2>{esc(b['name'])}</h2>
  <div class='meta'>About {b.minutes_median:.0f} min by road (usually {b.minutes_p10:.0f}–{b.minutes_p90:.0f} min){' · ☎ ' + esc(ph) if ph else ''}</div>
  <div class='meta' style='color:{GREY}'>{esc(address_of(b))}</div>
  <div style='margin-top:.5rem'>{role_txt}</div>
  {dots(b.p_first_in_window)}
  <div class='cc-odds'>In about <b>{engine.in_ten(b.p_first_in_window)} of 10</b> simulated trips, the patient reaches someone who can help within {win} minutes.
  Definitive care within {win} minutes: about {engine.in_ten(b.p_definitive_in_window)} in 10.</div>
</div>"""), unsafe_allow_html=True)
            a, c = st.columns(2)
            with a:
                st.markdown("**Why this one**")
                st.markdown(quotes_html(b.facility_id, svc) + badge(b), unsafe_allow_html=True)
                if b.role.startswith("Can likely stabilise"):
                    st.markdown(quotes_html(b.facility_id, "emergency_24x7", 2), unsafe_allow_html=True)
            with c:
                st.markdown("**What we're not sure about**")
                cav = engine.caveats(b) or ["No data-quality problems found in this record."]
                st.markdown("".join(f"<div class='cc-unsure'>{esc(x)}</div>" for x in cav), unsafe_allow_html=True)
                if b.win_pct < 0.3:
                    st.markdown(f"<div class='cc-unsure'>Close call: several nearby options score almost the same.</div>", unsafe_allow_html=True)
            if len(plans) > 1:
                bk = plans.iloc[1]; ph2 = phone_of(bk)
                st.markdown(f"**Backup:** {bk['name']}, about {bk.minutes_median:.0f} min "
                            f"({engine.likelihood_words(bk.p_def)} to treat){' · ☎ ' + ph2 if ph2 else ''}")
            pts = plans.head(4).assign(kind=["first choice"] + ["backup"] * (min(4, len(plans)) - 1))
            st.plotly_chart(map_fig(pts, loc), use_container_width=True)
            with st.expander("How we decided"):
                st.write(f"We simulated {out['n_worlds']:,} possible trips. In each one, every nearby facility either can or can't treat "
                         "this (drawn from how strongly its records support it), may or may not be operating, may sit a little away from its listed "
                         "location, and road speed varies. A facility that can only stabilise slows the clock while a transfer is arranged. "
                         "We pick the first stop with the best time-adjusted outcome on average.")
                show = plans.head(8)[["name", "role", "minutes_median", "p_def", "p_stab", "p_first_in_window", "win_pct", "typical_transfer"]].copy()
                show.columns = ["Facility", "Role", "Minutes (median)", "Can treat (prob.)", "Can stabilise (prob.)",
                                f"Help within {win} min", "Best choice in % of trips", "Usual transfer"]
                st.dataframe(show.style.format({"Minutes (median)": "{:.0f}", "Can treat (prob.)": "{:.0%}", "Can stabilise (prob.)": "{:.0%}",
                                                f"Help within {win} min": "{:.0%}", "Best choice in % of trips": "{:.0%}"}), hide_index=True)
                fig = go.Figure()
                fig.add_histogram(x=out["t_first_best"], name="someone who can help", opacity=0.65, marker_color=TEAL)
                fig.add_histogram(x=out["t_defin_best"], name="definitive care", opacity=0.55, marker_color=AMBER)
                fig.add_vline(x=win, line_dash="dash", line_color=RED, annotation_text=f"{win}-minute window")
                fig.update_layout(barmode="overlay", height=300, margin=dict(t=30, l=0, r=0, b=0), xaxis_title="minutes", yaxis_title="simulated trips")
                st.plotly_chart(fig, use_container_width=True)
                if out["baseline"]:
                    bl = out["baseline"]
                    st.write(f"For comparison, simply going to the nearest facility whose record mentions this service "
                             f"({bl['name']}) reaches help within {win} min in about {engine.in_ten(bl['p_first_in_window'])} of 10 trips "
                             f"(time-adjusted outcome {bl['exp_benefit']:.2f} vs {b.exp_benefit:.2f} for our choice).")

        # ---------------------------------- LOW severity ----------------------------------
        else:
            out = engine.rank_low(svc, loc, FAC, SC, k=3)
            res = out["results"]
            if res.empty:
                st.warning(f"No facility with any evidence for {SERVICES[svc]['label'].lower()} within {out['radius']:.0f} km. Try a larger nearby city.")
                st.stop()
            st.markdown(f"#### Best options for {SERVICES[svc]['label']} near {esc(loc['label'])}")
            cols = st.columns(len(res))
            for col, (_, r) in zip(cols, res.iterrows()):
                stab = ("stable choice" if r.p_top1 >= 0.7 else "likely top choice" if r.p_top1 >= 0.4 else "close call") if _ == 0 else None
                ph = phone_of(r)
                cav = engine.caveats(r)[:3]
                col.markdown(html_block(f"""
<div class='cc-card'>
  <h3>{esc(r['name'])}</h3>
  <div style='color:{GREY};font-size:.9rem'>{esc(address_of(r))}</div>
  <div style='margin:.4rem 0'><span class='cc-like'>{engine.likelihood_words(r.p_capable).capitalize()}</span> to offer it
     ({r.p_lo:.0%}–{r.p_hi:.0%}) · about {r.road_km:.0f} km{' · ☎ ' + esc(ph) if ph else ''}</div>
  {badge(r)}
  {quotes_html(r.facility_id, svc, 2)}
  {''.join(f"<div class='cc-unsure'>{esc(x)}</div>" for x in cav)}
  {f"<div style='margin-top:.4rem;color:{GREY};font-size:.88rem'>Ranking: {stab} ({r.p_top1:.0%} of re-weightings)</div>" if stab else ''}
</div>"""), unsafe_allow_html=True)
            ns = out.get("nearest_strong")
            if res.tier.iloc[0] != "Strong" and ns is not None and len(ns):
                st.info(f"The top option's evidence is {res.tier.iloc[0].lower()}. The nearest facility with strong evidence is "
                        f"{ns.iloc[0]['name']} (about {ns.iloc[0].road_km:.0f} km).")
            a, c = st.columns([1.15, 1])
            cand = out["candidates"].copy()
            cand["pick"] = np.where(cand.facility_id.isin(res.facility_id), "top 3", "other")
            fig = px.scatter(cand, x="road_km", y="p_capable", color="pick", hover_name="name", height=380,
                             color_discrete_map={"top 3": RED, "other": "#9FB3B8"},
                             labels={"road_km": "km by road", "p_capable": "how likely it offers the service"})
            p = out["pareto"].sort_values("road_km")
            fig.add_scatter(x=p.road_km, y=p.p_capable, mode="lines", line=dict(color=INK, dash="dot"), name="best trade-offs")
            fig.update_layout(yaxis_tickformat=".0%", margin=dict(t=30, l=0, r=0, b=0), legend=dict(orientation="h", y=1.08, title=None),
                              title=dict(text="Closer vs. more certain", font=dict(size=15)))
            a.plotly_chart(fig, use_container_width=True)
            c.plotly_chart(map_fig(res.assign(kind="option"), loc), use_container_width=True)
            with st.expander("How we decided"):
                st.write(f"We looked at {len(cand)} facilities within {out['radius']:.0f} km whose records mention this service. Each is "
                         "scored on how likely it really offers the service (quotes in its record, what similar facilities offer, a text model, "
                         "government-scheme listing), how consistent its record is, and distance. The scores are combined so a facility has to do "
                         "reasonably on all of them. We re-ran the ranking 300 times with different weightings to check the order holds. "
                         "The dotted line shows the best trade-offs: no other facility is both closer and more likely to offer it.")
                t = res[["name", "p_capable", "p_lo", "p_hi", "trust", "access", "reg_factor", "score", "p_top1"]].copy()
                t.columns = ["Facility", "Likely offers it", "low", "high", "Record trust", "Distance score", "Govt list", "Overall", "Rank stability"]
                st.dataframe(t.style.format({c: "{:.0%}" for c in ["Likely offers it", "low", "high", "Rank stability"]} |
                                            {c: "{:.2f}" for c in ["Record trust", "Distance score", "Govt list", "Overall"]}), hide_index=True)
        st.caption("CareCompass supports decisions; it doesn't replace medical judgement. Records come from public web listings and may be "
                   "out of date. Always call ahead.")

with tab_how:
    st.markdown("""
#### From a description to a recommendation
1. **Read the need.** We look for danger signs in English, Hindi and Hinglish (*behosh*, *bahut khoon*, *saans nahi*, *saanp ne kaata*…) and combine them with what we know about the service. Missing an emergency counts 6× worse than over-calling one, so anything above a 14% chance is treated as an emergency.
2. **Planned care** (dialysis, MRI, chemotherapy…): find the facility most likely to really offer it, at a reasonable distance, with a consistent record. Likelihood combines quotes from the facility's own listing, what similar facilities offer, a text model, and government-scheme listings.
3. **Emergencies** (accidents, bleeding after delivery, snakebite, stroke…): simulate 1,000 possible trips, allowing for uncertain capability, location and road speed, and pick the first stop that gets the patient to competent hands soonest. A nearby hospital that can stabilise often beats a famous one further away.
4. **Check against government lists.** PM-JAY and state-scheme empanelment raises confidence; appearing on a de-empanelled or suspended list lowers it, more so if recent.
5. **Say it honestly.** We use words like *likely* and *possible* with ranges, quote the records, and list what we couldn't confirm.
""")
    st.markdown("#### Clinical time windows we assume")
    st.dataframe(pd.DataFrame([{"Need": v["label"], "Severity": v["severity"], "Care should start within (min)": v["clock"]["window"]}
                               for k, v in SERVICES.items() if v["severity"] == "high" and not v.get("hidden")]), hide_index=True)

with tab_data:
    n = len(FAC)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Facilities", f"{n:,}")
    k2.metric("Location from the record", f"{(FAC.coord_source == 'record').mean():.0%}")
    flags = FAC[["flag_geo_state", "flag_geo_city", "flag_pin_state", "flag_merge", "flag_overclaim"]].any(axis=1)
    k3.metric("Records with a data-quality flag", f"{flags.mean():.0%}")
    linked = FAC.reg_status.astype(str).str.startswith("empanelled").sum()
    k4.metric("Matched to a government list", f"{linked:,}" if linked else "not loaded")
    cov = D.get("coverage_summary", pd.DataFrame())
    if len(cov):
        m = cov.melt(id_vars=["service", "window_min"], var_name="care", value_name="pct")
        m["care"] = m.care.map({"cities_with_stabilisation_in_window": "can stabilise", "cities_with_definitive_in_window": "can treat"})
        fig = px.bar(m, x="service", y="pct", color="care", barmode="group", height=380,
                     color_discrete_map={"can stabilise": "#9FB3B8", "can treat": TEAL},
                     labels={"pct": "% of cities within the time window", "service": ""},
                     title="How many places reach emergency care in time?")
        st.plotly_chart(fig, use_container_width=True)
    gaps = D.get("care_gaps", pd.DataFrame())
    if len(gaps):
        opts = [c for c in gaps.columns if c.startswith("min_to_") and c != "min_to_stabilise"]
        pick = st.selectbox("Show drive time to", opts, format_func=lambda c: SERVICES.get(c[7:], {"label": c})["label"])
        g = gaps.copy(); g[pick] = pd.to_numeric(g[pick], errors="coerce")
        fig = px.scatter_geo(g, lat="lat", lon="lon", color=g[pick].clip(upper=240), hover_name="city_norm",
                             color_continuous_scale="RdYlGn_r", height=560, labels={"color": "minutes"})
        fig.update_geos(fitbounds="locations", showcountries=True); fig.update_layout(margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    te = D.get("triage_eval", pd.DataFrame())
    if len(te):
        missed = ((te.expected == "high") & (te.predicted == "low")).sum()
        st.markdown(f"**Severity check:** {missed} missed emergencies out of {(te.expected == 'high').sum()} in our hand-written test set.")
        st.dataframe(te[["query", "expected", "predicted", "why"]], hide_index=True)
    st.caption(f"Data source: {D['_source']}.")
