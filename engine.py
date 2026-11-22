"""
CareCompass engine: the query-time logic shared by the Databricks notebook and the dashboard.

Flow (mirrors the team's design):
  query -> triage() decides severity (hard-coded clinical priors + red-flag words, cost-sensitive)
        -> LOW  severity: rank_low()            multi-criteria ranking, uncertainty ranges, Pareto view
        -> HIGH severity: simulate_emergency()  Monte Carlo over capability, location and travel time,
                                                 with stabilise-then-transfer outcomes
        -> registry verification (PM-JAY / state / de-empanelled) is precomputed per facility and enters
           both branches as `reg_factor`
        -> explanations use hedged wording (never states a claim as fact)
"""
import re, json, math, difflib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# 1. Service ontology. Severity and clocks are hard-coded clinical priors (stated as assumptions).
#    clock.window = minutes within which care should start; clock.tau = benefit e-folding time.
# ---------------------------------------------------------------------------------------------
SERVICES = {
    # ---------------- high severity ----------------
    "trauma": dict(
        label="Accident / trauma care", severity="high", clock=dict(window=60, tau=60),
        synonyms=["trauma", "accident", "road accident", "polytrauma", "injury", "injured", "fracture", "crash",
                  "takkar", "ghayal", "chot"],
        core=[r"\btrauma (centre|center|care|unit|surg|icu|ward)", r"\bpolytrauma", r"\blevel[- ](i|1|ii|2) trauma",
              r"\baccident (and|&) emergency", r"\bemergency (and|&) trauma", r"\bcasualty (ward|department)\b"],
        support=[r"\borthop(a)?edic", r"\bneurosurg", r"\bct[- ]?scan", r"\bblood bank", r"\bicu\b", r"\bemergency\b",
                 r"\b24\s*[x/×]\s*7\b", r"\bambulance"],
        prereq={"emergency department": [r"\bemergency\b", r"\bcasualty"],
                "imaging (CT/X-ray)": [r"\bct\b", r"\bx[- ]?ray", r"\bradiolog"]},
        pmjay=r"trauma|orthop|emergency|general surg"),
    "emergency_csection": dict(
        label="Emergency C-section / high-risk delivery", severity="high", clock=dict(window=120, tau=90),
        synonyms=["c-section", "c section", "caesarean", "cesarean", "high-risk delivery", "high risk delivery",
                  "emergency delivery", "obstructed labour", "bleeding after delivery", "lscs", "prasav", "delivery"],
        core=[r"\bc[- ]?sections?\b", r"\bc(a)?esarean\b", r"\blscs\b", r"\bemergency obstetric", r"\bcemonc\b",
              r"\bhigh[- ]risk (pregnanc|deliver|obstet|labou?r)", r"\bcomprehensive emergency obstetric",
              r"\bmaternal[- ]fetal medicine\b"],
        support=[r"\bobstetric", r"\bgyn(a)?ecolog", r"\bmaternity\b", r"\blabou?r (room|ward)\b", r"\bdeliver(y|ies)\b",
                 r"\bnicu\b", r"\bblood bank\b", r"\b24\s*[x/×]\s*7\b"],
        prereq={"operating theatre": [r"\boperation theat", r"\boperating (room|theat)", r"\b(ot|o\.t\.)\b", r"\bsurg"],
                "anaesthesia": [r"\ban(a)?esthe"],
                "blood availability": [r"\bblood (bank|storage|transfusion)"]},
        pmjay=r"obstet|gyn|o\s*&\s*g|maternity"),
    "nicu": dict(
        label="Newborn intensive care (NICU)", severity="high", clock=dict(window=120, tau=120),
        synonyms=["nicu", "neonatal intensive care", "newborn icu", "sncu", "premature", "newborn", "navjaat"],
        core=[r"\bnicu\b", r"\bneonatal (intensive|icu|care unit)", r"\bsncu\b", r"\bspecial newborn care",
              r"\bneonatal ventilat", r"\blevel (ii|iii|2|3) (nicu|neonatal)"],
        support=[r"\bneonatolog", r"\bp(a)?ediatric", r"\bincubator", r"\bradiant warmer", r"\bphototherapy",
                 r"\bbubble cpap\b|\bcpap\b", r"\bpicu\b"],
        prereq={"neonatal/paediatric specialist": [r"\bneonatolog", r"\bp(a)?ediatric"]},
        pmjay=r"neo.?nat|p(a)?ediatric"),
    "burn_care": dict(
        label="Burn care", severity="high", clock=dict(window=120, tau=120),
        synonyms=["burn", "burns", "burn care", "burn unit", "jal gaya", "jhulas"],
        core=[r"\bburns? (unit|ward|centre|center|care|icu|management|treatment)\b", r"\bburns? and plastic",
              r"\bskin graft", r"\bplastic surg[a-z]* (and|&) burns?", r"\bburns? surg"],
        support=[r"\bplastic surg", r"\breconstructive surg", r"\bicu\b", r"\bwound care\b"],
        prereq={"surgical capability": [r"\bsurg", r"\boperation theat"]},
        pmjay=r"burn|plastic"),
    "heart_attack": dict(
        label="Heart attack (cath lab / clot-busting)", severity="high", clock=dict(window=90, tau=90),
        synonyms=["heart attack", "chest pain", "cardiac arrest", "dil ka daura", "seene mein dard", "seene me dard", "mi", "stemi"],
        core=[r"\bcath(eteri[sz]ation)? lab", r"\bangioplast", r"\bprimary pci\b|\bpci\b", r"\bthromboly",
              r"\bcoronary angiograph", r"\bstent"],
        support=[r"\bcardiolog", r"\biccu\b|\bccu\b|\bcardiac care unit", r"\becg\b|\bekg\b", r"\becho(cardiograph)?"],
        prereq={"cardiac ICU": [r"\biccu\b", r"\bccu\b", r"\bicu\b", r"\bintensive care"]},
        pmjay=r"cardiol"),
    "stroke": dict(
        label="Stroke care", severity="high", clock=dict(window=270, tau=180),
        synonyms=["stroke", "paralysis", "lakwa", "face drooping", "slurred speech", "brain attack", "muh tedha"],
        core=[r"\bstroke (unit|care|centre|center|management|thromboly)", r"\bthromboly[a-z]* .*stroke",
              r"\bmechanical thrombectomy", r"\bneuro[- ]?intervention"],
        support=[r"\bneurolog", r"\bneurosurg", r"\bct[- ]?scan", r"\bmri\b"],
        prereq={"brain imaging (CT/MRI)": [r"\bct\b", r"\bmri\b"]},
        pmjay=r"neuro"),
    "snakebite": dict(
        label="Snakebite (anti-venom)", severity="high", clock=dict(window=120, tau=120),
        synonyms=["snakebite", "snake bite", "saanp", "saap", "sanp", "anti venom", "antivenom"],
        core=[r"\banti[- ]?(snake )?venom", r"\basv\b", r"\bsnake ?bite"],
        support=[r"\bventilator", r"\bicu\b", r"\bemergency\b", r"\btoxicolog|\bpoison"],
        prereq={"ventilator/ICU": [r"\bventilat", r"\bicu\b"]},
        pmjay=r"general medicine|emergency"),
    "icu": dict(
        label="ICU / critical care", severity="high", clock=dict(window=60, tau=60),
        synonyms=["icu", "intensive care", "critical care", "ventilator", "unconscious", "poisoning"],
        core=[r"\bicu\b", r"\bintensive care", r"\bcritical care", r"\bmicu\b|\bsicu\b", r"\bventilator"],
        support=[r"\bmonitor", r"\bdefibrillat", r"\boxygen", r"\bemergency\b"],
        prereq={},
        pmjay=r"general medicine|critical|emergency"),
    # helper used as the "can stabilise" capability in emergencies (hidden from menus)
    "emergency_24x7": dict(
        label="24/7 emergency (stabilisation)", severity="high", hidden=True, clock=dict(window=60, tau=60),
        synonyms=[],
        core=[r"\b24\s*[x/×]\s*7\b.{0,25}\b(emergency|casualty|trauma|hospital|open)", r"\b(emergency|casualty)\b.{0,25}\b24\s*[x/×]\s*7\b",
              r"\bemergency (department|ward|room|services?|care|unit)\b", r"\bcasualty\b",
              r"\baccident (and|&) emergency", r"\bround the clock\b", r"\b24 hours? emergency"],
        support=[r"\bicu\b", r"\bambulance", r"\bblood bank", r"\boxygen"],
        prereq={}, pmjay=r"emergency|general medicine"),
    # ---------------- low severity ----------------
    "dialysis": dict(
        label="Dialysis", severity="low", decay_km=60,
        synonyms=["dialysis", "haemodialysis", "hemodialysis", "kidney failure", "renal failure", "gurda", "kidney"],
        core=[r"\b(ha?emo)?dialy(sis|ses|zer|ser)\b", r"\bperitoneal dialysis\b", r"\bcapd\b", r"\bcrrt\b",
              r"\bdialysis (unit|centre|center|machine|facility)\b", r"\brenal replacement therapy\b"],
        support=[r"\bnephrolog", r"\bkidney\b", r"\brenal\b", r"\breverse osmosis\b", r"\bro (plant|water)\b"],
        prereq={}, pmjay=r"nephro|dialys|urolog"),
    "mri": dict(
        label="MRI scan", severity="low", decay_km=80,
        synonyms=["mri", "magnetic resonance"],
        core=[r"\bmri\b", r"\bmagnetic resonance"],
        support=[r"\bradiolog", r"\bimaging\b", r"\bct scan|\bcomputed tomography", r"\bdiagnostic cent"],
        prereq={}, pmjay=None),
    "ct_scan": dict(
        label="CT scan", severity="low", decay_km=60,
        synonyms=["ct scan", "cat scan", "computed tomography"],
        core=[r"\bct[- ]?scan", r"\bcomputed tomography", r"\b\d{2,3}[- ]slice\b", r"\bmdct\b", r"\bcat scan"],
        support=[r"\bradiolog", r"\bimaging\b", r"\bmri\b"],
        prereq={}, pmjay=None),
    "cardiac_surgery": dict(
        label="Planned heart surgery", severity="low", decay_km=150,
        synonyms=["cardiac surgery", "heart surgery", "bypass", "cabg", "open heart", "valve replacement", "cardiothoracic"],
        core=[r"\bcardiac surg", r"\bcardio[- ]?thoracic", r"\bcabg\b", r"\b(coronary )?bypass surg", r"\bopen[- ]heart",
              r"\bvalve (replacement|repair)", r"\bctvs\b", r"\bheart surg"],
        support=[r"\bcath(eteri[sz]ation)? lab", r"\bangioplast", r"\bcardiolog", r"\bperfusion", r"\becho(cardiograph)?"],
        prereq={"ICU": [r"\bicu\b", r"\bintensive care", r"\biccu\b", r"\bccu\b"],
                "operating theatre": [r"\boperation theat", r"\boperating (room|theat)", r"\bsurg"]},
        pmjay=r"cardio.?thoracic|ctvs|cardiac surg"),
    "cancer_chemo": dict(
        label="Cancer care / chemotherapy", severity="low", decay_km=150,
        synonyms=["chemotherapy", "chemo", "cancer", "oncology", "radiotherapy", "tumour", "tumor"],
        core=[r"\bchemotherap", r"\bmedical oncolog", r"\bradiation oncolog|\bradiotherap", r"\blinac\b|\blinear accelerator",
              r"\boncology (unit|department|centre|center)"],
        support=[r"\boncolog", r"\bcancer\b", r"\btumou?r\b", r"\bpet[- ]?ct\b"],
        prereq={}, pmjay=r"oncolog"),
}
VISIBLE_SERVICES = [k for k, v in SERVICES.items() if not v.get("hidden")]
EMERGENCY_SPEED_KMH = 35
PLANNED_SPEED_KMH = 45
ROAD_FACTOR = 1.3

# ---------------------------------------------------------------------------------------------
# 2. Severity triage: hard-coded priors + red-flag dictionary (English / Hindi / Hinglish),
#    combined in log-odds and thresholded with an asymmetric cost (missing an emergency = 6x worse).
# ---------------------------------------------------------------------------------------------
RED_FLAGS = [
    (r"unconscious|behosh|behos|hosh nahi|not responding|fainted|collapsed|passed out", 3.0, "unconscious / not responding", "icu"),
    (r"heavy bleeding|bleeding a lot|bahut khoon|khoon beh|khoon nahi ruk|blood loss|won'?t stop bleeding", 3.0, "heavy bleeding", None),
    (r"chest pain|seene (me|mein) dard|heart attack|dil ka daura|cardiac arrest", 3.0, "chest pain / heart attack", "heart_attack"),
    (r"can'?t breathe|cannot breathe|saans (nahi|lene)|breathless|difficulty breathing|gasping", 3.0, "breathing difficulty", "icu"),
    (r"snake ?bite|sa+n?p ne ka+t|sa+np|sa+p ne", 3.0, "snakebite", "snakebite"),
    (r"\bburn(ed|t|s)?\b|jal gay|jhulas|aag (lag|se)", 2.5, "burns", "burn_care"),
    (r"accident|crash|hit by|takkar|ghayal|fell from|injur", 2.5, "accident / injury", "trauma"),
    (r"seizure|fits?\b|convulsion|daura pad|mirgi", 2.5, "seizure", "icu"),
    (r"face droop|slurred|lakwa|paralys|one side weak|muh tedha|stroke", 3.0, "stroke signs", "stroke"),
    (r"poison|zeher|zahar|pesticide|keetnashak|overdose", 3.0, "poisoning", "icu"),
    (r"labou?r pain|water broke|pani gir|dard shuru|delivery (pain|now|abhi)|obstructed", 2.0, "labour / pregnancy danger", "emergency_csection"),
    (r"newborn|navjaat|premature|baby not breathing|baby (is )?blue|neela pad", 2.5, "newborn danger sign", "nicu"),
    (r"emergency|urgent|turant|jaldi|right now|abhi", 1.0, "user says urgent", None),
]
PLANNED_CUES = r"appointment|next week|next month|routine|check-?up|scheduled|planned|follow.?up|twice a week|weekly|report|second opinion|cost|kitna kharcha"
PREGNANCY = r"pregnan|delivery|deliver|garbh|prasav|after birth|postpartum|labou?r"
COST_RATIO = 6.0                              # cost(missed emergency) / cost(over-called emergency)
HIGH_THRESHOLD = 1.0 / (1.0 + COST_RATIO)     # Bayes-optimal threshold ≈ 0.14

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def infer_service(text):
    t = (text or "").lower()
    best = None
    for svc in VISIBLE_SERVICES:
        for syn in sorted(SERVICES[svc]["synonyms"], key=len, reverse=True):
            if re.search(rf"\b{re.escape(syn)}\b", t) and (best is None or len(syn) > best[1]):
                best = (svc, len(syn))
    return best[0] if best else None

def triage(text, service=None):
    """Return severity decision with reasons. `service` overrides service detection."""
    t = (text or "").lower()
    svc = service or infer_service(t)
    reasons, score, hints = [], 0.0, []
    for rx, w, label, hint in RED_FLAGS:
        m = re.search(rx, t)
        if m:
            reasons.append(f"“{m.group(0)}” → {label}")
            score += w
            if hint:
                hints.append((w, hint))
    planned = bool(re.search(PLANNED_CUES, t))
    if planned:
        reasons.append("mentions planned/routine care")
    prior = 0.85 if (svc and SERVICES[svc]["severity"] == "high") else (0.08 if svc else 0.25)
    logit = math.log(prior / (1 - prior)) + 1.1 * score - (1.2 if planned else 0.0)
    p_high = _sigmoid(logit)
    severity = "high" if p_high >= HIGH_THRESHOLD else "low"
    # Which clinical need drives an emergency?
    emergency_service = None
    if severity == "high":
        if svc and SERVICES[svc]["severity"] == "high":
            emergency_service = svc
        elif re.search(PREGNANCY, t) and re.search(r"bleed|khoon|fit|seizure|pain|dard|unconscious|behosh", t):
            emergency_service = "emergency_csection"
        elif hints:
            emergency_service = max(hints)[1]
        else:
            emergency_service = "icu"
    if svc is None and emergency_service:
        svc = emergency_service
    return dict(service=svc, severity=severity, p_high=p_high, threshold=HIGH_THRESHOLD, reasons=reasons,
                emergency_service=emergency_service,
                escalated=bool(svc and SERVICES[svc]["severity"] == "low" and severity == "high"))

# ---------------------------------------------------------------------------------------------
# 3. Helpers: geometry, list parsing, wording
# ---------------------------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v]
    if isinstance(v, float) and math.isnan(v):
        return []
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
        except Exception:
            pass
    return [s] if s and s.lower() not in ("none", "nan", "null") else []

def likelihood_words(p):
    if p >= 0.85: return "very likely"
    if p >= 0.65: return "likely"
    if p >= 0.40: return "possible"
    return "unconfirmed"

def in_ten(p):
    return int(round(10 * max(0.0, min(1.0, p))))

# ---------------------------------------------------------------------------------------------
# 4. Location resolution: coordinates, PIN, dataset gazetteer, small built-in gazetteer
# ---------------------------------------------------------------------------------------------
GAZETTEER = {  # approximate city centres used only when the dataset can't place a name
    "delhi": (28.61, 77.21), "new delhi": (28.61, 77.21), "mumbai": (19.08, 72.88), "kolkata": (22.57, 88.36),
    "chennai": (13.08, 80.27), "bengaluru": (12.97, 77.59), "bangalore": (12.97, 77.59), "hyderabad": (17.39, 78.49),
    "ahmedabad": (23.02, 72.57), "pune": (18.52, 73.86), "jaipur": (26.91, 75.79), "lucknow": (26.85, 80.95),
    "kanpur": (26.45, 80.33), "patna": (25.59, 85.14), "gaya": (24.79, 85.00), "muzaffarpur": (26.12, 85.39),
    "bhagalpur": (25.24, 86.98), "darbhanga": (26.15, 85.90), "purnia": (25.78, 87.47), "ranchi": (23.34, 85.31),
    "bhopal": (23.26, 77.41), "indore": (22.72, 75.86), "nagpur": (21.15, 79.09), "nashik": (20.00, 73.79),
    "aurangabad": (19.88, 75.34), "surat": (21.17, 72.83), "varanasi": (25.32, 82.97), "agra": (27.18, 78.01),
    "guwahati": (26.14, 91.74), "bhubaneswar": (20.30, 85.82), "thiruvananthapuram": (8.52, 76.94),
    "kochi": (9.93, 76.27), "coimbatore": (11.02, 76.96), "madurai": (9.93, 78.12), "visakhapatnam": (17.69, 83.22),
    "chandigarh": (30.73, 76.78), "dehradun": (30.32, 78.03), "srinagar": (34.08, 74.80), "raipur": (21.25, 81.63),
    "jodhpur": (26.24, 73.02), "udaipur": (24.58, 73.71), "amritsar": (31.63, 74.87),
}
LOC_SIGMA_KM = {"exact point": 0.2, "PIN code centroid": 3, "PIN sorting-district centroid": 15,
                "PIN postal-region centroid": 40, "city centroid": 5, "district centroid": 12,
                "gazetteer city centre": 5, "state centroid (low precision)": 80}

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(s).lower())).strip()

def resolve_location(q, fac):
    """fac needs: lat, lon, geo_ok, postcode, city_norm, district_norm, state_norm."""
    q = (q or "").strip()
    pool = fac[fac["geo_ok"].astype(bool)]
    def cen(sub, label, precision):
        if len(sub) == 0:
            return None
        return dict(lat=float(sub.lat.median()), lon=float(sub.lon.median()), label=label, precision=precision,
                    n=int(len(sub)), sigma_km=LOC_SIGMA_KM.get(precision.split(" (")[0], 10))
    m = re.fullmatch(r"\(?\s*(-?\d{1,2}(?:\.\d+)?)\s*[, ]\s*(-?\d{2,3}(?:\.\d+)?)\s*\)?", q)
    if m:
        return dict(lat=float(m.group(1)), lon=float(m.group(2)), label=f"{q}", precision="exact point", n=None, sigma_km=0.2)
    if re.fullmatch(r"[1-9]\d{5}", q):
        pc = pool.postcode.fillna("").astype(str)
        for k, prec in [(6, "PIN code centroid"), (3, "PIN sorting-district centroid"), (2, "PIN postal-region centroid")]:
            sub = pool[pc.str[:k] == q[:k]]
            if len(sub):
                return cen(sub, f"PIN {q}", prec)
        return None
    parts = q.split(",", 1)
    place, state = _norm(parts[0]), (_norm(parts[1]) if len(parts) > 1 else "")
    if state:
        st = difflib.get_close_matches(state, pool.state_norm.dropna().unique(), n=1, cutoff=0.75)
        if st:
            pool = pool[pool.state_norm == st[0]]
    for col, prec in [("city_norm", "city centroid"), ("district_norm", "district centroid")]:
        if place and place in set(pool[col].dropna()):
            return cen(pool[pool[col] == place], q.title(), prec)
    if place in GAZETTEER:
        lat, lon = GAZETTEER[place]
        return dict(lat=lat, lon=lon, label=q.title(), precision="gazetteer city centre", n=None, sigma_km=5)
    if place and place in set(pool["state_norm"].dropna()):
        return cen(pool[pool.state_norm == place], q.title(), "state centroid (low precision)")
    for col, prec in [("city_norm", "city centroid"), ("district_norm", "district centroid")]:
        hit = difflib.get_close_matches(place, pool[col].dropna().unique(), n=1, cutoff=0.8)
        if hit:
            r = cen(pool[pool[col] == hit[0]], f"{hit[0].title()} (closest match to “{q}”)", prec)
            return r
    hit = difflib.get_close_matches(place, list(GAZETTEER), n=1, cutoff=0.8)
    if hit:
        lat, lon = GAZETTEER[hit[0]]
        return dict(lat=lat, lon=lon, label=hit[0].title(), precision="gazetteer city centre", n=None, sigma_km=5)
    return None

# ---------------------------------------------------------------------------------------------
# 5. Explanations
# ---------------------------------------------------------------------------------------------
FIELD_LABEL = {"procedure": "procedures list", "equipment": "equipment list", "capability": "capabilities list",
               "specialties": "specialties list", "description": "facility description", "name": "facility name"}

def quotes(evidence, facility_id, service, n=3):
    """Direct (core) quotes first; a related (support) quote only if there are fewer than two direct ones."""
    if evidence is None or len(evidence) == 0:
        return []
    e = evidence[(evidence.facility_id == facility_id) & (evidence.service == service)].copy()
    if e.empty:
        return []
    e["o"] = (e.kind != "core").astype(int) * 2 + e.negated.astype(int)
    e = e.sort_values("o").drop_duplicates("snippet")
    n_core = int((e.kind == "core").sum())
    e = e[e.kind == "core"].head(n) if n_core >= 2 else pd.concat([e[e.kind == "core"], e[e.kind != "core"].head(1)])
    return [dict(field=FIELD_LABEL.get(r.field, r.field), snippet=r.snippet, matched=r.matched,
                 negated=bool(r.negated), supporting=r.kind != "core") for r in e.head(n).itertuples()]

def registry_line(r):
    st = str(r.get("reg_status", "not checked"))
    p = r.get("reg_match_p")
    if st == "empanelled":
        sp = " for this specialty" if r.get("reg_specialty_match") else ""
        return ("ok", f"Listed as empanelled under a government scheme{sp} (record match {p:.0%}).")
    if st == "empanelled (possible match)":
        return ("maybe", f"Possibly the same hospital as a government-scheme listing (match {p:.0%}); confirm the name.")
    if st in ("de-empanelled", "suspended", "inactive"):
        d = r.get("reg_action_date") or "date unknown"
        return ("warn", f"Appears on a government {st} list ({d}). Confirm it can still treat scheme patients.")
    if st == "not found":
        return ("none", "Not found in the government-scheme lists we checked. That doesn't mean it can't help.")
    return ("none", "Government-scheme lists not checked for this facility.")

def caveats(r):
    out = []
    cs = str(r.get("coord_source", "record"))
    if cs != "record": out.append(f"Location estimated from a {cs.split(' (')[0]}; distance is approximate.")
    if r.get("flag_geo_state"): out.append("Map coordinates don't match the stated state. Confirm the address.")
    if r.get("flag_geo_city"): out.append("Map coordinates are far from the stated city.")
    if r.get("flag_pin_state"): out.append("PIN code doesn't match the stated state.")
    if r.get("flag_merge"): out.append("This record may mix information from more than one facility.")
    if r.get("flag_overclaim"): out.append("Lists many more services than its reported size suggests.")
    if (r.get("n_neg_core") or 0) > 0: out.append("Some text mentions this service as planned, referred out or unavailable.")
    if r.get("flag_out_of_place"): out.append("The rest of its profile doesn't look like a typical provider of this service.")
    for m in as_list(r.get("missing_prereqs")): out.append(f"No mention of {m}.")
    if (r.get("n_fields_core") or 0) < 2 and r.get("tier") in ("Strong", "Moderate"):
        out.append("Mentioned in only one part of the record (no cross-check).")
    return out

# ---------------------------------------------------------------------------------------------
# 6. LOW severity: multi-criteria ranking with uncertainty ranges and weight-sensitivity
# ---------------------------------------------------------------------------------------------
LOW_WEIGHTS = {"capability": 0.45, "trust": 0.20, "access": 0.25, "registry": 0.10}

def _low_score(df, w):
    return (df.p_capable.clip(1e-6) ** w["capability"] * df.trust.clip(1e-6) ** w["trust"]
            * df.access.clip(1e-9) ** w["access"] * df.reg_factor.clip(1e-6) ** w["registry"])

def rank_low(service, loc, fac, scores, k=3, n_sim=300, seed=0):
    d = SERVICES[service]
    rng = np.random.default_rng(seed)
    c = scores[(scores.service == service) & (scores.tier != "Unverified")].merge(fac, on="facility_id")
    c = c[c.coord_source != "none"].copy()
    if c.empty:
        return dict(results=c, pareto=c, candidates=c, radius=0)
    c["km"] = haversine_km(loc["lat"], loc["lon"], c.lat.values, c.lon.values)
    c["road_km"] = c.km * ROAD_FACTOR
    c["minutes"] = c.road_km / PLANNED_SPEED_KMH * 60
    c["access"] = np.exp(-c.road_km / d["decay_km"])
    radius = d["decay_km"] * 8
    c = c[c.road_km <= radius].sort_values("trust", ascending=False).drop_duplicates("dup_group")
    if c.empty:
        return dict(results=c, pareto=c, candidates=c, radius=radius)
    c["score"] = _low_score(c, LOW_WEIGHTS)
    c = c.sort_values("score", ascending=False).reset_index(drop=True)
    pool = c.head(25).copy()
    top1, topk = {}, {}
    for _ in range(n_sim):
        w = {kk: v * rng.uniform(0.6, 1.4) for kk, v in LOW_WEIGHTS.items()}
        order = pool.assign(s=_low_score(pool, w)).sort_values("s", ascending=False).facility_id.tolist()
        top1[order[0]] = top1.get(order[0], 0) + 1
        for f in order[:k]:
            topk[f] = topk.get(f, 0) + 1
    pool["p_top1"] = pool.facility_id.map(lambda f: top1.get(f, 0) / n_sim)
    pool["p_topk"] = pool.facility_id.map(lambda f: topk.get(f, 0) / n_sim)
    # Pareto frontier: nobody else is both closer and more likely to offer the service
    srt = c.sort_values(["road_km", "p_capable"], ascending=[True, False])
    best, keep = -1, []
    for i, r in srt.iterrows():
        if r.p_capable > best:
            keep.append(i); best = r.p_capable
    c["pareto"] = c.index.isin(keep)
    res = pool.head(k).copy()
    nearest_strong = c[c.tier == "Strong"].sort_values("km").head(1)
    return dict(results=res, candidates=c, pareto=c[c.pareto], radius=radius, nearest_strong=nearest_strong)

# ---------------------------------------------------------------------------------------------
# 7. HIGH severity: Monte Carlo over capability, operating status, location and travel time.
#    Plan = "go to facility i first". Outcome in each simulated world:
#      i can give definitive care      -> benefit exp(-t_i/tau)
#      i can only stabilise            -> exp(-t_i/tau) * exp(-(handover + transfer)/(tau*SLOW))
#      i can do neither                -> redirected unstabilised: exp(-(t_i + door + t_to_capable)/tau)
# ---------------------------------------------------------------------------------------------
DOOR_MIN, HANDOVER_MIN, SLOW = 10, 20, 4.0
DEFAULT_P = {"hospital": 0.15, "other": 0.03}

def _p_lookup(scores, service, ids, ftype):
    s = scores[scores.service == service].set_index("facility_id")["p_capable"]
    base = np.where(pd.Series(ftype).fillna("").str.contains("hospital", case=False).values,
                    DEFAULT_P["hospital"], DEFAULT_P["other"])
    return np.where(pd.Series(ids).isin(s.index).values, pd.Series(ids).map(s).fillna(0).values, base)

def simulate_emergency(service, loc, fac, scores, n_worlds=1000, n_cand=25, seed=0):
    d = SERVICES[service]
    win, tau = d["clock"]["window"], d["clock"]["tau"]
    rng = np.random.default_rng(seed)
    f = fac[fac.coord_source != "none"].copy()
    f["km"] = haversine_km(loc["lat"], loc["lon"], f.lat.values, f.lon.values)
    reach_km = EMERGENCY_SPEED_KMH * (2.0 * win / 60) / ROAD_FACTOR + 15
    near = f[f.km <= reach_km]
    if len(near) < 8:
        near = f.nsmallest(200, "km")
    near = near.copy()
    near["p_def"] = _p_lookup(scores, service, near.facility_id, near.facility_type)
    near["p_stab"] = np.maximum(_p_lookup(scores, "emergency_24x7", near.facility_id, near.facility_type), near.p_def)
    t_est = near.km * ROAD_FACTOR / EMERGENCY_SPEED_KMH * 60
    near["pre"] = ((near.p_def + 0.5 * near.p_stab) * np.exp(-t_est / tau) * (0.5 + 0.5 * near.trust)
                   * near.reg_factor.clip(0.2))
    cand = near.sort_values("pre", ascending=False).drop_duplicates("dup_group").head(n_cand).reset_index(drop=True)
    N, W = len(cand), n_worlds
    if N == 0:
        return None
    lat0 = cand.lat.values[None, :]; lon0 = cand.lon.values[None, :]
    sig = cand.sigma_km.fillna(5).values[None, :]
    lat = lat0 + rng.normal(0, 1, (W, N)) * sig / 111.0
    lon = lon0 + rng.normal(0, 1, (W, N)) * sig / (111.0 * np.cos(np.radians(lat0)))
    osig = loc.get("sigma_km", 1) or 1
    olat = loc["lat"] + rng.normal(0, 1, (W, 1)) * osig / 111.0
    olon = loc["lon"] + rng.normal(0, 1, (W, 1)) * osig / (111.0 * math.cos(math.radians(loc["lat"])))
    rf = rng.lognormal(math.log(ROAD_FACTOR), 0.12, (W, 1))
    sp = EMERGENCY_SPEED_KMH * rng.lognormal(0, 0.2, (W, 1))
    t0 = haversine_km(olat, olon, lat, lon) * rf / sp * 60
    tij = haversine_km(lat[:, :, None], lon[:, :, None], lat[:, None, :], lon[:, None, :]) * rf[:, :, None] / sp[:, :, None] * 60
    oper = rng.random((W, N)) < (0.85 + 0.13 * cand.trust.values)[None, :]
    cap_def = rng.random((W, N)) < cand.p_def.values[None, :]
    cap_stab = cap_def | (rng.random((W, N)) < cand.p_stab.values[None, :])
    ok_def, ok_stab = cap_def & oper, cap_stab & oper
    BIG = 1e6
    t_to_def = np.where(ok_def[:, None, :], tij, BIG)
    next_def = t_to_def.min(axis=2)
    next_def_idx = t_to_def.argmin(axis=2)
    t_to_stab = np.where(ok_stab[:, None, :], tij, BIG).min(axis=2)

    B = np.where(ok_def, np.exp(-t0 / tau),
                 np.where(ok_stab, np.exp(-t0 / tau) * np.exp(-(HANDOVER_MIN + next_def) / (tau * SLOW)),
                          np.exp(-(t0 + DOOR_MIN + next_def) / tau)))
    t_first = np.where(ok_stab, t0, t0 + DOOR_MIN + t_to_stab)                      # first competent hands
    t_defin = np.where(ok_def, t0, np.where(ok_stab, t0 + HANDOVER_MIN + next_def, t0 + DOOR_MIN + next_def))
    win_counts = np.bincount(B.argmax(axis=1), minlength=N)

    rows = []
    for i in range(N):
        dest = None
        mask = ~ok_def[:, i] & (next_def[:, i] < BIG / 2)
        if mask.any():
            j = np.bincount(next_def_idx[mask, i], minlength=N).argmax()
            dest = cand.name.iloc[j]
        role = ("Can likely treat" if cand.p_def.iloc[i] >= 0.5 else
                "Can likely stabilise, then transfer" if cand.p_stab.iloc[i] >= 0.5 else "Uncertain capability")
        tf = t_first[:, i]; td = t_defin[:, i]
        rows.append(dict(facility_id=cand.facility_id.iloc[i], name=cand.name.iloc[i], role=role,
                         p_def=cand.p_def.iloc[i], p_stab=cand.p_stab.iloc[i],
                         minutes_median=float(np.median(t0[:, i])), minutes_p10=float(np.percentile(t0[:, i], 10)),
                         minutes_p90=float(np.percentile(t0[:, i], 90)),
                         exp_benefit=float(B[:, i].mean()), win_pct=float(win_counts[i] / W),
                         p_first_in_window=float((tf <= win).mean()), p_definitive_in_window=float((td <= win).mean()),
                         typical_transfer=dest, km=float(cand.km.iloc[i])))
    plans = pd.DataFrame(rows).sort_values("exp_benefit", ascending=False).reset_index(drop=True)
    plans = plans.merge(cand.drop(columns=[c for c in ["name", "km", "p_def", "p_stab", "pre"] if c in cand]), on="facility_id", how="left")
    best_i = int(cand.index[cand.facility_id == plans.facility_id.iloc[0]][0])
    # Naive baseline: the nearest facility whose record mentions the service at all
    mention = set(scores[(scores.service == service) & (scores.tier != "Unverified")].facility_id)
    nb = cand[cand.facility_id.isin(mention)].sort_values("km")
    baseline = None
    if len(nb):
        bi = int(nb.index[0])
        baseline = dict(name=cand.name.iloc[bi], exp_benefit=float(B[:, bi].mean()),
                        p_first_in_window=float((t_first[:, bi] <= win).mean()),
                        p_definitive_in_window=float((t_defin[:, bi] <= win).mean()), km=float(cand.km.iloc[bi]))
    return dict(plans=plans, window=win, tau=tau, n_worlds=W, baseline=baseline,
                t_first_best=t_first[:, best_i][t_first[:, best_i] < BIG / 2],
                t_defin_best=t_defin[:, best_i][t_defin[:, best_i] < BIG / 2])
