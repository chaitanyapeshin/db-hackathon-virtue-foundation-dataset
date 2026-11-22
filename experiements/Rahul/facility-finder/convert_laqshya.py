import sys, re, pandas as pd, pdfplumber
src, out = sys.argv[1], sys.argv[2]
rows = []
with pdfplumber.open(src) as pdf:
    for i, p in enumerate(pdf.pages):
        for t in p.extract_tables():
            for r in t:
                if r and r[0] and re.match(r"^\d+$", r[0].strip()):
                    rows.append([i + 1] + [(c or "").replace("\n", " ").strip() for c in r])
df = pd.DataFrame(rows, columns=["page", "sno", "state", "district", "ftype", "name", "dept", "certified", "conditional", "date_result", "valid_to"])

def subseq(word, s):
    it = iter(s.replace(" ", "").lower()); return all(ch in it for ch in word.replace(" ", "").lower())

fixes = 0
for k, r in df.iterrows():
    ft = r["ftype"]
    if ft not in ("DH", "SDH", "CHC", "MC", "PHC"):
        up = "".join(c for c in ft if c.isupper()); low = "".join(c for c in ft if c.islower())
        df.at[k, "ftype"] = up; df.at[k, "district"] = r["district"] + low; fixes += 1
    d = df.at[k, "district"]
    m = re.match(r"^([a-z]+)([A-Z].*)$", d)          # state overflow, e.g. 'Arunachal Prades' + 'h'
    if m: df.at[k, "state"] = r["state"] + m.group(1); df.at[k, "district"] = m.group(2); fixes += 1
    dp = r["dept"]
    if dp not in ("Labour Room", "Maternity OT", ""):
        df.at[k, "dept"] = "Labour Room" if subseq("labourroom", dp) else ("Maternity OT" if subseq("maternityot", dp) else "")
        fixes += 1

MANUAL = {"1346": ("Shahid Bhagat Singh Nagar", "DH", "DH Nawanshahr"), "1347": ("Shahid Bhagat Singh Nagar", "DH", "DH Nawanshahr"),
          "1950": ("Gautam Buddha Nagar", "DH", "District Combined Hospital GB Nagar"), "1951": ("Gautam Buddha Nagar", "DH", "District Combined Hospital GB Nagar")}
for k, r in df.iterrows():
    if r["sno"] in MANUAL:
        df.at[k, "district"], df.at[k, "ftype"], df.at[k, "name"] = MANUAL[r["sno"]]; fixes += 1

REG = {"Labour Room": ("LAQSHYA_LR", "emergency childbirth;high-risk delivery"),
       "Maternity OT": ("LAQSHYA_MOT", "emergency c-section;emergency childbirth;high-risk delivery"),
       "": ("LAQSHYA", "emergency childbirth;high-risk delivery")}
def iso(d):
    try: return pd.to_datetime(d, dayfirst=True).strftime("%Y-%m-%d")
    except Exception: return ""
outdf = pd.DataFrame({
    "registry": df["dept"].map(lambda d: REG[d][0]),
    "facility_name": df["name"].str.replace(r"\s+", " ", regex=True),
    "state": df["state"], "district": df["district"], "pincode": "",
    "services": df["dept"].map(lambda d: REG[d][1]),
    "valid_to": df["valid_to"].map(iso),
    "status": ["conditional" if c.strip() else ("certified" if ce.strip() else "unknown") for ce, c in zip(df["certified"], df["conditional"])],
    "facility_type": df["ftype"],
    "source_url": "https://qps.nhsrcindia.org/sites/default/files/2026-04/LaQshya_Master_data-08-04-2026.pdf",
    "source_ref": [f"LaQshya master list 08-04-2026, p.{p}, S.No {s}" for p, s in zip(df["page"], df["sno"])],
    "retrieved_on": "2026-11-22",
})
outdf.to_csv(out, index=False)
print(f"rows {len(outdf)} | cells repaired {fixes} | by registry {outdf.registry.value_counts().to_dict()}")
print("status", outdf.status.value_counts().to_dict(), "| facility types", outdf.facility_type.value_counts().to_dict())
today = "2026-11-22"
print("still valid on", today, ":", (outdf.valid_to >= today).sum(), "| expired:", (outdf.valid_to < today).sum(), "| no date:", (outdf.valid_to == "").sum())
print("states", outdf.state.nunique(), sorted(outdf.state.unique()))
