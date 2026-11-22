# CareCompass: CMU Datathon Fall 2026

```
CareCompass_Pipeline.py   Databricks notebook: EDA → trust → evidence → graph + text models → registry linking
                          → calibrated P(capable) → severity classifier → low/high branches → validation → impact → gold tables
app/engine.py             shared query-time logic (triage, ranking, Monte Carlo); the notebook imports it too
app/app.py                Streamlit dashboard
app/app.yaml              Databricks App config
registries/               CSV templates for PM-JAY / state / de-empanelled lists
```

## 1. Run the notebook (Databricks Free Edition)
1. Put this folder in your workspace, keeping the layout (the notebook imports `app/engine.py`).
   The easiest way is to push it to GitHub and use **Workspace → Create → Git folder**. Otherwise create a folder and import
   `CareCompass_Pipeline.py` (it becomes a notebook), then create an `app` subfolder and upload the files in `app/`.
2. Optional, for government verification: run the notebook once (it creates the Volume), then upload CSVs to
   `/Volumes/workspace/care_compass/registries/`. Use the templates' columns, loosely matched. File names containing
   `deempanel`, `suspend` or `inactive` count as negative lists. Re-run from section 6.
3. **Run all.** Check the column-mapping table in section 1 on the first run. Section 13 writes gold tables to
   `workspace.care_compass.*` and a parquet snapshot to `/Volumes/workspace/care_compass/exports/`.

## 2. Host the dashboard as a Databricks App (recommended for judging)
1. **Compute → Apps → Create app → Custom**, name it `carecompass`.
2. Under **App resources**, add your SQL warehouse (Free Edition has a Serverless Starter Warehouse) with permission
   *Can use*, and resource key `sql-warehouse`.
3. Deploy from the workspace path of the `app/` folder.
4. Give the app's service principal read access. Its ID is shown on the app page. Run this in a SQL editor:
   ```sql
   GRANT USE CATALOG ON CATALOG workspace TO `<app-service-principal-id>`;
   GRANT USE SCHEMA, SELECT ON SCHEMA workspace.care_compass TO `<app-service-principal-id>`;
   ```
5. Open the app URL. If it reports a data error, the grant or warehouse resource is the usual cause.

## 3. Backup: run anywhere
Download the 6 parquet files from the exports Volume into `app/data/`, then:
```bash
cd app && pip install -r requirements.txt && streamlit run app.py
```
The same folder deploys unchanged to Streamlit Community Cloud. Commit `app/data/` to the repo and choose `app/app.py` as the entry point.

## Demo script (≈ 2 minutes)
1. *"accident hua hai, behosh hai" · Lucknow*: triage explains which words made it an emergency, shows "Call 108", one place to go first, and the 10-dot odds of reaching help in time.
2. *"bahut khoon beh raha hai delivery ke baad" · Nashik*: the same flow with the 102 maternal-transport line, plus stabilise-then-transfer if the nearest place can't operate.
3. *"dialysis twice a week" · Gaya, Bihar*: planned care. Three options with likelihood ranges, quotes from each record, government-list badges, and the closer-vs-more-certain chart.
4. **How we decided** expander: the Monte Carlo table, the time-to-care histogram, and the comparison with "nearest facility that mentions it".
5. **Data health** tab: care-gap map and the severity safety test.
