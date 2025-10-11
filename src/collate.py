#!/usr/bin/env python
# coding: utf-8

# # Data collation
# ---

# In[2]:


#Importing packages
import os
import glob
import re
from pathlib import Path

import pandas as pd
import numpy as np


# In[3]:


# base project path
project_root = Path(os.getcwd()).resolve().parent  # change as needed
interim_base = project_root / "data" / "interim"

da_folder = interim_base / "da"
rt_folder = interim_base / "rt"

assert da_folder.exists() or rt_folder.exists(), "Make sure the interim/da or interim/rt folders exist."


# In[4]:


# find files
da_files = sorted(glob.glob(str(da_folder / "data_da_*_cleaned_no_neg.csv")))
rt_files = sorted(glob.glob(str(rt_folder / "data_rt_*_cleaned_no_neg.csv")))

print(f"Found {len(da_files)} DA files and {len(rt_files)} RT files.")

# helper: canonical asset id column name (common names)
def _asset_id_col(df):
    for candidate in ["Masked Asset ID", "asset_id", "asset id", "Asset ID"]:
        if candidate in df.columns:
            return candidate
    # fallback: first numeric-like column with many unique values
    return df.columns[0]

from pathlib import Path

def read_and_prepare(path, market_type):
    """
    Read CSV and return df with:
      - canonical timestamp '_ts' (prefers int_start / interval_start),
      - 'date' (YYYY-MM-DD str) and 'hour' (HH:MM:SS) derived from _ts,
      - market_type and source_file columns.
    """
    path = Path(path)
    df = pd.read_csv(path)
    src = path.name

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # rename asset id
    aid = _asset_id_col(df)
    df = df.rename(columns={aid: "asset_id"})

    # map lower->original for quick lookup
    cols_low = {c.lower(): c for c in df.columns}

    # priority: int_start / interval_start, then int_end / interval_end, then ts/timestamp, then Date+hour
    start_candidates = ["int_start", "interval_start", "intervalstart", "intstart"]
    end_candidates   = ["int_end", "interval_end", "intervalend", "intend"]
    ts_candidates    = ["ts", "timestamp", "time_stamp"]

    def find_col(cands):
        for x in cands:
            if x in cols_low:
                return cols_low[x]
        return None

    ts_col = find_col(start_candidates) or find_col(end_candidates) or find_col(ts_candidates)

    if ts_col:
        df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    else:
        # fallback: Date + hour
        date_col = next((cols_low[c] for c in ("date", "day") if c in cols_low), None)
        hour_col = next((cols_low[c] for c in ("hour", "time", "hour_of_day") if c in cols_low), None)
        if date_col and hour_col:
            df["_ts"] = pd.to_datetime(df[date_col].astype(str) + " " + df[hour_col].astype(str),
                                       errors="coerce")
        elif date_col:
            df["_ts"] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        else:
            # last resort: try parsing any column that looks like datetimes
            parsed = None
            for c in df.columns:
                if df[c].dtype == object:
                    cand = pd.to_datetime(df[c], errors="coerce")
                    if not cand.isna().all():
                        parsed = cand
                        break
            df["_ts"] = parsed if parsed is not None else pd.NaT

    # If some rows still NaT, try to fill from any column that contains 'int' + 'start'/'end' in the name
    if df["_ts"].isna().any():
        for c in df.columns:
            cl = c.lower()
            if ("int" in cl and ("start" in cl or "end" in cl)) or ("interval" in cl and ("start" in cl or "end" in cl)):
                candidate = pd.to_datetime(df[c], errors="coerce")
                fill_mask = df["_ts"].isna() & ~candidate.isna()
                if fill_mask.any():
                    df.loc[fill_mask, "_ts"] = candidate[fill_mask]

    # Build date and hour from _ts (hour reflects the interval start time)
    df["date"] = df["_ts"].dt.date.astype("str")
    df["hour"] = df["_ts"].dt.strftime("%H:%M:%S")    # HH:MM:SS from interval_start

    df["market_type"] = market_type
    df["source_file"] = src

    return df


# In[5]:


# read
da_dfs = [read_and_prepare(p, "da") for p in da_files]
rt_dfs = [read_and_prepare(p, "rt") for p in rt_files]

# sort within each dataframe by timestamp and asset id
def _sort_df(df):
    if "_ts" not in df.columns:
        df["_ts"] = pd.NaT
    return df.sort_values(["_ts", "asset_id"]).reset_index(drop=True)

da_dfs = [_sort_df(df) for df in da_dfs]
rt_dfs = [_sort_df(df) for df in rt_dfs]

# concat DA first then RT
wide_da = pd.concat(da_dfs, ignore_index=True) if da_dfs else pd.DataFrame()
wide_rt = pd.concat(rt_dfs, ignore_index=True) if rt_dfs else pd.DataFrame()
wide_all = pd.concat([wide_da, wide_rt], ignore_index=True)

# ensure final chronological order inside each market and DA above RT
wide_all["_market_order"] = wide_all["market_type"].map({"da": 0, "rt": 1}).fillna(1)
wide_all = wide_all.sort_values(["_market_order", "_ts", "asset_id"]).reset_index(drop=True)
wide_all = wide_all.drop(columns=["_market_order"])

# save wide to processed folder
processed_folder = project_root / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)
wide_path = processed_folder / "master_data_wide.csv"
wide_all.to_csv(wide_path, index=False)
print(f"Saved wide consolidated -> {wide_path} (shape: {wide_all.shape})")


# In[6]:


# concat DA then RT (they are each internally sorted)
wide_da = pd.concat(da_dfs, ignore_index=True) if da_dfs else pd.DataFrame()
wide_rt = pd.concat(rt_dfs, ignore_index=True) if rt_dfs else pd.DataFrame()

# final wide dataset: DA first then RT
wide_all = pd.concat([wide_da, wide_rt], ignore_index=True)

# optional: ensure a clear chronological index column for the whole DF
wide_all = wide_all.sort_values(["market_type", "_ts", "asset_id"]).reset_index(drop=True)

# save to processed folder
processed_folder = project_root / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)
wide_path = processed_folder / "master_data_wide.csv"
wide_all.to_csv(wide_path, index=False)

print(f"Saved wide consolidated to: {wide_path}")
print("Wide shape:", wide_all.shape)


# In[7]:


# Diagnostic cell for wide_all and segX_mw columns
import re
from pprint import pprint

print("wide_all.shape:", wide_all.shape)
print("\nColumns (first 50):")
pprint(list(wide_all.columns)[:50])

print("\nFirst 6 rows preview:")
display(wide_all.head(6))

# find segX_mw and segX_price columns explicitly (1..10)
seg_mw_cols = []
seg_price_cols = []
for i in range(1, 11):
    # possible variants
    for cand in (f"seg{i}_mw", f"seg_{i}_mw", f"segment {i} mw", f"Segment {i} MW", f"Segment_{i}_MW", f"seg{i}mw"):
        matches = [c for c in wide_all.columns if c.lower() == cand.lower()]
        if matches:
            seg_mw_cols.append(matches[0])
            break
    # price
    for cand in (f"seg{i}_price", f"seg_{i}_price", f"seg{i}_prc", f"segment {i} price"):
        matches = [c for c in wide_all.columns if c.lower() == cand.lower()]
        if matches:
            seg_price_cols.append(matches[0])
            break

# fallback: regex detect any col with digit + mw/prc/price anywhere
if not seg_mw_cols:
    seg_mw_cols = sorted([c for c in wide_all.columns if re.search(r'\bseg[_\s\-]?\d+[_\s\-]?mw\b', c, re.I)])
if not seg_price_cols:
    seg_price_cols = sorted([c for c in wide_all.columns if re.search(r'(price|prc)', c, re.I) and re.search(r'\d', c)])

print("\nDetected seg_mw columns:")
pprint(seg_mw_cols)
print("\nDetected seg_price-like columns:")
pprint(seg_price_cols)

# show dtype and non-null counts and some sample non-null values for each seg mw col
print("\nseg MW columns diagnostics:")
for c in seg_mw_cols:
    n_non = int(wide_all[c].notna().sum())
    dtype = wide_all[c].dtype
    # sample up to 6 non-null unique values
    sample_vals = wide_all[c].dropna().astype(str).unique()[:6].tolist()
    print(f" - {c} | dtype={dtype} | non-null={n_non} | sample values={sample_vals}")

# show if _ts is present and some examples
if "_ts" in wide_all.columns:
    n_nat = int(wide_all["_ts"].isna().sum())
    print(f"\n_ts present: True | NaT count: {n_nat}")
    print("Sample _ts values:", wide_all["_ts"].dropna().unique()[:6])
else:
    print("\n_ts column not present in wide_all")

# confirm market types present
if "market_type" in wide_all.columns:
    print("\nmarket_type unique values:", wide_all["market_type"].unique())
else:
    print("\nmarket_type column not present")


# In[8]:


cols = list(wide_all.columns)
seg_mw_map = {}
seg_price_map = {}

def candidate_cols_for_seg(n):
    """Return columns that likely correspond to segment n and MW or price."""
    nstr = str(n)
    mw_cands = []
    price_cands = []
    for c in cols:
        lc = c.lower()
        # candidate if it contains the segment number and 'mw'
        if re.search(r'\b' + re.escape(nstr) + r'\b', lc) and 'mw' in lc:
            mw_cands.append(c)
        # allow seg1 in condensed names like 'seg1_mw' (no word boundary)
        if re.search(r'seg[_\-\s]*' + re.escape(nstr), lc) and 'mw' in lc:
            mw_cands.append(c)
        # price candidates: number + price/prc or segN_price etc
        if (re.search(r'\b' + re.escape(nstr) + r'\b', lc) or re.search(r'seg[_\-\s]*' + re.escape(nstr), lc)) \
           and ('price' in lc or 'prc' in lc):
            price_cands.append(c)
    # also include any 'Segment N MW' style (space) matches
    for c in cols:
        lc = c.lower()
        if f"segment {n}" in lc and 'mw' in lc:
            if c not in mw_cands:
                mw_cands.append(c)
        if f"segment {n}" in lc and ('price' in lc or 'prc' in lc):
            if c not in price_cands:
                price_cands.append(c)
    # deduplicate
    mw_cands = list(dict.fromkeys(mw_cands))
    price_cands = list(dict.fromkeys(price_cands))
    return mw_cands, price_cands

# pick best column (highest non-null count) among candidates
for n in range(1, 11):
    mw_cands, price_cands = candidate_cols_for_seg(n)
    best_mw = None
    best_price = None

    if mw_cands:
        counts = {c: int(wide_all[c].notna().sum()) for c in mw_cands}
        # choose column with max non-null
        best_mw = max(counts, key=lambda k: counts[k])
        seg_mw_map[n] = best_mw

    if price_cands:
        counts_p = {c: int(wide_all[c].notna().sum()) for c in price_cands}
        best_price = max(counts_p, key=lambda k: counts_p[k])
        seg_price_map[n] = best_price

print("Chosen MW columns per segment (by max non-null):")
for n in sorted(seg_mw_map.keys()):
    c = seg_mw_map[n]
    print(f"  Segment {n}: {c} | non-null = {int(wide_all[c].notna().sum())}")

print("\nChosen Price columns per segment (by max non-null):")
for n in sorted(seg_price_map.keys()):
    c = seg_price_map[n]
    print(f"  Segment {n}: {c} | non-null = {int(wide_all[c].notna().sum())}")

if not seg_mw_map:
    raise RuntimeError("No MW candidate columns found. Please run diagnostic and paste column list here.")

# Coerce chosen columns to numeric (remove commas/spaces)
for n, col in seg_mw_map.items():
    wide_all[col] = pd.to_numeric(wide_all[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
for n, col in seg_price_map.items():
    wide_all[col] = pd.to_numeric(wide_all[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")

# Report counts after coercion
print("\nAfter coercion: non-null counts by chosen MW column")
for n in sorted(seg_mw_map.keys()):
    col = seg_mw_map[n]
    print(f"  Segment {n}: {col} non-null = {int(wide_all[col].notna().sum())}")

# Build long frames using chosen columns
long_frames = []
for seg_num in sorted(seg_mw_map.keys()):
    mw_col = seg_mw_map[seg_num]
    price_col = seg_price_map.get(seg_num, None)
    tmp = pd.DataFrame({
        "date": wide_all["_ts"].dt.date.astype(str) if "_ts" in wide_all.columns else wide_all.get("Date", pd.Series(index=wide_all.index, dtype="object")),
        "hour": wide_all["_ts"].dt.strftime("%H:%M:%S") if "_ts" in wide_all.columns else wide_all.get("hour", pd.Series(index=wide_all.index, dtype="object")),
        "asset_id": wide_all.get("asset_id", wide_all.columns[0]),
        "segment": seg_num,
        "MW": wide_all[mw_col],
        "price": wide_all[price_col] if price_col is not None else np.nan,
        "market_type": wide_all.get("market_type", np.nan),
        "source_file": wide_all.get("source_file", np.nan),
        "_ts": wide_all.get("_ts", pd.NaT)
    })
    kept = tmp[tmp["MW"].notna()].copy()
    print(f" Segment {seg_num}: kept rows = {len(kept)} (from column: {mw_col})")
    long_frames.append(kept)

if not any(len(f) for f in long_frames):
    raise RuntimeError("No rows retained across all segments after filtering. Please inspect outputs above.")

long_all = pd.concat(long_frames, ignore_index=True)
long_all["_market_order"] = long_all["market_type"].map({"da": 0, "rt": 1}).fillna(1)
long_all = long_all.sort_values(["_market_order", "_ts", "asset_id", "segment"]).reset_index(drop=True)
long_all = long_all.drop(columns=["_market_order"])

# save long into processed folder
processed_folder = project_root / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)
long_path = processed_folder / "master_data_long.csv"
long_all.to_csv(long_path, index=False)

print("\nSaved long consolidated ->", long_path, " shape:", long_all.shape)
display(long_all.head(30))

