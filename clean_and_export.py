import glob
import os
import re
import pandas as pd

RAW_DIR    = "Raw_data"
OUT_DIR    = os.path.join("Extracted_csvs")
os.makedirs(OUT_DIR, exist_ok=True)

# keep track of which files we've already written this run
processed = set()

# Optional: list tasks (the first filename token) to skip entirely
SKIP_TASKS = [
    "FoodAlone",
    "LightAlone",
    "ToyAlone",
    "FoodLight",
    "ToyLight",
    "NVK",
    "ChocolateMilk",
    "ChickenBroth"
]

# Possible manipulation groups (must match the cleaned token exactly)
GROUPS = ["WT", "Excitatory", "Inhibitory"]

# Which groups to skip entirely (don’t include "" here unless you really
# want to drop any file with no recognized group)
SKIP_GROUPS = [
    # "WT",
    # "Excitatory",
    # "Inhibitory",
    # ""        # if you still want to skip files with no group
]

KEEP    = ["Trial time", "X center", "Y center"]
KEEP_LC = [c.lower() for c in KEEP]
COL_MAP = dict(zip(KEEP_LC, KEEP))

def find_header_row(df):
    for i, row in df.iterrows():
        vals = [str(x).strip().lower() for x in row]
        if set(KEEP_LC).issubset(vals):
            return i
    return None

def find_animal_id(df):
    # 1) look for “Animal ID”
    for i, v in enumerate(df.iloc[:, 0]):
        if str(v).strip().lower() == "animal id":
            return str(df.iat[i, 1]).strip()
    # 2) fallback to “Subject”
    for i, v in enumerate(df.iloc[:, 0]):
        if str(v).strip().lower() == "subject":
            return str(df.iat[i, 1]).strip()
    return None

for fp in glob.glob(os.path.join(RAW_DIR, "*.xlsx")):
    base   = os.path.splitext(os.path.basename(fp))[0]
    # split on '_' and strip whitespace from each token
    tokens = [t.strip() for t in base.split("_")]

    # 1) Task = first token
    task = tokens[0]
    if task in SKIP_TASKS:
        print(f"[SKIP TASK] {base} -> task '{task}' is skipped")
        continue

    # 2) Find manipulation group token robustly
    group = ""
    for t in tokens:
        # remove any trailing non-letters/digits, lowercase
        clean = re.sub(r"[\W\d]+$", "", t).lower()
        for g in GROUPS:
            if clean == g.lower():
                group = g
                break
        if group:
            break

    print(f"  -> tokens={tokens}  DETECTED group='{group or 'NoGroup'}'")

    # Skip unwanted groups
    if group in SKIP_GROUPS:
        disp = group or "NoGroup"
        print(f"[SKIP GROUP] {base} → group '{disp}' is skipped")
        continue

    # 3) Condition = first 'ghrelin', 'saline' or 'ghrelin<number>', 'saline<number>' token
    condition = ""
    run_no    = ""
    cond_idx  = None

    for i, t in enumerate(tokens):
        tl = t.lower()
        # case 1: standalone token “ghrelin” or “saline”
        if tl in ("ghrelin", "saline"):
            condition = tl
            cond_idx  = i
            # next token may be the run number
            if i+1 < len(tokens) and tokens[i+1].isdigit():
                run_no = tokens[i+1]
            break

        # case 2: combined token “ghrelin1”, “saline2”, etc.
        m = re.match(r'^(ghrelin|saline)(\d+)$', tl)
        if m:
            condition = m.group(1)  # “ghrelin” or “saline”
            run_no    = m.group(2)  # the digits
            cond_idx  = i
            break

    # parse run_no (unused)
    run_no = ""
    if cond_idx is not None and cond_idx+1 < len(tokens) and tokens[cond_idx+1].isdigit():
        run_no = tokens[cond_idx+1]

    xls = pd.ExcelFile(fp, engine="openpyxl")
    for sheet in xls.sheet_names:
        df0 = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")

        # 4) Animal ID
        raw_id = find_animal_id(df0) or ""
        raw_id = raw_id.replace("Nazy", "Navy")
        raw_id = re.sub(r"[\s\-]+", "", raw_id)
        raw_id = re.sub(r"_\d+$",   "", raw_id)

        if not raw_id or "none" in raw_id.lower():
            print(f"[SKIP] {sheet!r}: invalid Animal ID '{raw_id}'")
            continue

        lid = raw_id.lower()
        if lid == "medsua":
            animal = "Medusa"
        elif lid == "offwhite":
            animal = "OffWhite"
        else:
            animal = raw_id

        # 5) Find header & slice
        hdr = find_header_row(df0)
        if hdr is None:
            print(f"[SKIP] {sheet!r}: no data header row")
            continue

        data = df0.iloc[hdr+1 :].copy()
        data.columns = [str(x).strip().lower() for x in df0.iloc[hdr]]
        if not set(KEEP_LC).issubset(data.columns):
            miss = set(KEEP_LC) - set(data.columns)
            print(f"[SKIP] {sheet!r}: missing columns {miss}")
            continue

        df_clean = (
            data[KEEP_LC]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(subset=KEEP_LC)
            .rename(columns=COL_MAP)
        )

        # 6) Build filename including group & condition
        parts = [task, animal]
        if group:
            parts.append(group)
        if condition:
            parts.append(condition)
        fname = "_".join(parts) + ".csv"
        outp  = os.path.join(OUT_DIR, fname)

        # 7) Write: overwrite on first encounter this run, else append
        parts = [task, animal]
        if group:
            parts.append(group)
        if condition:
            parts.append(condition)
        fname = "_".join(parts) + ".csv"
        outp  = os.path.join(OUT_DIR, fname)

        if fname in processed:
            # already wrote this file earlier in this run -> append
            mode, write_header = "a", False
        else:
            # first time this run → overwrite any old file
            mode, write_header = "w", True
            processed.add(fname)

        df_clean.to_csv(outp, mode=mode, index=False, header=write_header)
        print(f"[OK] {('Appended' if mode=='a' else 'Wrote')} {len(df_clean)} rows -> {fname}")


print("done")