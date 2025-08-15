# README

This project contains two distinct analysis pipelines:

1. **Movement analysis**  
   – Reads raw tracking Excel files stored in `Data/Raw_data/`  
   – Cleans & exports per-trial CSVs  
   – Computes movement metrics and produces behavioral plots  

2. **Modeling & illustration**  
   – Generates standalone schematic figures (performance curves, decision-maps, phase diagrams)  
   – Does *not* depend on the raw or cleaned data  

All code and scripts live in the same project directory.

---

## Prerequisites

• Python 3.7+  
• Packages:
```
pip install pandas openpyxl numpy scipy seaborn matplotlib tqdm
```

---

## 0. Importing Excel files

Before running the movement analysis, you must download the raw `.xlsx` files from [TBD] into the `Data/Raw_data/` folder.

1. Create the folder (if it doesn’t exist):
   ```bash
   mkdir -p Data/Raw_data
   ```
2. Copy or move your `.xlsx` files there:
   ```bash
   cp /path/to/my_data/*.xlsx Data/Raw_data/
   ```
3. **Naming convention**  
   Filenames follow the pattern:
   ```
   <Task>_<AnimalID>_<Group>_<Condition>.xlsx
   ```
   where:
   - `<Task>` matches one of your task names (e.g. “ToyAlone”, “FoodLight”, etc.)  
   - `<AnimalID>` is your subject identifier  
   - `<Group>` is one of `WT`, `Excitatory`, `Inhibitory` (or omitted for no‐manipulation)  
   - `<Condition>` is `saline`, `ghrelin`, `saline1`, `ghrelin2`, etc.  
   
   This lets `clean_and_export.py` parse task, animal, group, and treatment automatically.

---

## 1. Movement Analysis

### Directory structure
```
.
├── Data
│   ├── Raw_data              # ← Excel files here
│   ├── Extracted_csvs        # ← auto-generated CSVs
│   └── clean_and_export.py   # Excel → cleaned CSV exporter
├── analysis.py               # Reads CSVs and writes movement plots
└── plots                     # ← auto-generated PDFs
```

### Workflow

1. **Clean & export**  
   ```bash
   python3 Data/clean_and_export.py
   ```
   - Scans `Data/Raw_data/*.xlsx`  
   - Extracts “Trial time”, “X center”, “Y center” plus metadata  
   - Writes `<task>_<animal>_<group>_<condition>.csv` into `Data/Extracted_csvs/`

2. **Generate movement plots**  
   ```bash
   python3 analysis.py
   ```
   - Reads all CSVs in `Data/Extracted_csvs/`  
   - Computes step lengths, turn angles, entropy, CV, radius of gyration, MSD exponent, velocity autocorr, turn frequency, pause fraction, tortuosity, quadrant proportions, etc.  
   - Produces PDFs under `plots/`:
     - **Non-manipulation**: paired line-plots & signed –log p bar charts  
     - **Manipulation**: boxplots, ANOVA heatmaps, radar (spider) plots  

Both scripts will create their output directories automatically if they don’t exist.

---

## 2. Modeling & Illustration

These scripts are self-contained and do not read the Excel or CSV data. Each produces one or more conceptual figures:

- **blood_levels_model.py** → `perf_complexity.pdf`  
  • Panel A: inverted-U performance curves  
  • Panel B: success-probability heatmap  

- **decision_making_maps.py** → `low_ghrelin_maps.pdf`, `high_ghrelin_maps.pdf`  
  • Decision-avoidance maps with/without dimension dependence  

- **dimensionality_by_strio_activity_10_levels.py** → `decision_space.pdf`  
  • DMS/DLS stackplots of decision-space dimensionality  

- **phase_plot.py** → `phase_plot.pdf`, `line_analysis.pdf`  
  • Phase diagram & summary line plots

- **ability_to_perform_tasks.py** → `ability_to_perform_tasks.pdf`
  • Line plots of how shifts in DMS/DLS activity impact performance

Run them in any order:
```bash
python3 perf_complexity.py
python3 decision_maps.py
python3 decision_space.py
python3 phase_plot.py
python3 ability_to_perform_tasks.py
```
