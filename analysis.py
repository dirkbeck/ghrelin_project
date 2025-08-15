import os
import glob
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.stats import ttest_rel, f_oneway, entropy as shannon_entropy, linregress
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42

# ───────────────────────────────────────────────────────────────────────────
# 0) CONFIGURATION & CLEANUP
RAW_DIR = "Data/Extracted_csvs"
PLOTS   = "plots"
if os.path.exists(PLOTS):
    shutil.rmtree(PLOTS)

MAN_BOX    = os.path.join(PLOTS, "Manipulations", "boxplots")
MAN_ANOVA  = os.path.join(PLOTS, "Manipulations", "ANOVA_heatmaps")
MAN_SPIDER = os.path.join(PLOTS, "Manipulations", "spiderplots")
NON_PAIRED = os.path.join(PLOTS, "Non_manipulations", "paired_lineplots")
NON_LOGP   = os.path.join(PLOTS, "Non_manipulations", "logp_plot")
for d in (MAN_BOX, MAN_ANOVA, MAN_SPIDER, NON_PAIRED, NON_LOGP):
    os.makedirs(d, exist_ok=True)

allowed_tasks = ["ToyAlone", "FoodAlone", "LightAlone", "ToyLight", "FoodLight"]
four_metrics  = ["turn_freq", "mean_abs_angle", "mean_d", "msd_exp"]
metrics = {
    "mean_d":"Mean step length",
    "mean_abs_angle":"Mean abs turn-angle",
    "entropy_steps":"Entropy steps (bits)",
    "entropy_angles":"Entropy turns (bits)",
    "cv_steps":"CV steps",
    "cv_angles":"CV turns",
    "radius_gyration":"Radius of gyration",
    "msd_exp":"MSD exponent",
    "vel_autocorr":"Velocity autocorr (lag1)",
    "turn_freq":"Turn freq (>90°)",
    "pause_fraction":"Pause fraction",
    "tortuosity":"Tortuosity",
    "prop_Q1":"Prop. in Q1",
    "prop_Q2":"Prop. in Q2",
    "prop_Q3":"Prop. in Q3",
    "prop_Q4":"Prop. in Q4"
}

# ─────────────────────────────────────────────────────────────────────────────
def parse_fn(fn):
    base  = os.path.splitext(os.path.basename(fn))[0]
    parts = base.split("_")
    if   len(parts)==4: task,animal,group,tr = parts
    elif len(parts)==3: task,animal,tr = parts; group = ""
    else: return None
    tr = tr.lower()
    if tr not in ("saline","ghrelin"): return None
    if group not in ("","WT","Excitatory","Inhibitory"): return None
    return task, animal, group, tr

# ──────────────────────────────────────────────────────────────────────────
sigma_frames = 1.0
rows_steps, rows_angles, rows_moves = [], [], []

for path in tqdm(glob.glob(RAW_DIR+"/*.csv"), desc="Reading CSVs"):
    pr = parse_fn(path)
    if not pr: continue
    task,animal,group,tr = pr
    if task not in allowed_tasks:
        continue

    df = pd.read_csv(path, usecols=["X center","Y center"]).dropna()
    x, y = df["X center"].values, df["Y center"].values
    if len(x)<10: continue

    x_s = gaussian_filter1d(x, sigma=sigma_frames, mode="mirror")
    y_s = gaussian_filter1d(y, sigma=sigma_frames, mode="mirror")
    dx, dy = np.diff(x_s), np.diff(y_s)
    d       = np.hypot(dx, dy)
    theta       = np.arctan2(dy, dx)
    turns   = np.abs((np.diff(theta) + np.pi) % (2*np.pi) - np.pi)

    rows_steps += [
        dict(task=task, animal_id=animal, group=group,
             treatment=tr, d=di)
        for di in d if di>0
    ]
    rows_angles += [
        dict(task=task, animal_id=animal, group=group,
             treatment=tr, angle=ti)
        for ti in turns
    ]

    def ent(a,bins=20):
        c,_ = np.histogram(a, bins=bins)
        p   = c/c.sum() if c.sum()>0 else np.ones_like(c)/len(c)
        return shannon_entropy(p, base=2)

    H_d    = ent(d)
    H_t    = ent(turns)
    CV_d   = np.std(d)/np.mean(d) if d.mean()>0 else np.nan
    CV_t   = np.std(turns)/np.mean(turns) if turns.mean()>0 else np.nan
    x0,y0  = x_s.mean(), y_s.mean()
    Rg     = np.sqrt(np.mean((x_s-x0)**2 + (y_s-y0)**2))
    max_lag= min(20, len(x_s)//3)
    lags   = np.arange(1, max_lag+1)
    msd    = [np.mean((x_s[l:]-x_s[:-l])**2 + (y_s[l:]-y_s[:-l])**2)
              for l in lags]
    slope,*_ = linregress(np.log(lags), np.log(msd))

    v      = np.vstack((dx,dy)).T
    norms  = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = np.divide(v, norms, where=norms>0)
    vac    = np.mean((v_unit[1:]*v_unit[:-1]).sum(axis=1))

    freq_t     = np.mean(turns>np.pi/2)
    pause_frac = np.mean(d < np.percentile(d,10))
    net_disp   = np.hypot(x_s[-1]-x_s[0], y_s[-1]-y_s[0])
    tortuosity = net_disp / d.sum() if d.sum()>0 else np.nan

    mid_x = (x_s.min()+x_s.max())/2
    mid_y = (y_s.min()+y_s.max())/2
    quad_ids = ((x_s>mid_x).astype(int)*2 + (y_s>mid_y).astype(int))
    props    = {f"prop_Q{i+1}": np.mean(quad_ids==i) for i in range(4)}

    rows_moves.append(dict(
        task=task, animal_id=animal, group=group,
        treatment=tr,
        entropy_steps=H_d, entropy_angles=H_t,
        cv_steps=CV_d, cv_angles=CV_t,
        radius_gyration=Rg, msd_exp=slope,
        vel_autocorr=vac, turn_freq=freq_t,
        pause_fraction=pause_frac, tortuosity=tortuosity,
        **props
    ))

# ─────────────────────────────────────────────────────────────────────
df_steps  = pd.DataFrame(rows_steps)
df_angles = pd.DataFrame(rows_angles)
df_moves  = pd.DataFrame(rows_moves)

sess_d = df_steps.groupby(["task","group","animal_id","treatment"])["d"]\
                 .mean().reset_index(name="mean_d")
sess_a = df_angles.groupby(["task","group","animal_id","treatment"])["angle"]\
                  .apply(lambda a: np.mean(np.abs(a))).reset_index(name="mean_abs_angle")

sess = sess_d.merge(sess_a, on=["task","group","animal_id","treatment"])\
             .merge(df_moves, on=["task","group","animal_id","treatment"])
sess["treatment"] = sess["treatment"].str.lower()
sess["animal_id"] = sess["animal_id"].str.strip()

# ────────────────────────────────────────────────────────────────────────────
# 4) NON-MANIPULATION: paired lineplots & signed‐logp
summary_nm = []
for metr in four_metrics:
    label = metrics[metr]
    for task in sess[sess.group==""].task.unique():
        sub = sess[(sess.group=="") & (sess.task==task)]
        dfp = sub.pivot(index="animal_id", columns="treatment", values=metr).dropna()
        if dfp.shape[0] < 2:
            continue
        tstat, pval = ttest_rel(dfp["saline"], dfp["ghrelin"])
        delta = dfp["ghrelin"].mean() - dfp["saline"].mean()
        summary_nm.append(dict(task=task, metric=metr, delta=delta, pval=pval))

        task_dir = os.path.join(NON_PAIRED, task)
        os.makedirs(task_dir, exist_ok=True)

        plt.figure(figsize=(4,4))
        for _, r in dfp.iterrows():
            plt.plot(["saline","ghrelin"], [r["saline"],r["ghrelin"]],
                     "-o", color="gray", alpha=0.6)
        means = [dfp["saline"].mean(), dfp["ghrelin"].mean()]
        plt.plot(["saline","ghrelin"], means, "-o", color="red", lw=3, ms=8)
        plt.title(f"{task}: {label}")
        plt.ylabel(label)
        plt.text(0.5, max(means)*1.05, f"p = {pval:.3f}", ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, f"{metr}.pdf"))
        plt.close()

df_sum_nm = pd.DataFrame(summary_nm)
df_sum_nm["signed_logp"] = -np.log10(df_sum_nm["pval"]) * np.sign(df_sum_nm["delta"])
pivot_slp_4 = (df_sum_nm.pivot(index="task", columns="metric", values="signed_logp")
               .reindex(columns=four_metrics))

tasks_list = pivot_slp_4.index.tolist()
x = np.arange(len(tasks_list))
width = 0.18
palette = sns.color_palette("tab10", n_colors=len(four_metrics))
thr05, thr01 = -np.log10(0.05), -np.log10(0.01)

fig, ax = plt.subplots(figsize=(8,4.5))
for i, m in enumerate(four_metrics):
    vals = pivot_slp_4[m].fillna(0).values
    ax.bar(x + (i - 1.5)*width, vals, width, label=metrics[m], color=palette[i])
for y, txt in [(thr05, "p=0.05"), (thr01, "p=0.01"), (-thr05, "p=0.05"), (-thr01, "p=0.01")]:
    ax.axhline(y, color="gray", ls="--")
    ax.text(len(tasks_list)-0.3, y, txt,
            ha="right", va=("bottom" if y>0 else "top"), color="gray", fontsize=9)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(tasks_list, rotation=45, ha="right")
ax.set_ylabel("sign(delta) × –log10(p)")
ax.set_title("Non-manipulation: signed –log10(p) (4 metrics)")
ax.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(NON_LOGP, "signed_logp_4metrics.pdf"))
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5a) MANIPULATIONS BOXPLOTS
for metr in four_metrics:
    label = metrics[metr]
    for task in allowed_tasks:
        sub = sess[(sess.task==task) & (sess.group!="")]
        if sub.empty:
            continue
        task_dir = os.path.join(MAN_BOX, task)
        os.makedirs(task_dir, exist_ok=True)

        plt.figure(figsize=(4,4))
        sns.boxplot(x="group", y=metr, data=sub,
                    order=["WT","Excitatory","Inhibitory"], palette="pastel")
        sns.swarmplot(x="group", y=metr, data=sub,
                      order=["WT","Excitatory","Inhibitory"], color=".25")
        plt.title(f"{task}: {label}")
        plt.xlabel("Group")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, f"{metr}.pdf"))
        plt.close()

# ──────────────────────────────────────────────────────────────────────────
# 5b) MANIPULATIONS ANOVA HEATMAPS
group_sum = []
for metr in four_metrics:
    for task in allowed_tasks:
        for tr in ("saline","ghrelin"):
            sub = sess[(sess.task==task) & (sess.treatment==tr)]
            cnt = sub.groupby("group").animal_id.nunique()
            valid = cnt[cnt>=2].index.tolist()
            if len(valid) < 2:
                continue
            arrays = [sub[sub.group==g][metr].dropna() for g in valid]
            _, p = f_oneway(*arrays)
            group_sum.append(dict(task=task, treatment=tr, metric=metr, pval=p))

df_gsum = pd.DataFrame(group_sum)
df_gsum["cat"] = df_gsum["pval"].apply(lambda p: 2 if p<=0.01 else (1 if p<=0.05 else 0))

for tr in ("saline","ghrelin"):
    pivot_cat_4 = (df_gsum[df_gsum.treatment==tr]
                   .pivot(index="metric", columns="task", values="cat")
                   .reindex(index=four_metrics))
    fig, ax = plt.subplots(figsize=(len(allowed_tasks)*0.5+2, 4))
    sns.heatmap(pivot_cat_4,
                cmap=ListedColormap(["lightgray","orange","red"]),
                norm=BoundaryNorm([-0.5,0.5,1.5,2.5], ncolors=3),
                cbar=False, linewidths=0.5, linecolor="white", ax=ax)
    cbar = fig.colorbar(ax.collections[0], ax=ax,
                        boundaries=[-0.5,0.5,1.5,2.5], ticks=[0,1,2])
    cbar.set_ticklabels(["ns","0.05<p≤0.05","p≤0.01"])
    ax.set_title(f"{tr.capitalize()} group‐ANOVA (4 metrics)")
    plt.tight_layout()
    plt.savefig(os.path.join(MAN_ANOVA, f"group_ANOVA_{tr}_4metrics.pdf"))
    plt.close()

# ────────────────────────────────────────────────────────────────
# 6) SPIDER
conds, labels = [], []
for g in ["WT","Excitatory","Inhibitory"]:
    for tr in ["saline","ghrelin"]:
        if not sess[(sess.group==g) & (sess.treatment==tr)].empty:
            conds.append((g,tr))
            labels.append(f"{g}-{tr}")

N = len(four_metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
spider_labels = [metrics[m] for m in four_metrics]

def compute_delta(task=None):
    M = []
    for g,tr in conds:
        sub = sess[(sess.group==g) & (sess.treatment==tr)]
        if task: sub = sub[sub.task==task]
        M.append([sub[m].median() if len(sub)>0 else np.nan for m in four_metrics])
    M = np.array(M)
    base = M[0]
    delta = M - base
    ma = np.nanmax(np.abs(delta), axis=0)
    ma[ma==0] = 1.0
    return delta/ma

deltac = compute_delta()
fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.axvspan(angles[0], angles[1], facecolor="lightgray", alpha=0.3)
ax.axvspan(angles[2], angles[3], facecolor="lightblue", alpha=0.3)
for i, row in enumerate(deltac):
    vals = list(row) + [row[0]]
    ax.plot(angles, vals, "-o", label=labels[i])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(spider_labels, fontsize=10)
ax.set_title("Normalized ghrelin–saline (4-metric radar)", y=1.08)
ax.grid(True)
ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), title="Ghrelin - Saline")
plt.tight_layout()
plt.savefig(os.path.join(MAN_SPIDER, "movement_radar_4metrics.pdf"))
plt.close()

for task in allowed_tasks:
    deltat = compute_delta(task=task)
    if np.isnan(deltat).all():
        continue
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.axvspan(angles[0], angles[1], facecolor="lightgray", alpha=0.3)
    ax.axvspan(angles[2], angles[3], facecolor="lightblue", alpha=0.3)
    for i, row in enumerate(deltat):
        vals = list(row) + [row[0]]
        ax.plot(angles, vals, "-o", label=labels[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(spider_labels, fontsize=10)
    ax.set_title(f"{task}: normalized ghrelin–saline (4 metrics)", y=1.08)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), title="Ghrelin - Saline")
    plt.tight_layout()
    plt.savefig(os.path.join(MAN_SPIDER, f"{task}_radar_4metrics.pdf"))
    plt.close()

print("All pdfs saved")