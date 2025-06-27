#!/usr/bin/env python3
# paired_metrics_plots.py  –  Original · Generated · Black-input
import json, pathlib, re, textwrap
import numpy as np                     # ← add this line
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
# 1. CONFIG                                                          #
# ------------------------------------------------------------------ #
RESULTS_DIR = pathlib.Path(
    r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\evaluation_results"
)
METRICS = ["Dice", "IoU", "ASSD", "HD95"]

# ------------- legacy originals/extras (only *_post.json) ----------
LEGACY = {
    "resultsorgpost":        "Original",
    "resultsorgpostGT2":     "Original 2nd GT",
    "resultsextrapost":      "Extra Data",
    "resultsextrapostGT2":   "Extra Data 2nd GT",
}

# ---------------- generated mixes (IDs 541-556) --------------------
GEN_MAP = {
    541: "All sequences",
    542: "T1",
    543: "T1C",
    544: "T2",
    545: "STIR",
    546: "DWI",
    547: "ADC",
    548: "DWI+ADC",
    551: "30% T2+30% STIR",
    552: "T2/STIR + all DWI/ADC",
    553: "T2/STIR + 60% DWI/ADC",
    554: "30% T1+30% T1C",
    555: "T1/T1C + 60% DWI/ADC",
    556: "T1/T1C + DWI/ADC + T2/STIR",
}

# --------------- black-input mixes (IDs 641-656) -------------------
BLACK_MAP = {k+100: v for k, v in GEN_MAP.items()}

# colour palette for the paired boxes
PALETTE = {"Generated": "#0343df",  # blue
           "Black":      "#FF007F"} # pink

# regex for *_post.json and *_post_GTreal.json
post_re = re.compile(r"(\d{3})_post(?:_GTreal)?$")

# ------------------------------------------------------------------ #
# 2. LOAD JSONs                                                      #
# ------------------------------------------------------------------ #
records = []

for jf in RESULTS_DIR.glob("*.json"):
    stem = jf.stem

    # legacy (single blue bar)
    if stem in LEGACY:
        with open(jf) as f: data = json.load(f)
        for case in data["metric_per_case"]:
            for m in METRICS:
                records.append({"Group": LEGACY[stem],
                                "Source": "Generated",
                                "Metric": m,
                                "Value": case["metrics"]["1"][m]})
        continue

    # generated or black mixes
    m = post_re.match(stem)
    if not m: continue
    ds_id = int(m.group(1))

    if ds_id in GEN_MAP:
        group = GEN_MAP[ds_id]
        src = "Generated"
    elif ds_id in BLACK_MAP:
        group = BLACK_MAP[ds_id]
        src = "Black"
    else:
        continue

    with open(jf) as f: data = json.load(f)
    for case in data["metric_per_case"]:
        for m in METRICS:
            records.append({"Group": group,
                            "Source": src,
                            "Metric": m,
                            "Value": case["metrics"]["1"][m]})

df = pd.DataFrame.from_records(records)
if df.empty:
    raise RuntimeError("No matching post-JSON files found")

# ------------------------------------------------------------------ #
# 3. BUILD x-axis order only from labels that exist                  #
# ------------------------------------------------------------------ #
def sort_key(g):
    if   g.startswith("Original"): return (0, g)
    elif g.startswith("Extra"):    return (1, g)
    else:                          return (2, g)

present_groups = sorted(df["Group"].unique(), key=sort_key)
df["Group"]  = pd.Categorical(df["Group"], categories=present_groups, ordered=True)
df["Source"] = pd.Categorical(df["Source"], categories=["Generated","Black"])

# ------------------------------------------------------------------ #
# 4. PLOTS  –  default legend, points centred on their box           #
# ------------------------------------------------------------------ #
import itertools, matplotlib.patches as mpatches
sns.set(style="whitegrid", context="talk")

BOX_W     = 0.55
PALETTE   = {"Black": "#ff69b4", "Generated": "#0343df"}
HUE_ORDER = ["Black", "Generated"]              # pink first, then blue
x_order   = ["Original", "All sequences", "T1", "T1C", "T2", "STIR", "DWI", "ADC"]

# keep only groups that exist
df["Group"]  = pd.Categorical(df["Group"],
                              categories=[g for g in x_order if g in df["Group"].unique()],
                              ordered=True)
df["Source"] = pd.Categorical(df["Source"], categories=HUE_ORDER)

for metric in METRICS:
    sub = df[df["Metric"] == metric]
    fig, ax = plt.subplots(figsize=(15, 7))

    # --- main boxes -------------------------------------------------
    sns.boxplot(
        data=sub, x="Group", y="Value", hue="Source",
        order=df["Group"].cat.categories, hue_order=HUE_ORDER,
        palette=PALETTE, ax=ax,
        width=BOX_W,
        showcaps=True,
        showfliers=False,           #  ←  outliers gone
        medianprops={"color": "white", "lw": 0},
        boxprops={"edgecolor": "black", "lw": 1.2},
        whiskerprops={"lw": 1.2}
    )


    # ------------------------------------------------------------------ #
    #   centres = {}  –  find the true centre & width of every box       #
    # ------------------------------------------------------------------ #
    # More robust way of indexing centres
    centres = {}
    xticks = ax.get_xticks()
    num_groups = len(df["Group"].cat.categories)
    num_hue = len(HUE_ORDER)

    offset = -BOX_W / 2 + BOX_W / num_hue / 2
    step = BOX_W / num_hue

    for i, grp in enumerate(df["Group"].cat.categories):
        for j, src in enumerate(HUE_ORDER):
            x_centre = xticks[i] + offset + step * j
            centres[(grp, src)] = (x_centre, step)


    # ------------------------------------------------------------------ #
    #   dots / mean stripe / median diamond                              #
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(seed=0)                 # reproducible jitter
    for (grp, src), (x_c, bw) in centres.items():
        vals = sub[(sub["Group"] == grp) & (sub["Source"] == src)]["Value"]
        if vals.empty:
            continue

        med = vals.median()

        # ----------------------------------------------------------------
        # 1. label all points "effectively equal to the median"
        #    atol=1e-6 catches any float-rounding wiggle
        # ----------------------------------------------------------------
        is_median = np.isclose(vals, med, atol=1e-6, rtol=0)

        # 2. keep only the NON-median values for the jittered scatter
        vals_jitter = vals[~is_median]

        # 3. jitter-scatter the surviving points
        jitter = (rng.random(len(vals_jitter)) - 0.5) * bw * 0.50    # ±¼ box-width
        ax.scatter(x_c + jitter, vals_jitter,
                color="black", s=30, alpha=.7, zorder=4)

        # 4. mean stripe
        mean = vals.mean()
        ax.plot([x_c - bw/2, x_c + bw/2], [mean, mean],
                color="black", lw=2.2, zorder=5)

        # 5. median diamond (make it big enough to dominate visually)
        ax.scatter(x_c, med, marker="D", s=30,
                facecolor="white", edgecolor="black", zorder=5)




    # ------------------------------------------------------------------ #
    #   legend – keep the default handles/labels, bring the frame back   #
    # ------------------------------------------------------------------ #
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,
            frameon=True,      # <-- draw the usual border again
            loc="lower right", bbox_to_anchor=(0.95, 0.05),
            borderaxespad=0.0, facecolor="white", edgecolor="black")




    # --- titles & labels -------------------------------------------
    ax.set_title(textwrap.fill(f"{metric} Original vs Generated vs Black-Input", 60),
                 fontsize=20, weight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=18)
    ax.tick_params(axis="x", rotation=25, labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    plt.tight_layout()
    plt.show()
