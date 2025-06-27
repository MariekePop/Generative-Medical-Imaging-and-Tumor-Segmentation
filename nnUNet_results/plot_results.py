#!/usr/bin/env python3
"""
pretty_metrics_plots.py
-----------------------
Create four individual box-and-strip plots from multiple nnU-Net
results*.json files with custom names, order, and colours.
"""

import json
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# 1.  CONFIGURATION                                                  #
# ------------------------------------------------------------------ #
RESULTS_DIR = pathlib.Path(
    r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\evaluation_results"
)
METRICS = ["Dice", "IoU", "ASSD", "HD95"]

ORDER_LABELS = [
    ("resultsorg",            "Original"),
    ("resultsorgpost",        "Original Post"),
    ("resultsorgGT2",         "Original 2nd GT"),
    ("resultsorgpostGT2",     "Original Post 2nd GT"),
    ("resultsextra",          "Extra"),
    ("resultsextrapost",      "Extra Post"),
    ("resultsextraGT2",       "Extra 2nd GT"),
    ("resultsextrapostGT2",   "Extra Post 2nd GT"),
]

# nice, distinct fill colours
BLUE_LIGHT        = "#95d0fc"      # Original
BLUE_DARK         = "#0343df"      # Original Post
BLUE_LIGHT_GT2    = "#a2cffe" # Original 2nd GT
BLUE_DARK_GT2     = "#0165fc"      # Original Post 2nd GT
ORANGE_LIGHT      = "#fdaa48"      # Extra
ORANGE_DARK       = "#ff5b00"      # Extra Post
ORANGE_LIGHT_GT2  = "#f9c894"      # Extra 2nd GT
ORANGE_DARK_GT2   = "#f97306"      # Extra Post 2nd GT

# ------------------------------------------------------------------ #
# 2.  LOAD ALL JSON FILES                                            #
# ------------------------------------------------------------------ #
label_map     = dict(ORDER_LABELS)                      # raw-stem → pretty
order_pretty  = [pretty for _, pretty in ORDER_LABELS]  # canonical order

records = []
for jf in RESULTS_DIR.glob("*.json"):
    stem = jf.stem
    if stem not in label_map:
        continue                                        # skip unknown files
    with open(jf) as f:
        data = json.load(f)

    pretty = label_map[stem]
    for case in data["metric_per_case"]:
        for m in METRICS:
            records.append(
                {"Label": pretty, "Metric": m, "Value": case["metrics"]["1"][m]}
            )

df = pd.DataFrame.from_records(records)
if df.empty:
    raise RuntimeError("No matching JSON files found in folder.")

present_labels = [lbl for lbl in order_pretty if lbl in df["Label"].unique()]
df["Label"] = pd.Categorical(df["Label"], categories=present_labels, ordered=True)

# ------------------------------------------------------------------ #
# 3.  COLOUR MAP                                                     #
# ------------------------------------------------------------------ #
colour_map = {}
for _, lbl in ORDER_LABELS:
    if lbl.startswith("Original"):
        if "2nd GT" in lbl:
            colour_map[lbl] = BLUE_DARK_GT2 if "Post" in lbl else BLUE_LIGHT_GT2
        else:
            colour_map[lbl] = BLUE_DARK      if "Post" in lbl else BLUE_LIGHT
    else:  # Extra…
        if "2nd GT" in lbl:
            colour_map[lbl] = ORANGE_DARK_GT2 if "Post" in lbl else ORANGE_LIGHT_GT2
        else:
            colour_map[lbl] = ORANGE_DARK     if "Post" in lbl else ORANGE_LIGHT

# ------------------------------------------------------------------ #
# 4.  PLOT  –  one figure per metric                                 #
# ------------------------------------------------------------------ #
sns.set(style="whitegrid", context="talk")
BOX_WIDTH = 0.55

for metric in METRICS:
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = df[df["Metric"] == metric]

    palette = [colour_map[lbl] for lbl in present_labels]

    sns.boxplot(
        data=sub, x="Label", y="Value",
        order=present_labels, palette=palette, ax=ax,
        width=BOX_WIDTH, showcaps=True, showfliers=True,
        medianprops={"color": "white", "linewidth": 0},
        boxprops={"edgecolor": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        fliersize=3,
    )

    # individual case dots
    sns.stripplot(
        data=sub, x="Label", y="Value", order=present_labels,
        color="black", size=4, alpha=0.6, ax=ax, jitter=True,
    )

    # mean bar & small median diamond
    grp = sub.groupby("Label")["Value"]
    for x, lbl in enumerate(present_labels):
        if lbl not in grp.groups:
            continue
        mean, median = grp.mean()[lbl], grp.median()[lbl]

        # mean
        ax.plot([x - BOX_WIDTH/2, x + BOX_WIDTH/2], [mean]*2,
                color="black", lw=2.2, zorder=5)
        # median
        ax.scatter(x, median, marker="D", s=30,
                   facecolor="white", edgecolor="black", zorder=6)

    ax.set_title(f"Comparison of {metric}", fontsize=22, weight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=18)
    ax.tick_params(axis="x", rotation=25, labelsize=13)
    ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    plt.show()
