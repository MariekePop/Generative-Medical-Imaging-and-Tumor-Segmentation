#!/usr/bin/env python3
# pretty_metrics_plots.py – now with friendlier labels
import json, pathlib, re, textwrap
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# 1. CONFIGURATION – adjust only in this block                       #
# ------------------------------------------------------------------ #
RESULTS_DIR = pathlib.Path(
    r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\evaluation_results"
)
METRICS = ["Dice", "IoU", "ASSD", "HD95"]

# legacy (unchanged)
LEGACY = [
    ("resultsorg",            "Original (raw)",           "#95d0fc"),
    ("resultsorgpost",        "Original (post)",          "#0343df"),
    ("resultsorgGT2",         "Original 2nd GT (raw)",    "#a2cffe"),
    ("resultsorgpostGT2",     "Original 2nd GT (post)",   "#0165fc"),
    ("resultsextra",          "Extra (raw)",              "#fdaa48"),
    ("resultsextrapost",      "Extra (post)",             "#ff5b00"),
    ("resultsextraGT2",       "Extra 2nd GT (raw)",       "#f9c894"),
    ("resultsextrapostGT2",   "Extra 2nd GT (post)",      "#f97306"),
]

# pretty names for the modality mixes (raw/post added automatically)
COMB_MAP = {
    541: "All sequences generated",
    542: "T1 generated",
    543: "T1C generated",
    544: "T2 generated",
    545: "STIR generated",
    546: "DWI generated",
    547: "ADC generated",
    548: "DWI + ADC generated",
    551: "30 % T2 + 30 % STIR gen.",
    552: "T2/STIR + all DWI/ADC gen.",
    553: "T2/STIR + 60 % DWI/ADC gen.",
    554: "30 % T1 + 30 % T1C gen.",
    555: "T1/T1C + 60 % DWI/ADC gen.",
    556: "T1/T1C + DWI/ADC + T2/STIR gen.",
}

# optional per-mix colours (leave empty to reuse palette_raw/post)
COMB_COLOURS = {
    # 541: "#a6cee3",
    # 542: "#1f78b4",
    # … fill if desired …
}

PALETTE_RAW  = "#95d0fc"   # default for every “(raw)”
PALETTE_POST = "#0343df"   # default for every “(post)”

# ------------------------------------------------------------------ #
# 2. LOAD JSONs                                                      #
# ------------------------------------------------------------------ #
records, colour_map = [], {}
legacy_map = {s: (p, c) for s, p, c in LEGACY}
combo_re   = re.compile(r"(\d{3})_(raw|post)$")

for jf in RESULTS_DIR.glob("*.json"):
    stem = jf.stem

    # ---- handle legacy files --------------------------------------
    if stem in legacy_map:
        pretty, col = legacy_map[stem]
        colour_map[pretty] = col
        with open(jf) as f:
            data = json.load(f)
        for case in data["metric_per_case"]:
            for m in METRICS:
                records.append({"Label": pretty,
                                "Metric": m,
                                "Value": case["metrics"]["1"][m]})
        continue

    # ---- handle combo files ---------------------------------------
    m = combo_re.match(stem)
    if not m: continue
    ds_id, stage = int(m.group(1)), m.group(2)
    if ds_id not in COMB_MAP: continue

    pretty = f"{COMB_MAP[ds_id]} ({'raw' if stage=='raw' else 'post'})"
    colour_map[pretty] = PALETTE_RAW if stage == "raw" else PALETTE_POST
    with open(jf) as f:
        data = json.load(f)
    for case in data["metric_per_case"]:
        for m in METRICS:
            records.append({"Label": pretty,
                            "Metric": m,
                            "Value": case["metrics"]["1"][m]})

df = pd.DataFrame.from_records(records)
if df.empty:
    raise RuntimeError("No matching result files found!")

# ------------------------------------------------------------------ #
# 3. ORDER labels                                                    #
# ------------------------------------------------------------------ #
order = [p for _, p, _ in LEGACY]
for ds in sorted(COMB_MAP):
    order += [f"{COMB_MAP[ds]} (raw)", f"{COMB_MAP[ds]} (post)"]
present = [lbl for lbl in order if lbl in df["Label"].unique()]
df["Label"] = pd.Categorical(df["Label"], categories=present, ordered=True)

# ------------------------------------------------------------------ #
# 4. PLOTS                                                           #
# ------------------------------------------------------------------ #
sns.set(style="whitegrid", context="talk")
BOX_W = 0.55

for metric in METRICS:
    fig, ax = plt.subplots(figsize=(14, 7))
    sub = df[df["Metric"] == metric]
    palette = [colour_map[lbl] for lbl in present]

    sns.boxplot(data=sub, x="Label", y="Value",
                order=present, palette=palette, ax=ax,
                width=BOX_W, showcaps=True, showfliers=True,
                medianprops={"color": "white", "linewidth": 0},
                boxprops={"edgecolor": "black", "linewidth": 1.2},
                whiskerprops={"linewidth": 1.2}, fliersize=3)

    sns.stripplot(data=sub, x="Label", y="Value",
                  order=present, color="black", size=4,
                  alpha=0.6, ax=ax, jitter=True)

    # add mean & median markers
    for x, lbl in enumerate(present):
        vals = sub[sub["Label"] == lbl]["Value"]
        if vals.empty: continue
        mean, med = vals.mean(), vals.median()
        ax.plot([x-BOX_W/2, x+BOX_W/2], [mean]*2, color="black", lw=2.2)
        ax.scatter(x, med, marker="D", s=30,
                   facecolor="white", edgecolor="black", zorder=5)

    ax.set_title(textwrap.fill(f"{metric} Across Generated Sequence Combinations", 60),
                 fontsize=20, pad=12, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=18)
    clean_labels = [re.sub(r" \((raw|post)\)$", "", lbl) for lbl in present]
    ax.set_xticklabels(clean_labels, rotation=30, ha="center", fontsize=18)
    ax.tick_params(axis="x", rotation=30, labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    plt.tight_layout()
    plt.show()
