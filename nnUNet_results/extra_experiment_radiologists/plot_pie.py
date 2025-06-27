#!/usr/bin/env python3
"""
pie_preference.py
─────────────────
Parse “Guess:” lines in a radiologist-preference file and plot a pie chart.

Run:
    python pie_preference.py  segmentation_votes.txt
"""

import sys
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

# ────────────────────────────── CONFIGURATION ──────────────────────────────
# Mapping from raw tokens found in the file → pretty labels on the plot
LABEL_DISPLAY = {
    "R-GT1":      "GT_trained",   # radiologist’s original ground truth (GT1)
    "PREDICTION": "Prediction",   # model output
    "S-GT2":      "GT2"           # colleague’s segmentation
}

# Desired left-to-right (clockwise) order for the wedges
ORDER_RAW = ["S-GT2", "R-GT1", "PREDICTION"]

# Colours for the wedges in the same order as ORDER_RAW
COLORS = ["#d95f02", "#0343df", "#4daf4a"]
# ───────────────────────────────────────────────────────────────────────────


def main():
    # ---------------------------------------------------------------------
    # 0  Get input file
    # ---------------------------------------------------------------------
    if len(sys.argv) < 2:
        print("Usage: python pie_preference.py <results_file.txt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 1  Parse lines containing “Guess:”
    # ---------------------------------------------------------------------
    wanted = {k.lower() for k in LABEL_DISPLAY.keys()}  # lower-case set
    choices = []

    for raw in path.read_text().splitlines():
        if "guess" not in raw.lower():
            continue
        last_token = raw.split()[-1]          # last word in the line
        if last_token.lower() in wanted:
            choices.append(last_token.upper())    # normalise to upper case

    if not choices:
        print("No valid 'Guess:' lines found.")
        sys.exit(1)

    # ---------------------------------------------------------------------
    # 2  Count occurrences in the specified order
    # ---------------------------------------------------------------------
    counter = Counter(choices)
    counts  = [counter.get(k, 0) for k in ORDER_RAW]
    labels  = [LABEL_DISPLAY[k]   for k in ORDER_RAW]
    total   = sum(counts)

    # ---------------------------------------------------------------------
    # 3  Plot pie chart
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct=lambda pct: f"{pct:.0f}%",
        startangle=140,
        colors=COLORS,
        textprops={'fontsize': 14}
    )

    ax.set_title(
        f"Radiologist Preference",
        fontsize=16, pad=20
    )
    ax.axis('equal')          # perfect circle
    plt.tight_layout()
    plt.show()

    # Optional: save to file
    # fig.savefig("preference_pie.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
