#!/usr/bin/env python3
"""
plot_reader_confusions.py
—————————————
Visualise multi-reader real-vs-generated experiments.

  • Confusion matrices: expert, students, all readers
  • Sequence-level error breakdown (stacked bars)

Usage:
    python plot_reader_confusions.py experiment_results_multi.txt
"""

import re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────── CONFIGURE HERE ──────────────────────────────
# Put the exact names (case-sensitive) that appear after “Guesses …:” and that
# you want treated as professionals / radiologists.  Everyone else counts as a
# student.
PRO_READERS = {"Roland"}
# ────────────────────────────────────────────────────────────────────────────
# Desired left-to-right (or top-to-bottom) order for sequences
SEQ_ORDER = ["ADC", "DWI", "STIR", "T2", "T1C", "T1"]

# ---------------------------------------------------------------------------
# 1.  PARSE THE MASTER TEXT FILE
# ---------------------------------------------------------------------------
def parse_results(path):
    """
    Returns
    -------
    true_labels   : dict  slide → 'gen' | 'real'
    predictions   : dict  reader → {slide: pred}
    slides_info   : dict  slide → (sequence, truth_label)
    """
    true_labels, predictions = {}, defaultdict(dict)
    slides_info = {}                 # slide → (sequence, truth)

    section, reader_name = None, None
    next_anon = 1

    with Path(path).open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()

            # ---- Section headers -------------------------------------------
            if low.startswith("gen:"):
                section, reader_name = "gen", None
                continue
            if low.startswith("real:"):
                section, reader_name = "real", None
                continue
            m = re.match(r"guesses\s*([^:]*):", line, re.I)
            if m:
                reader_name = m.group(1).strip() or f"Reader{next_anon}"
                if not m.group(1).strip():
                    next_anon += 1
                section = "guess"
                continue

            # ---- Content lines ----------------------------------------------
            if section in {"gen", "real"}:
                # e.g. "10 T2"
                m = re.match(r"(\d+)\s+(\S+)", line)
                if m:
                    slide = int(m.group(1))
                    seq   = m.group(2).upper()
                    truth = "gen" if section == "gen" else "real"
                    true_labels[slide]  = truth
                    slides_info[slide] = (seq, truth)

            elif section == "guess" and reader_name:
                # e.g. "10 gen"
                m = re.match(r"(\d+)\s+(gen|real)", line, re.I)
                if m:
                    slide = int(m.group(1))
                    pred  = m.group(2).lower()
                    predictions[reader_name][slide] = pred

    return true_labels, predictions, slides_info


# ---------------------------------------------------------------------------
# 2.  METRICS
# ---------------------------------------------------------------------------
def confusion(true_labels, pred_dict):
    """Return 2×2 matrix [[TN FP], [FN TP]]."""
    TN = FP = FN = TP = 0
    for slide, truth in true_labels.items():
        pred = pred_dict.get(slide)
        if pred is None:
            continue
        if truth == "gen":
            if pred == "gen":
                TP += 1
            else:
                FN += 1
        else:  # truth == "real"
            if pred == "real":
                TN += 1
            else:
                FP += 1
    return np.array([[TN, FP],
                     [FN, TP]], dtype=int)


def metrics(cm):
    TN, FP = cm[0]
    FN, TP = cm[1]
    n = cm.sum()
    acc  = (TP + TN) / n
    prec = TP / (TP + FP) if TP + FP else 0
    rec  = TP / (TP + FN) if TP + FN else 0
    spec = TN / (TN + FP) if TN + FP else 0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, spec, f1


# ---------------------------------------------------------------------------
# 3.  PLOTTING HELPERS
# ---------------------------------------------------------------------------
def plot_cm(ax, cm, title, vmax=None):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    fontsize=16, weight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Generated"], fontsize=14)
    ax.set_yticklabels(["Real", "Generated"], fontsize=14)
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(title, fontsize=16, pad=10)


def sequence_breakdown(slides_info, predictions, reader_subset):
    """
    Returns dict: sequence → {'TN', 'FP', 'FN', 'TP'} counts summed
                  over reader_subset.
    """
    stats = defaultdict(lambda: dict(TN=0, FP=0, FN=0, TP=0))
    for reader in reader_subset:
        preds = predictions[reader]
        for slide, (seq, truth) in slides_info.items():
            pred = preds.get(slide)
            if pred is None:
                continue
            if truth == "gen" and pred == "gen":
                stats[seq]["TP"] += 1
            elif truth == "gen" and pred == "real":
                stats[seq]["FN"] += 1
            elif truth == "real" and pred == "real":
                stats[seq]["TN"] += 1
            elif truth == "real" and pred == "gen":
                stats[seq]["FP"] += 1
    return stats


def plot_stacked(ax, stats, title):
    seqs = [s for s in SEQ_ORDER if s in stats]
    TN = [stats[s]["TN"] for s in seqs]
    TP = [stats[s]["TP"] for s in seqs]
    FP = [stats[s]["FP"] for s in seqs]
    FN = [stats[s]["FN"] for s in seqs]

    y = np.arange(len(seqs))
    ax.barh(y, TN,                   color="#08519c", label="TN")
    ax.barh(y, TP, left=TN,          color="#3182bd", label="TP")
    ax.barh(y, FP, left=np.array(TN)+TP,   color="#fd8d3c", label="FP")
    ax.barh(y, FN, left=np.array(TN)+TP+FP, color="#de2d26", label="FN")

    ax.set_yticks(y)
    ax.set_yticklabels(seqs, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)


# ---------------------------------------------------------------------------
# 4.  MAIN
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_reader_confusions.py results.txt")
        sys.exit(1)

    fname = sys.argv[1]
    true_lbls, preds, slides_info = parse_results(fname)

    # Split readers
    pro_readers     = [r for r in preds if r in PRO_READERS]
    student_readers = [r for r in preds if r not in PRO_READERS]

    # ------------------------------------------------------------------
    # 4a. Per-reader & group confusion matrices
    # ------------------------------------------------------------------
    pro_cm     = np.zeros((2, 2), int)
    student_cm = np.zeros((2, 2), int)

    print("\n=== Individual readers ===")
    for reader, p in preds.items():
        cm = confusion(true_lbls, p)
        acc, prec, rec, spec, f1 = metrics(cm)
        print(f"{reader:12s}  CM={cm.tolist()}  acc={acc:.3f}  f1={f1:.3f}")
        if reader in PRO_READERS:
            pro_cm += cm
        else:
            student_cm += cm

    agg_cm = pro_cm + student_cm
    acc, prec, rec, spec, f1 = metrics(agg_cm)
    print("\n=== Aggregate (all readers) ===")
    print(f"CM={agg_cm.tolist()}  acc={acc:.3f}  f1={f1:.3f}")

    # ------------------------------------------------------------------
    # 4b. Plot heat-maps
    # ------------------------------------------------------------------
    vmax = agg_cm.max()
    fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_cm(axes[0], pro_cm,     "Radiologist", vmax)
    plot_cm(axes[1], student_cm, "Students",    vmax)
    plot_cm(axes[2], agg_cm,     "All readers", vmax)
    fig1.tight_layout()

    # ------------------------------------------------------------------
    # 4c. Sequence-level stacked bars
    # ------------------------------------------------------------------
    pro_stats     = sequence_breakdown(slides_info, preds, pro_readers)
    student_stats = sequence_breakdown(slides_info, preds, student_readers)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    plot_stacked(axes2[0], pro_stats,     "Radiologist by sequence")
    plot_stacked(axes2[1], student_stats, "Students by sequence")
    axes2[1].legend(loc="lower right", fontsize=10)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
