import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_results(path: str):
    """
    Parse the results text file and return two dictionaries:
    true_labels[slide]  -> 'gen' | 'real'
    pred_labels[slide]  -> 'gen' | 'real'
    """
    true_labels, pred_labels = {}, {}
    section = None                    # Tracks whether we’re in Gen, Real or guesses

    with Path(path).open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            lower = line.lower()
            if lower.startswith("gen:"):
                section = "gen";   continue
            if lower.startswith("real:"):
                section = "real";  continue
            if lower.startswith("guesses:"):
                section = "guess"; continue

            # -----------------------------
            # Parse slide-number & label
            # -----------------------------
            if section in {"gen", "real"}:      # True-label sections
                m = re.match(r"(\d+)\s", line)
                if m:
                    slide = int(m.group(1))
                    true_labels[slide] = "gen" if section == "gen" else "real"

            elif section == "guess":            # Prediction section
                m = re.match(r"(\d+)\s+(gen|real)", line, re.I)
                if m:
                    slide = int(m.group(1))
                    pred  = m.group(2).lower()
                    pred_labels[slide] = pred

    return true_labels, pred_labels


def compute_confusion(true_labels, pred_labels):
    """
    Returns a 2×2 NumPy array:
        [[TN, FP],
         [FN, TP]]
    """
    TP = TN = FP = FN = 0
    for slide, true_lbl in true_labels.items():
        pred_lbl = pred_labels.get(slide)
        if pred_lbl is None:                       # Safety check
            continue

        if true_lbl == "gen" and pred_lbl == "gen":
            TP += 1
        elif true_lbl == "gen" and pred_lbl == "real":
            FN += 1
        elif true_lbl == "real" and pred_lbl == "real":
            TN += 1
        elif true_lbl == "real" and pred_lbl == "gen":
            FP += 1

    return np.array([[TN, FP],
                     [FN, TP]])



def plot_cm(
        cm: np.ndarray,
        title: str = "Radiologists Prediction",
        *,
        cell_fs: int   = 18,   # numbers inside each square
        title_fs: int  = 20,   # plot title
        label_fs: int  = 18,   # axis-label font size
        tick_fs: int   = 16,   # “Real / Generated” tick labels
        cmap: str      = "Blues",
        figsize: tuple = (5, 5)   # overall figure size in inches
    ):
    """
    Draw a 2×2 confusion matrix with comfortably large fonts.

    Parameters
    ----------
    cm : 2×2 NumPy array
        [[TN FP]
         [FN TP]]
    title : str
        Title shown above the heat-map.
    *  : keyword-only arguments (see defaults above)
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap=cmap)

    # Numbers in each cell
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center",
                    fontsize=cell_fs, weight="bold")

    # Axis ticks & labels
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Real", "Generated"], fontsize=tick_fs)
    ax.set_yticklabels(["Real", "Generated"], fontsize=tick_fs)
    ax.set_xlabel("Predicted label", fontsize=label_fs)
    ax.set_ylabel("True label", fontsize=label_fs)

    # Title
    ax.set_title(title, fontsize=title_fs, pad=16)

    # Tight layout for neat cropping
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Parse file
    true_lbls, pred_lbls = parse_results("experiment_results.txt")

    # 2. Compute confusion-matrix counts
    cm = compute_confusion(true_lbls, pred_lbls)
    print("Confusion matrix:\n", cm)

    # 3. Plot
    plot_cm(cm)
