import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# -----------------------------------------------------------
# 1. TEXT-PARSING
# -----------------------------------------------------------
def parse_results(path: str):
    """Return true_labels dict and predictions dict-of-dicts."""
    true_labels = {}
    predictions = defaultdict(dict)

    section = None          # "gen", "real" or current reader name
    reader_name = None      # only valid while in a Guesses block
    next_anon_id = 1        # fallback names: Reader1, Reader2, …

    with Path(path).open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            lower = line.lower()

            # --- Detect section headers ---
            if lower.startswith("gen:"):
                section = "gen"; reader_name = None; continue
            if lower.startswith("real:"):
                section = "real"; reader_name = None; continue

            m = re.match(r"guesses\s*([^:]*):", lower)
            if m:
                name = m.group(1).strip().title() or f"Reader{next_anon_id}"
                next_anon_id += 1 if not m.group(1).strip() else 0
                section, reader_name = "guess", name
                continue

            # --- Parse the content lines ---
            # Slides & sequences => true labels
            if section in {"gen", "real"}:
                m = re.match(r"(\d+)\s", line)
                if m:
                    slide = int(m.group(1))
                    true_labels[slide] = "gen" if section == "gen" else "real"

            # Reader guesses => predictions[reader][slide]
            elif section == "guess":
                m = re.match(r"(\d+)\s+(gen|real)", line, re.I)
                if m and reader_name:
                    slide = int(m.group(1))
                    pred  = m.group(2).lower()
                    predictions[reader_name][slide] = pred

    return true_labels, predictions


# -----------------------------------------------------------
# 2. METRICS
# -----------------------------------------------------------
def confusion(true, pred):
    """Return 2×2 matrix [[TN FP],[FN TP]] for one reader."""
    TN = FP = FN = TP = 0
    for slide, t in true.items():
        p = pred.get(slide)
        if p is None:              # missing guess ⇒ skip
            continue
        if t == "gen":
            TP += p == "gen"
            FN += p == "real"
        else:  # t == "real"
            TN += p == "real"
            FP += p == "gen"
    return np.array([[TN, FP],
                     [FN, TP]], dtype=int)


def metrics(cm):
    """Return accuracy, precision, recall, specificity, F1."""
    TN, FP, FN, TP = *cm[0], *cm[1]
    acc = (TP + TN) / cm.sum()
    prec = TP / (TP + FP) if TP + FP else 0
    rec  = TP / (TP + FN) if TP + FN else 0
    spec = TN / (TN + FP) if TN + FP else 0
    f1   = 2*prec*rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, spec, f1


# -----------------------------------------------------------
# 3. PLOTTING
# -----------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np   # only if you create cm on the fly

def plot_cm(
        cm: np.ndarray,
        title: str = "All Predictions",
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



# -----------------------------------------------------------
# 4. MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    true_lbls, preds = parse_results("experiment_results_multi.txt")

    aggregate = np.zeros((2, 2), dtype=int)

    print("\n=== Individual readers ===")
    for reader, p in preds.items():
        cm = confusion(true_lbls, p)
        aggregate += cm
        acc, prec, rec, spec, f1 = metrics(cm)
        print(f"{reader:10s}  CM={cm.tolist()}  acc={acc:.3f}  f1={f1:.3f}")

    print("\n=== Aggregate over all readers ===")
    acc, prec, rec, spec, f1 = metrics(aggregate)
    print(f"CM={aggregate.tolist()}  acc={acc:.3f}  f1={f1:.3f}")

    # Uncomment if you want the aggregate heat-map
    plot_cm(aggregate)


