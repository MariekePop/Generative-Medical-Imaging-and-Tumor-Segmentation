#!/usr/bin/env python3
"""
Evaluate binary 3-D segmentations against ground truth and emit
nnU-Net-style JSON metrics.

Usage
-----
python eval_segmentation.py \
       --pred_dir  /path/to/predictions \
       --ref_dir   /path/to/ground_truth \
       --out_json  /path/to/metrics.json
"""

import argparse, json, os, pathlib
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt

# --------------------------------------------------------------------------- #
#                           distance–based metrics                            #
# --------------------------------------------------------------------------- #

def _surface_mask(binary: np.ndarray) -> np.ndarray:
    """Return a boolean array marking the surface voxels of `binary`."""
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=bool)
    kernel = np.ones((3, 3, 3), dtype=bool)
    return binary & ~binary_erosion(binary, kernel, border_value=0)

def _surface_distances(seg_a: np.ndarray,
                       seg_b: np.ndarray,
                       spacing   : tuple[float, float, float]) -> np.ndarray:
    """
    Symmetric surface-to-surface distances A→B and B→A, in **mm**.

    If either object is empty, returns an empty array.
    """
    surface_a = _surface_mask(seg_a)
    surface_b = _surface_mask(seg_b)
    if surface_a.sum() == 0 or surface_b.sum() == 0:
        return np.array([])          # undefined → handled upstream

    # Distance transform of the *complement* surface
    dt_b = distance_transform_edt(~surface_b, sampling=spacing)
    dt_a = distance_transform_edt(~surface_a, sampling=spacing)

    dist_a2b = dt_b[surface_a]
    dist_b2a = dt_a[surface_b]
    return np.concatenate((dist_a2b, dist_b2a))


# --------------------------------------------------------------------------- #
#                             per-case evaluation                             #
# --------------------------------------------------------------------------- #

def _case_metrics(pred: np.ndarray,
                  ref : np.ndarray,
                  spacing: tuple[float, float, float]) -> dict[str, float]:
    """
    Compute all requested metrics for one prediction/ground-truth pair.
    """
    pred = pred.astype(bool)
    ref  = ref.astype(bool)

    # Confusion-matrix counts
    TP = int(np.logical_and(pred, ref).sum())
    FP = int(np.logical_and(pred, ~ref).sum())
    FN = int(np.logical_and(~pred, ref).sum())
    TN = int(np.logical_and(~pred, ~ref).sum())
    n_pred = TP + FP
    n_ref  = TP + FN

    # Overlap scores ---------------------------------------------------------
    if n_pred + n_ref == 0:          # both empty
        dice = 1.0
        iou  = 1.0
    else:
        dice = 2 * TP / (n_pred + n_ref) if (n_pred + n_ref) else 0.0
        iou  = TP / (TP + FP + FN)   if (TP + FP + FN)       else 0.0

    # Distance scores --------------------------------------------------------
    if n_pred == 0 and n_ref == 0:
        assd = hd95 = 0.0
    elif n_pred == 0 or  n_ref == 0:
        assd = hd95 = float("inf")   # degenerate case
    else:
        dists = _surface_distances(pred, ref, spacing)
        assd  = float(np.mean(dists))
        hd95  = float(np.percentile(dists, 95))

    return dict(ASSD=assd, Dice=dice, FN=FN, FP=FP, HD95=hd95,
                IoU=iou, TN=TN, TP=TP, n_pred=n_pred, n_ref=n_ref)


# --------------------------------------------------------------------------- #
#                              top-level driver                               #
# --------------------------------------------------------------------------- #

def _load_nifti(path: os.PathLike) -> tuple[np.ndarray, tuple[float,float,float]]:
    """Load image → boolean array, plus physical voxel spacing in mm."""
    img = nib.load(str(path))
    data = img.get_fdata() > 0                # binarise
    spacing = img.header.get_zooms()[:3]
    return data, spacing

def evaluate_folder(pred_dir: os.PathLike, ref_dir: os.PathLike) -> dict:
    pred_dir = pathlib.Path(pred_dir)
    ref_dir  = pathlib.Path(ref_dir)

    foreground_lists: dict[str, list] = {k: [] for k in
        ["ASSD","Dice","FN","FP","HD95","IoU","TN","TP","n_pred","n_ref"]}

    metric_per_case = []

    for pred_file in sorted(pred_dir.glob("*.nii*")):
        ref_file = ref_dir / pred_file.name
        if not ref_file.exists():
            raise FileNotFoundError(f"Missing reference for {pred_file.name}")

        pred_vol, spacing = _load_nifti(pred_file)
        ref_vol , _       = _load_nifti(ref_file)

        m = _case_metrics(pred_vol, ref_vol, spacing)

        # accumulate means ----------------------------------------------------
        for k in foreground_lists:
            foreground_lists[k].append(m[k])

        metric_per_case.append({
            "metrics": {"1": m},     # “class 1” → tumour
            "prediction_file": str(pred_file),
            "reference_file" : str(ref_file),
        })

    # foreground_mean == mean over *all* cases -------------------------------
    fg_mean = {k: float(np.mean(v)) for k, v in foreground_lists.items()}
    result  = {
        "foreground_mean": fg_mean,
        "mean"           : {"1": fg_mean},
        "metric_per_case": metric_per_case
    }
    return result


# --------------------------------------------------------------------------- #
#                                 CLI stub                                   #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate binary segmentations and write nnU-Net-style JSON.")
    ap.add_argument("--pred_dir", required=True,
                    help="Folder with predicted *.nii.gz volumes.")
    ap.add_argument("--ref_dir",  required=True,
                    help="Folder with ground-truth *.nii.gz volumes.")
    ap.add_argument("--out_json", default="metrics.json",
                    help="Path to output JSON file.")
    args = ap.parse_args()

    result = evaluate_folder(args.pred_dir, args.ref_dir)

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Metrics written to {args.out_json}")

if __name__ == "__main__":
    main()
