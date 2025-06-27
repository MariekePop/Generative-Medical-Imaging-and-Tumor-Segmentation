import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import lpips
from collections import defaultdict

# -------------------- CONFIG --------------------
GT_FOLDER = r"C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_raw\Dataset520_NeckTumour\imagesTs"
PRED_FOLDER = r"C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_predictions"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- SETUP --------------------
loss_fn_lpips = lpips.LPIPS(net='alex').to(device)

def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)

def match_prediction_scale(pred, gt):
    """Rescale prediction to match ground truth dynamic range."""
    gt_max = np.max(gt)
    pred_max = np.max(pred)
    if gt_max > 0 and pred_max > 0:
        return pred * (gt_max / pred_max)
    return pred

def normalize(img):
    """Normalize to [0, 1] range based on global min/max."""
    return np.clip((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8), 0, 1)

def compute_metrics(gt_img, pred_img):
    assert gt_img.shape == pred_img.shape

    # Align scales
    pred_img = match_prediction_scale(pred_img, gt_img)

    # Normalize for structural/perceptual metrics
    gt_img_norm = normalize(gt_img)
    pred_img_norm = normalize(pred_img)

    ssim_scores, psnr_scores, l1_scores, lpips_scores = [], [], [], []

    for i in range(gt_img.shape[2]):  # axial slices
        gt_slice = gt_img_norm[:, :, i]
        pred_slice = pred_img_norm[:, :, i]

        ssim_scores.append(ssim(gt_slice, pred_slice, data_range=1.0))
        psnr_scores.append(psnr(gt_img[:, :, i], pred_img[:, :, i], data_range=np.max(gt_img)))  # raw scale
        l1_scores.append(np.mean(np.abs(gt_img[:, :, i] - pred_img[:, :, i])))  # MPa on raw data

        # LPIPS input: normalized [0,1] → [-1,1], 3 channels
        gt_tensor = torch.from_numpy(gt_slice).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        pred_tensor = torch.from_numpy(pred_slice).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        lpips_val = loss_fn_lpips(gt_tensor * 2 - 1, pred_tensor * 2 - 1).item()
        lpips_scores.append(lpips_val)

    return (
        np.mean(ssim_scores),
        np.mean(psnr_scores),
        np.mean(l1_scores),    # MPa
        np.mean(lpips_scores)  # LPIPS
    )

# -------------------- METRIC STORAGE --------------------
all_ssim, all_psnr, all_mpa, all_lpips = [], [], [], []
metrics_by_seq = defaultdict(lambda: {"ssim": [], "psnr": [], "mpa": [], "lpips": []})

# -------------------- EVALUATION --------------------
print("Evaluating predictions...\n")
for patient_id in sorted(os.listdir(PRED_FOLDER)):
    pred_seq_dir = os.path.join(PRED_FOLDER, patient_id, "multi2one_inference")
    if not os.path.isdir(pred_seq_dir):
        continue

    for seq in range(6):  # sequence indices: 0 to 5
        pred_path = os.path.join(pred_seq_dir, f"translate_tgt_{seq}.nii.gz")
        gt_path = os.path.join(GT_FOLDER, f"{patient_id}_{seq:04d}.nii.gz")

        if not os.path.isfile(pred_path) or not os.path.isfile(gt_path):
            print(f"Skipping missing file: {pred_path} or {gt_path}")
            continue

        gt_img = load_nifti(gt_path)
        pred_img = load_nifti(pred_path)

        ssim_val, psnr_val, mpa_val, lpips_val = compute_metrics(gt_img, pred_img)

        # Store overall
        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_mpa.append(mpa_val)
        all_lpips.append(lpips_val)

        # Store by sequence
        metrics_by_seq[seq]["ssim"].append(ssim_val)
        metrics_by_seq[seq]["psnr"].append(psnr_val)
        metrics_by_seq[seq]["mpa"].append(mpa_val)
        metrics_by_seq[seq]["lpips"].append(lpips_val)

        print(f"Patient {patient_id} | Seq {seq} → "
              f"SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, "
              f"MPa: {mpa_val:.4f}, LPIPS: {lpips_val:.4f}")

# -------------------- SUMMARY --------------------
def summarize(metric_list, name, units=""):
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    print(f"{name:<6}: {mean:.4f} ± {std:.4f} {units}")

print("\n=== Overall Results ===")
summarize(all_ssim, "SSIM")
summarize(all_psnr, "PSNR", "dB")
summarize(all_mpa, "MPa")
summarize(all_lpips, "LPIPS")

print("\n=== Per-Sequence Results ===")
for seq in sorted(metrics_by_seq.keys()):
    print(f"\n[ Sequence {seq} ]")
    summarize(metrics_by_seq[seq]["ssim"], "SSIM")
    summarize(metrics_by_seq[seq]["psnr"], "PSNR", "dB")
    summarize(metrics_by_seq[seq]["mpa"], "MPa")
    summarize(metrics_by_seq[seq]["lpips"], "LPIPS")
