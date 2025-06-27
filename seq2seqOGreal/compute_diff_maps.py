import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

# ------------------ CONFIG ------------------
GT_FOLDER = r"C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_raw\Dataset520_NeckTumour\imagesTs"
PRED_FOLDER = r"C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_predictions"
OUT_FOLDER = r"C:\Users\P095789\Downloads\seq2seqOG\difference_maps"

os.makedirs(OUT_FOLDER, exist_ok=True)

def normalize(img):
    return np.clip((img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8), 0, 1)

def load_nifti(path):
    return nib.load(path)

def save_nifti(data, ref_img, out_path):
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref_img.affine), out_path)

summary = []

for patient_id in sorted(os.listdir(PRED_FOLDER)):
    pred_seq_dir = os.path.join(PRED_FOLDER, patient_id, "multi2one_inference")
    if not os.path.isdir(pred_seq_dir):
        continue

    for seq in range(6):
        pred_path = os.path.join(pred_seq_dir, f"translate_tgt_{seq}.nii.gz")
        gt_path = os.path.join(GT_FOLDER, f"{patient_id}_{seq:04d}.nii.gz")

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            print(f"Skipping missing: {patient_id} seq {seq}")
            continue

        pred_img = load_nifti(pred_path).get_fdata().astype(np.float32)
        gt_nib = load_nifti(gt_path)
        gt_img = gt_nib.get_fdata().astype(np.float32)

        pred_norm = normalize(pred_img)
        gt_norm = normalize(gt_img)

        signed_diff = pred_norm - gt_norm
        abs_diff = np.abs(signed_diff)

        # Save absolute diff as NIfTI
        out_name = f"{patient_id}_seq{seq}_diff.nii.gz"
        save_nifti(abs_diff, gt_nib, os.path.join(OUT_FOLDER, out_name))

        # Plot mid slice using global color scale
        mid = signed_diff.shape[2] // 2
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.imshow(gt_norm[:, :, mid], cmap='gray')
        plt.title("GT")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_norm[:, :, mid], cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(signed_diff[:, :, mid], cmap='bwr', vmin=-1.0, vmax=1.0)
        plt.title("Difference Map")
        plt.colorbar(shrink=0.6)
        plt.axis('off')

        
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_FOLDER, f"{patient_id}_seq{seq}_vis.png"), dpi=200)
        plt.close()

        summary.append({
            "Patient": patient_id,
            "Sequence": seq,
            "Mean Abs Diff": np.mean(abs_diff),
            "Std Abs Diff": np.std(abs_diff)
        })

# ------------------ SUMMARY ------------------
df = pd.DataFrame(summary)
print("\n=== Difference Map Summary ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT_FOLDER, "difference_summary.csv"), index=False)
