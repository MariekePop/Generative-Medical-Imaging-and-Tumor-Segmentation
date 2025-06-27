import os
import numpy as np
import nibabel as nib

# Configuration
preprocessed_npz_dir  = r"C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_preprocessed\Dataset520_NeckTumour\nnSeq2SeqPlans_3d"  # Folder with .npz files
original_nii_dir  = r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_raw\Dataset520_NeckTumour\imagesTr"   # Folder with original .nii.gz files
output_dir = r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\NestedFormer-main\NestedFormer_preprocessed"   # Where to save converted .nii.gz files
os.makedirs(output_dir, exist_ok=True)

TARGET_SHAPE = (560, 560, 29)

for fname in sorted(os.listdir(preprocessed_npz_dir)):
    if not fname.endswith(".npz"):
        continue

    npz_path = os.path.join(preprocessed_npz_dir, fname)
    patient_id = fname.replace(".npz", "")
    print(f"📦 Converting {patient_id}...")

    npz = np.load(npz_path)
    data = npz["data"]  # shape: (C, D, H, W)

    num_channels = data.shape[0]

    for i in range(num_channels):
        original_nii_fname = f"{patient_id}_{i:04d}.nii.gz"
        original_nii_path = os.path.join(original_nii_dir, original_nii_fname)

        if not os.path.exists(original_nii_path):
            print(f"  ⚠️ Skipping missing original: {original_nii_fname}")
            continue

        affine = nib.load(original_nii_path).affine
        cropped_vol = data[i]  # shape: (D, H, W)

        # Step 1: Transpose to (H, W, D)
        volume = np.transpose(cropped_vol, (1, 2, 0))  # from (D, H, W)

        # Step 2: Flip Y and X *only* (Z is fine)
        volume = np.flip(volume, axis=0)  # Flip Y (up/down in sagittal/coronal)

        # Step 3: Rotate to fix the switched axes view
        volume = np.rot90(volume, k=3, axes=(0, 1))  # rotate -90° in-plane

        # Step 4: Pad to target shape
        padded = np.zeros(TARGET_SHAPE, dtype=np.float32)
        h, w, d = volume.shape
        th, tw, td = TARGET_SHAPE

        if h > th or w > tw or d > td:
            print(f"  ❌ Volume shape {volume.shape} is larger than target {TARGET_SHAPE}")
            continue

        sh = (th - h) // 2
        sw = (tw - w) // 2
        sd = (td - d) // 2

        padded[sh:sh + h, sw:sw + w, sd:sd + d] = volume

        out_nifti = nib.Nifti1Image(padded, affine)
        out_path = os.path.join(output_dir, f"{patient_id}_{i:04d}.nii.gz")
        nib.save(out_nifti, out_path)
        print(f"  ✅ Saved: {out_path}")