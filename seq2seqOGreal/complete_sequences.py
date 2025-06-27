import os
import shutil
from collections import defaultdict

# ---- CONFIG ----
original_dir = r"C:\Users\P095789\Downloads\mr_subset_resampled\imagesTr"
generated_base = r"C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_pred_extra_data"
output_dir = r"C:\Users\P095789\Downloads\mr_subset_completed\imagesTr"

os.makedirs(output_dir, exist_ok=True)

expected_modalities = [f"{i:04d}" for i in range(6)]  # 0000 to 0005

# ---- Index all original files by patient ----
original_files = os.listdir(original_dir)
patients = defaultdict(set)

for fname in original_files:
    if fname.endswith(".nii.gz") and "_" in fname:
        pid, mod = fname.replace(".nii.gz", "").split("_")
        patients[pid].add(mod)
        # Copy all existing files to output
        src = os.path.join(original_dir, fname)
        dst = os.path.join(output_dir, fname)
        shutil.copy2(src, dst)
        print(f"Copied original: {fname}")

# ---- Fill missing modalities using generated files ----
for pid, mods in patients.items():
    for idx, mod in enumerate(expected_modalities):
        if mod not in mods:
            gen_fname = f"translate_tgt_{idx}.nii.gz"
            gen_path = os.path.join(generated_base, pid, "multi2one_inference", gen_fname)
            out_fname = f"{pid}_{mod}.nii.gz"
            out_path = os.path.join(output_dir, out_fname)

            if os.path.exists(gen_path):
                shutil.copy2(gen_path, out_path)
                print(f"Filled missing with generated: {out_fname}")
            else:
                print(f"Missing both original and generated: {out_fname}")

print("\nDone: Completed folder with original and correctly renamed generated files.")
