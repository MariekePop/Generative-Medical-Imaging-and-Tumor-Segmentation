import os
import json
from glob import glob
from sklearn.model_selection import train_test_split

# === CONFIG ===
image_dir = r"C:/Users/P095789/OneDrive - Amsterdam UMC/Documenten/NestedFormer-main/NestedFormer_preprocessed"
output_json = "headneck_datajson.json"
num_modalities = 6
label_dir = r"C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_raw\Dataset520_NeckTumour\labelsTr"  # adjust if you have a separate label folder

# === Scan all files ===
patients = {}
for file in sorted(glob(os.path.join(image_dir, "*.nii*"))):
    basename = os.path.basename(file)
    patient_id, modality = basename.split("_")
    modality_idx = int(modality.replace(".nii.gz", "").replace(".nii", ""))

    if patient_id not in patients:
        patients[patient_id] = [""] * num_modalities
    patients[patient_id][modality_idx] = file.replace("\\", "/")

# === Label path assumption ===
def guess_label_path(pid):
    # Adjust this if your label structure differs
    return os.path.join(label_dir, f"{pid}.nii.gz").replace("\\", "/")

# === Build entries ===
cases = []
for pid, modalities in patients.items():
    if all(modalities):  # Ensure all 6 modalities are present
        label = guess_label_path(pid)
        if os.path.exists(label):
            cases.append({
                "image": modalities,
                "label": label
            })

# === Split train/val
train_set, val_set = train_test_split(cases, test_size=15, random_state=42)

# === Write JSON ===
json_data = {
    "name": "HeadNeckTumor",
    "description": "Auto-generated from preprocessed dataset",
    "training": train_set,
    "validation": val_set
}
with open(output_json, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"✅ Saved {output_json} with {len(train_set)} train / {len(val_set)} val cases.")
