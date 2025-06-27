import os
import SimpleITK as sitk

# --- CONFIG ---
label_dir = r"C:\Users\P095789\Downloads\mr_subset_resampled\labelsTr"
fixed_label_dir = r"C:\Users\P095789\Downloads\mr_subset_completed\labelsTr_fixed"
os.makedirs(fixed_label_dir, exist_ok=True)

# --- FIX LABELS ---
for fname in os.listdir(label_dir):
    if not fname.endswith(".nii.gz"):
        continue

    path = os.path.join(label_dir, fname)
    label = sitk.ReadImage(path)
    label_array = sitk.GetArrayFromImage(label)

    # Replace 255 with 1
    label_array[label_array == 255] = 1

    fixed_label = sitk.GetImageFromArray(label_array)
    fixed_label.CopyInformation(label)  # keep spacing/origin/direction

    out_path = os.path.join(fixed_label_dir, fname)
    sitk.WriteImage(fixed_label, out_path)
    print(f"✅ Fixed label: {fname}")

print("\n🎯 All label files converted to binary [0, 1].")
