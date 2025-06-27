import os
import SimpleITK as sitk
from collections import defaultdict

# ---- CONFIG ----
base_in = r"C:\Users\P095789\Downloads\mr_subset"
base_out = r"C:\Users\P095789\Downloads\mr_subset_resampled"

target_size = [560, 560, 29]  # X, Y, Z (nnSeq2Seq expects Z last!)
images_in = os.path.join(base_in, "imagesTr")
labels_in = os.path.join(base_in, "labelsTr")
images_out = os.path.join(base_out, "imagesTr")
labels_out = os.path.join(base_out, "labelsTr")

os.makedirs(images_out, exist_ok=True)
os.makedirs(labels_out, exist_ok=True)

# ---- HELPERS ----
def compute_spacing(image, target_size):
    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()
    return [orig_spacing[i] * orig_size[i] / target_size[i] for i in range(3)]

def _resample_single_volume(img, size, spacing, is_label, reference_image):
    resample = sitk.ResampleImageFilter()
    resample.SetSize(size)
    resample.SetOutputSpacing(spacing)
    resample.SetOutputDirection(reference_image.GetDirection())
    resample.SetOutputOrigin(reference_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resample.Execute(img)

def resample_image(image, target_size, spacing, reference_image, is_label=False):
    if len(image.GetSize()) == 4:
        print("Found 4D image, reducing to first channel only...")
        # Reduce to first channel (assuming channel last)
        image = image[:, :, :, 0]  # Use only first volume
        if image.GetPixelID() == sitk.sitkUInt16:
            image = sitk.Cast(image, sitk.sitkFloat32)
        return _resample_single_volume(image, target_size, spacing, is_label, reference_image)
    else:
        if image.GetPixelID() == sitk.sitkUInt16:
            image = sitk.Cast(image, sitk.sitkFloat32)
        return _resample_single_volume(image, target_size, spacing, is_label, reference_image)

# ---- Group filenames by patient ----
def group_by_patient(filenames):
    grouped = defaultdict(list)
    for f in filenames:
        if f.endswith(".nii.gz"):
            patient_id = f.split("_")[0]
            grouped[patient_id].append(f)
    return grouped

# ---- RESAMPLE IMAGES ----
print("Resampling images...")
grouped_images = group_by_patient(os.listdir(images_in))

for patient, files in grouped_images.items():
    try:
        ref_path = os.path.join(images_in, sorted(files)[0])
        ref_img = sitk.ReadImage(ref_path)
        spacing = compute_spacing(ref_img, target_size)

        for fname in sorted(files):
            in_path = os.path.join(images_in, fname)
            out_path = os.path.join(images_out, fname)
            print(fname)
            img = sitk.ReadImage(in_path)
            resampled = resample_image(img, target_size, spacing, ref_img, is_label=False)
            sitk.WriteImage(resampled, out_path)
            print(f"Image: {fname}")
    except Exception as e:
        print(f"Patient {patient} failed: {e}")

print("\nResampling labels...")
for fname in sorted(os.listdir(labels_in)):
    if not fname.endswith(".nii.gz"):
        continue

    patient_id = fname.split(".")[0]  # e.g. '078'
    label_path = os.path.join(labels_in, fname)
    label_out_path = os.path.join(labels_out, fname)

    try:
        # Try to find any image for this patient to use as reference
        matching_images = [f for f in os.listdir(images_in) if f.startswith(f"{patient_id}_")]
        if not matching_images:
            print(f"Skipping label {fname}: no image found.")
            continue

        ref_img_path = os.path.join(images_in, sorted(matching_images)[0])
        ref_img = sitk.ReadImage(ref_img_path)
        spacing = compute_spacing(ref_img, target_size)

        lbl = sitk.ReadImage(label_path)
        resampled_lbl = resample_image(lbl, target_size, spacing, ref_img, is_label=True)
        sitk.WriteImage(resampled_lbl, label_out_path)
        print(f"Label: {fname}")
    except Exception as e:
        print(f"Label failed ({fname}): {e}")


print("\nDone! All images and labels resampled to (560, 560, 29)")
