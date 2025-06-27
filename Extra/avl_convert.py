import os
import dicom2nifti
import pydicom
import nibabel as nib
import numpy as np
import shutil
from rt_utils import RTStructBuilder

# --- CONFIGURE PATHS ---
dicom_main_folder = r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\AVL_OPSCC"  # Change this
output_folder = r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\Nifti_Output"  # Change this
label_output_folder = r"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\Labels_Output"  # Change this

# Define sequence mapping for nnUNet
sequence_mapping = {
    "T1": "0000",
    "T1C": "0001",
    "T2": "0002",
    "STIR": "0003",
    "DWI": "0004",
    "ADC": "0005"
}

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(label_output_folder, exist_ok=True)

# --- Step 1: Loop through all patient folders ---
for index, patient_folder in enumerate(sorted(os.listdir(dicom_main_folder)), start=89):
    patient_path = os.path.join(dicom_main_folder, patient_folder)

    if not os.path.isdir(patient_path):
        continue  # Skip non-folder files

    patient_id = f"{index:03d}"  # Start numbering from 089, 090, etc.
    print(f"Processing Patient: {patient_id} ({patient_folder})")

    # Store found sequences
    mri_files = {}
    rtstruct_path = None

    # --- Step 2: Find MRI Sequences and RTSTRUCT ---
    for root, _, files in os.walk(patient_path):
        folder_name = os.path.basename(root).upper()

        if folder_name not in sequence_mapping:
            continue

        for file in files:
            dicom_path = os.path.join(root, file)

            try:
                dcm = pydicom.dcmread(dicom_path)
                if hasattr(dcm, "SeriesDescription") and folder_name in sequence_mapping:
                    if folder_name not in mri_files:
                        mri_files[folder_name] = root  # Store the sequence folder
                
                if dcm.Modality == "RTSTRUCT":
                    print(f"📂 Found RTSTRUCT: {dicom_path} (Series Description: {getattr(dcm, 'SeriesDescription', 'UNKNOWN')})")
                    rtstruct_path = dicom_path  # Use the first RTSTRUCT found

            except Exception as e:
                print(f"Skipping {file}: {e}")

    # --- Step 3: Convert MRI Sequences to NIfTI ---
    for key, dicom_folder in mri_files.items():
        output_filename = f"{patient_id}_{sequence_mapping[key]}.nii.gz"
        output_path = os.path.join(output_folder, output_filename)

        try:
            temp_nifti_folder = os.path.join(output_folder, "temp")
            if os.path.exists(temp_nifti_folder):
                shutil.rmtree(temp_nifti_folder)
            os.makedirs(temp_nifti_folder, exist_ok=True)
            
            dicom2nifti.convert_directory(dicom_folder, temp_nifti_folder, compression=True)
            
            for file in os.listdir(temp_nifti_folder):
                if file.endswith(".nii.gz"):
                    shutil.move(os.path.join(temp_nifti_folder, file), output_path)
                    print(f"✅ Saved MRI: {output_filename}")

            shutil.rmtree(temp_nifti_folder)

        except Exception as e:
            print(f"❌ Error converting {key}: {e}")

    # --- Step 4: Convert RTSTRUCT to NIfTI ---
    if rtstruct_path:
        try:
            print(f"📂 Processing RTSTRUCT: {rtstruct_path}")
            rtstruct = RTStructBuilder.create_from(patient_path, rtstruct_path)
            
            structures = rtstruct.get_roi_names()
            print(f"🔹 Found structures in RTSTRUCT: {structures}")

            chosen_label = None
            if "Tumor T1" in structures:
                chosen_label = "Tumor T1"
            elif "Tumor STIR" in structures:
                chosen_label = "Tumor STIR"

            if chosen_label:
                print(f"✅ Using label: {chosen_label}")
                label_map = rtstruct.get_as_numpy_mask(chosen_label)
                label_nifti_path = os.path.join(label_output_folder, f"{patient_id}.nii.gz")
                
                affine = np.eye(4)
                label_nii = nib.Nifti1Image(label_map.astype(np.uint8), affine)
                nib.save(label_nii, label_nifti_path)
                print(f"✅ Saved Label: {patient_id}.nii.gz")
            else:
                print(f"❌ No valid segmentation found in RTSTRUCT for {patient_id}")

        except Exception as e:
            print(f"❌ Error processing RTSTRUCT: {e}")

print("✅ Finished processing all patients!")
