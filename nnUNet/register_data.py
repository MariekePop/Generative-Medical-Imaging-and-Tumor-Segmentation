import os
from glob import glob
import SimpleITK as sitk


def images_are_identical(fixed_image_path, moving_image_path):
    """
    Check if the fixed image and moving image have identical properties.

    Args:
        fixed_image_path (str): Path to the fixed image.
        moving_image_path (str): Path to the moving image.

    Returns:
        bool: True if images have identical properties, otherwise False.
    """
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)

    return (
        fixed_image.GetOrigin() == moving_image.GetOrigin() and
        fixed_image.GetSize() == moving_image.GetSize() and
        fixed_image.GetSpacing() == moving_image.GetSpacing() and
        fixed_image.GetDirection() == moving_image.GetDirection()
    )


def register_images(fixed_image_path, moving_image_path, output_path, interpolation=sitk.sitkLinear):
    """
    Registers the moving image to the fixed image and writes the registered image to the output path.

    Args:
        fixed_image_path (str): Path to the fixed image (reference image).
        moving_image_path (str): Path to the moving image to be registered.
        output_path (str): Path to save the registered image.
        interpolation: Interpolation method for resampling (default: sitk.sitkLinear).
    """
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform()))

    # Perform registration
    transform = registration_method.Execute(fixed_image, moving_image)

    # Resample the moving image to align with the fixed image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(interpolation)
    resampler.SetTransform(transform)
    registered_image = resampler.Execute(moving_image)

    # Save the registered image
    sitk.WriteImage(registered_image, output_path)
    print(f"Registered {moving_image_path} to {fixed_image_path} and saved to {output_path}")

def register_dataset(input_dir, output_dir, modalities, label_suffix=".nii.gz", fixed_modality_index=0):
    """
    Registers all modalities and ground truth labels in a structured dataset.

    Args:
        input_dir (str): Path to the directory containing the unregistered dataset (e.g., nnUNet_raw/Dataset250_Neck_Tumour).
        output_dir (str): Path to save the registered dataset (e.g., nnUNet_raw/Dataset250_Neck_Tumour_Registered).
        modalities (list): List of modality suffixes (e.g., ['_0000.nii.gz', '_0001.nii.gz']).
        label_suffix (str): File suffix for the ground truth labels (e.g., '.nii.gz').
        fixed_modality_index (int): Index of the fixed modality to use as reference.
    """
    os.makedirs(output_dir, exist_ok=True)

    # subfolders = ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']
    subfolders = ['labelsTr', 'labelsTs']
    fixed_modality_suffix = modalities[fixed_modality_index]

    for subfolder in subfolders:
        input_subfolder = os.path.join(input_dir, subfolder)
        output_subfolder = os.path.join(output_dir, subfolder)

        if not os.path.exists(input_subfolder):
            print(f"Warning: Subfolder {input_subfolder} does not exist. Skipping.")
            continue

        os.makedirs(output_subfolder, exist_ok=True)

        # Handle modalities (imagesTr/imagesTs)
        if 'images' in subfolder:
            cases = sorted(set("_".join(file.split("_")[:-1]) for file in os.listdir(input_subfolder) if file.endswith(fixed_modality_suffix)))
            for case in cases:
                print(f"Processing {subfolder}: {case}")

                fixed_image_path = os.path.join(input_subfolder, f"{case}{fixed_modality_suffix}")
                for modality in modalities:
                    moving_image_path = os.path.join(input_subfolder, f"{case}{modality}")
                    output_path = os.path.join(output_subfolder, f"{case}{modality}")
                    if modality == fixed_modality_suffix:
                        sitk.WriteImage(sitk.ReadImage(fixed_image_path), output_path)
                    else:
                        register_images(fixed_image_path, moving_image_path, output_path, interpolation=sitk.sitkLinear)

        # Handle labels (labelsTr/labelsTs)
        elif 'labels' in subfolder:
            cases = sorted(set(file.split(".")[0] for file in os.listdir(input_subfolder) if file.endswith(label_suffix)))
            for case in cases:
                print(f"Processing label for {case} in {subfolder}")

                label_path = os.path.join(input_subfolder, f"{case}{label_suffix}")
                label_output_path = os.path.join(output_subfolder, f"{case}{label_suffix}")

                fixed_image_path = os.path.join(input_dir, 'imagesTr', f"{case}_0001.nii.gz")
                if not os.path.exists(fixed_image_path):
                    print(f"Warning: Fixed image for {case} not found in imagesTr. Skipping label registration.")
                    continue

                # Register label
                if images_are_identical(fixed_image_path, label_path):
                    print(f"Skipping label registration for {label_path}: already aligned with {fixed_image_path}.")
                    sitk.WriteImage(sitk.ReadImage(label_path), label_output_path)
                else:
                    register_images(fixed_image_path, label_path, label_output_path, interpolation=sitk.sitkNearestNeighbor)

    print(f"Registration complete! Registered dataset saved to: {output_dir}")


def register_datasetOLD(input_dir, output_dir, modalities, label_suffix=".nii.gz", fixed_modality_index=0):
    """
    Registers all modalities and ground truth labels in a structured dataset.

    Args:
        input_dir (str): Path to the directory containing the unregistered dataset (e.g., nnUNet_raw/Dataset250_Neck_Tumour).
        output_dir (str): Path to save the registered dataset (e.g., nnUNet_raw/Dataset250_Neck_Tumour_Registered).
        modalities (list): List of modality suffixes (e.g., ['_0000.nii.gz', '_0001.nii.gz']).
        label_suffix (str): File suffix for the ground truth labels (e.g., '.nii.gz').
        fixed_modality_index (int): Index of the fixed modality to use as reference.
    """
    os.makedirs(output_dir, exist_ok=True)

    subfolders = ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']
    fixed_modality_suffix = modalities[fixed_modality_index]

    for subfolder in subfolders:
        input_subfolder = os.path.join(input_dir, subfolder)
        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)

        # Get all patient cases in the current subfolder
        cases = sorted(set("_".join(file.split("_")[:-1]) for file in os.listdir(input_subfolder) if file.endswith(fixed_modality_suffix)))

        for case in cases:
            print(f"Processing {subfolder}: {case}")

            # Fixed image for registration
            fixed_image_path = os.path.join(input_subfolder, f"{case}{fixed_modality_suffix}")

            # Register all modalities for the case
            for modality in modalities:
                moving_image_path = os.path.join(input_subfolder, f"{case}{modality}")
                output_path = os.path.join(output_subfolder, f"{case}{modality}")

                if modality == fixed_modality_suffix:
                    # Copy the fixed modality directly
                    sitk.WriteImage(sitk.ReadImage(fixed_image_path), output_path)
                else:
                    # Check if the moving image is already registered
                    if images_are_identical(fixed_image_path, moving_image_path):
                        print(f"Skipping registration for {moving_image_path}: already aligned with {fixed_image_path}.")
                        sitk.WriteImage(sitk.ReadImage(moving_image_path), output_path)
                        continue

                    # Register and save the modality
                    register_images(fixed_image_path, moving_image_path, output_path, interpolation=sitk.sitkLinear)

            # Register the label if the subfolder contains labels
            if 'labels' in subfolder:
                label_path = os.path.join(input_subfolder, f"{case}{label_suffix}")
                if os.path.exists(label_path):
                    label_output_path = os.path.join(output_subfolder, f"{case}{label_suffix}")
                    if images_are_identical(fixed_image_path, label_path):
                        print(f"Skipping registration for {label_path}: already aligned with {fixed_image_path}.")
                        sitk.WriteImage(sitk.ReadImage(label_path), label_output_path)
                    else:
                        register_images(fixed_image_path, label_path, label_output_path, interpolation=sitk.sitkNearestNeighbor)

    print(f"Registration complete! Registered dataset saved to: {output_dir}")


if __name__ == "__main__":
    # Define paths and modalities
    
    INPUT_DIR = "/home/rth/jdekok/thesis_folder/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour"
    OUTPUT_DIR = "/home/rth/jdekok/thesis_folder/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour_Registered"
    MODALITIES = ['_0000.nii.gz', '_0001.nii.gz', '_0002.nii.gz', '_0003.nii.gz', '_0004.nii.gz', '_0005.nii.gz']  # Update based on your dataset
    LABEL_SUFFIX = '.nii.gz'  # File suffix for labels (default for nnUNet)
    FIXED_MODALITY_INDEX = 1  # Use the second modality T1C (_0001.nii.gz) as the fixed modality

    # Run the dataset registration
    register_dataset(INPUT_DIR, OUTPUT_DIR, MODALITIES, LABEL_SUFFIX, FIXED_MODALITY_INDEX)
