#!/usr/bin/env python3
"""
make_black_niftis.py
--------------------
Create a set of nnU-Net-style NIfTI files whose voxel data is all zeros
but whose header & affine match the originals.
"""
import argparse, pathlib
import nibabel as nib
import numpy as np

def main(src: pathlib.Path, dst: pathlib.Path):
    dst.mkdir(parents=True, exist_ok=True)
    nifti_files = sorted(src.glob("*.nii.gz"))

    if not nifti_files:
        raise RuntimeError("No .nii.gz files found in source folder!")

    for f in nifti_files:
        img = nib.load(str(f))
        zero_data = np.zeros(img.shape, dtype=img.get_data_dtype())
        black_img = nib.Nifti1Image(zero_data, img.affine, img.header)

        out_path = dst / f.name
        nib.save(black_img, str(out_path))
        print(f"✓ {out_path.name}")

    print(f"\n✅  {len(nifti_files)} black NIfTIs written to {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Folder with original nnU-Net images (imagesTr or imagesTs)")
    ap.add_argument("--dst", required=True,
                    help="Output folder for black images")
    args = ap.parse_args()

    main(pathlib.Path(args.src), pathlib.Path(args.dst))
