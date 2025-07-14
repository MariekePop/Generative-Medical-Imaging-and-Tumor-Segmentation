# Generative Medical Imaging and Tumor Segmentation

This repository accompanies the MSc Artificial Intelligence thesis:

**Generating Missing MRI Sequences to Enable Clinically Robust Head and Neck Tumor Segmentation**\
**Author:** Fien de Kok\
**Institution:** University of Amsterdam / Amsterdam UMC (2025)

---

## Overview

This code supports experiments conducted during the thesis project focused on leveraging generative AI to address missing MRI sequences in head and neck cancer imaging. The pipeline combines:

- `nnU-Net` for tumor segmentation
- `Seq2Seq` for missing sequence synthesis
- `NestedFormer` (experimental) for Transformer-based segmentation

---

## Repository Structure

```bash
Generative-Medical-Imaging-and-Tumor-Segmentation/
├── Extra/            # Unordened files in which things were written to remember specifically for my thesis including notes on the usage of the Luna cluster.
├── NestedFormer-main/       # Experimental NestedFormer files
├── p44scans/            # Example outputs, graphs, difference maps and figures from the thesis
├── nnUNet/             # nnU-Net model files plus jobfiles, output files and registering data code.
├── nnUNet_results & nnUNet_with_emptyblack_experiments/       # Results, jobfiles, output files and more from Luna cluster for both original and empty input experiments. Some files will be the same in both folders, but for completeness they both contain all info plus the files specifically for their own experiments
├── seq2seqOGreal/            # Seq2Seq model files plus jobfiles, output files and all files needed to convert extra data to usable format
├── README.md           # You are here!
```

---
Please note that due to privacy concerns no data is made available

## Installation (General Setup)

> Note: Specific instructions per model are found in their respective sections below.

1. Clone this repository:

```bash
git clone https://github.com/MariekePop/Generative-Medical-Imaging-and-Tumor-Segmentation.git
cd Generative-Medical-Imaging-and-Tumor-Segmentation
```

2. Create environment:
3. Please see the specific .yml file(names) in their respective folders

```bash
conda env create -f environment.yml
conda activate "env"
```

---

# nnU-Net

### Original Repo

- GitHub: [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)

### Folder: `nnunet/`

Contains code for training and evaluating segmentation models on 3D multi-channel MRI data using `nnU-Net v2`.

### Environment Setup

```bash
conda create -n nnunetv2 python=3.9
conda activate nnunetv2
pip install torch
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

See [nnUNet Installation Guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation.md) for system-specific steps.


Alternatively use our included environment:

```bash
conda env create -f nnunett.yml
conda activate nnunetv2
```

### Files in Folder besides the original nnUNet files

- `fold_0_only` – Completge Luna files of original nnUNet training which includes all .json files
- `register_data.py` – Code to register data
- `.job & .out files` – The job and output files

### How to Use
For additional commands we recoomend you to look at the .job files in their respective folders

1. **Fingerprinting & Planning (optional if not reconfiguring):**

```bash
nnUNetv2_plan_and_preprocess -d 520 -i ./images -o ./preprocessed
```

2. **Train Baseline Model (1000 epochs):**

```bash
nnUNetv2_train 520 3d_fullres
```

3. **Inference:**

```bash
nnUNetv2_predict -d 520 -f 0 -c 3d_fullres -i /input/folder -o /output/folder
```

4. **Evaluate Predictions:**

```bash
python results_eval/eval_segmentation.py --pred_dir /output/folder --gt_dir /ground_truths
```

### Outputs

- Segmentation masks (Nifti format)
- Dice, IoU, ASSD, HD95 metrics

---

# Seq2Seq – Sequence Synthesis

### Original Repo

- GitHub: [https://github.com/MedARC-AI/Seq2Seq](https://github.com/MedARC-AI/Seq2Seq)

### Folder: `seq2seq/`

Code for training a generative model that synthesizes missing MRI modalities from present sequences. Based on the 2D Seq2Seq from [Han et al., 2023].

### Environment Setup

```bash
conda create -n nnseq2seq python=3.10
conda activate nnseq2seq
pip install torch torchvision torchaudio
git clone https://github.com/fiy2W/mri_seq2seq.git
cd mri_seq2seq
pip install -e 
```
See [seq2seq Installation Guide](https://github.com/fiy2W/mri_seq2seq/blob/main/nnseq2seq/docs/installation_instructions.md) for system-specific steps.


Alternatively use our included environment:

```bash
conda env create -f nnseq2seq.yml
conda activate nnseq2seq
```

### Files in Folder besides the original seq2seq files

- `.job & .out files` – The job and output files
- `Seq2seq_test_real_or_gen` Powerpoint with scans used for the blinded visual test
- `compute_diff_maps` Code to make the difference maps
- `complete_sequences.py` Make one folder with all real scans filled with the generated scans from the seq2seq folder
- `evaluate_metrics.py` Calculate metrics based on predictions
- `labelsto01` Not all data has binary labels, this code makes them binary to be compatible with our implementations
- `resample_all_to_560x660x29.py` resampling extra data, might need changes when used on other extra data

### How to Use
For additional commands we recoomend you to look at the .job files in their respective folders

1. **Train Seq2Seq (1000 epochs):**

```bash
python train_seq2seq.py --config configs/seq2seq_config.json
```

2. **Test Model on Holdout Patients:**

```bash
python test_seq2seq.py --checkpoint path/to/checkpoint.pth
```

3. **Visualize Differences:**

```bash
python utils/plot_difference_maps.py --real /real/scans --pred /predicted/scans
```

### Outputs

- Generated MRI sequences (Nifti)
- SSIM, PSNR, LPIPS, Pixel Accuracy metrics
- Difference maps

---

# NestedFormer (Experimental Transformer Model)

### Original Repo

- GitHub: [https://github.com/MedARC-AI/NestedFormer](https://github.com/MedARC-AI/NestedFormer) (similar architecture, not official)

### Folder: `nestedformer/`

Transformer-based segmentation model using modality-specific encoders and cross-attention for MRI fusion.

### Environment Setup

```bash
conda create -n nestedformer python=3.9
conda activate nestedformer
pip install monai
pip install tqdm
pip install tensorboardX
```
See [nestedformer Installation Guide](https://github.com/fiy2W/mri_seq2seq/blob/main/nnseq2seq/docs/installation_instructions.md) for system-specific steps.


Alternatively use our included environment:

```bash
conda env create -f nestedformer.yml
conda activate nestedformer
```

### Files in Folder

- `runs` – Multiple training attempts
- `nestedformer_model files` – Please start from scratch to avoid using code with possible flaws as results stayed suspiciously low

### How to Use

```bash
python train_nestedformer.py --config configs/nestedformer_config.json
```

Note: Validation Dice plateaued around 34%. Model not included in final analysis. For detailed methodology, see Appendix A in the thesis.

---



