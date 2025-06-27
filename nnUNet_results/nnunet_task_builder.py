# nnunet_task_builder.py
"""
Create 14 nnU-Net Task folders with different real ↔ generated mixes.
Generated files are named ..._0000g.nii.gz etc. in their source directory,
but inside each Task they are linked/ copied as ..._0000.nii.gz (nnU-Net spec).
A provenance.tsv is emitted per Task for easy inspection.
"""

import csv, json, os, random, shutil, sys
from pathlib import Path

# ────────────────  USER  SETTINGS  ─────────────────────────────────────────────
REAL_DIR   = Path(r"C:\Users\P095789\Downloads\seq2seqOGreal\nnSeq2Seq_raw\Dataset520_NeckTumour\imagesTs")       # originals: 000_0000.nii.gz …
GEN_DIR    = Path(r"C:\Users\P095789\Downloads\nnunet_imagesTr2")        # generated: 000_0000g.nii.gz …
LABELS_DIR = Path(r"C:\Users\P095789\Downloads\seq2seqOGreal\nnSeq2Seq_raw\Dataset520_NeckTumour\labelsTs")            # one label per case: 000.nii.gz …
DST_ROOT   = Path(r"C:\Users\P095789\Downloads\nnunet_imagesTr2\nnUNet_combinations")         # Task5xx_* will be created here

LINK       = shutil.copy2   # ← choose: shutil.copy2   |   os.link   |   os.symlink
SEED       = 42             # for reproducible random subsets
# ───────────────────────────────────────────────────────────────────────────────

# channel → human-readable name (keep consistent with your pipeline)
CHANS = {0: "T1", 1: "T1C", 2: "T2", 3: "STIR", 4: "DWI", 5: "ADC"}

# sanity checks on source trees
for p in (REAL_DIR, GEN_DIR, LABELS_DIR):
    if not p.exists():
        sys.exit(f"❌  Source folder not found: {p}")

# all case IDs present in real set
CASES = sorted(p.name[:3] for p in REAL_DIR.glob("*_0000.nii*"))
if not CASES:
    sys.exit("❌  No cases found in REAL_DIR")

random.seed(SEED)

# helper to choose subsets
def pick(frac):  # frac = 0.30, 0.60, …
    return set(random.sample(CASES, round(frac * len(CASES))))

# pre-compute the 30 / 60 % splits
T2_30, STIR_30 = pick(0.30), pick(0.30)
T1_30, T1C_30  = pick(0.30), pick(0.30)
DWI_60         = pick(0.60)

# ----------------------------------------------------------------------------- #
#                       Task definitions  (14 experiments)                      #
# ----------------------------------------------------------------------------- #
# Each triple = (TaskID, short_tag, rule(case_id, channel) → "real"|"gen")
exp = [
(541, "allGen",
      lambda c,k: "gen"),
(542, "T1Gen",
      lambda c,k: "gen" if k==0 else "real"),
(543, "T1CGen",
      lambda c,k: "gen" if k==1 else "real"),
(544, "T2Gen",
      lambda c,k: "gen" if k==2 else "real"),
(545, "STIRGen",
      lambda c,k: "gen" if k==3 else "real"),
(546, "DWIGen",
      lambda c,k: "gen" if k==4 else "real"),
(547, "ADCGen",
      lambda c,k: "gen" if k==5 else "real"),
(548, "DWI_ADC_Gen",
      lambda c,k: "gen" if k in (4,5) else "real"),
(551, "30T2_30STIR",
      lambda c,k: "gen" if (k==2 and c in T2_30) or (k==3 and c in STIR_30) else "real"),
(552, "30T2_30STIR_ALLDWIADC",
      lambda c,k: "gen" if (k in (4,5) or
                             (k==2 and c in T2_30) or
                             (k==3 and c in STIR_30)) else "real"),
(553, "30T2_30STIR_60DWIADC",
      lambda c,k: "gen" if ((k in (4,5) and c in DWI_60) or
                             (k==2 and c in T2_30) or
                             (k==3 and c in STIR_30)) else "real"),
(554, "30T1_30T1C",
      lambda c,k: "gen" if (k==0 and c in T1_30) or (k==1 and c in T1C_30) else "real"),
(555, "30T1_30T1C_60DWIADC",
      lambda c,k: "gen" if ((k in (4,5) and c in DWI_60) or
                             (k==0 and c in T1_30)  or
                             (k==1 and c in T1C_30)) else "real"),
(556, "30T1_30T1C_60DWIADC_30T2_30STIR",
      lambda c,k: "gen" if ((k in (4,5) and c in DWI_60) or
                             (k==0 and c in T1_30)  or
                             (k==1 and c in T1C_30) or
                             (k==2 and c in T2_30)  or
                             (k==3 and c in STIR_30)) else "real")
]

# ------------------------------------------------------------------------------
def link_or_copy(src: Path, dst: Path):
    """Wrapper so we can swap out the strategy in one place."""
    if LINK is shutil.copy2:
        shutil.copy2(src, dst)
    else:
        # create hard- or soft-link; if already exists, replace
        if dst.exists(): dst.unlink()
        LINK(src, dst)

def make_task(task_id: int, tag: str, rule):
    task_dir = DST_ROOT / f"Task{task_id}_{tag}"
    imgT     = task_dir / "imagesTr"
    lblT     = task_dir / "labelsTr"

    if task_dir.exists():
        shutil.rmtree(task_dir)
    imgT.mkdir(parents=True)
    lblT.mkdir()

    # link labels once
    for lbl in LABELS_DIR.glob("*.nii*"):
        link_or_copy(lbl, lblT / lbl.name)

    provenance = [("caseID", "channel", "source")]

    for case in CASES:
        for ch in range(6):
            src_type = rule(case, ch)          # "real" or "gen"
            if src_type == "real":
                src = REAL_DIR / f"{case}_{ch:04d}.nii.gz"
            else:  # generated: filenames carry trailing 'g'
                src = GEN_DIR / f"{case}_{ch:04d}g.nii.gz"

            if not src.exists():
                raise FileNotFoundError(f"Missing file: {src}")

            dst = imgT / f"{case}_{ch:04d}.nii.gz"
            link_or_copy(src, dst)
            provenance.append((case, ch, src_type))

    # minimal dataset.json
    ds = {
        "channel_names": {str(i): n for i,n in CHANS.items()},
        "labels": {"0": "background", "1": "lesion"},
        "file_ending": ".nii.gz"
    }
    (task_dir / "dataset.json").write_text(json.dumps(ds, indent=2))

    # provenance log
    with open(task_dir / "provenance.tsv", "w", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(provenance)

    print(f"✓  built {task_dir.name}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    for tid, tag, rule in exp:
        make_task(tid, tag, rule)

    print("\nAll Task folders are ready.  Example run:")
    print("   nnUNetv2_plan_and_preprocess -d 541")
    print("   nnUNetv2_train 541 3d_fullres")
