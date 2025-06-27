#!/usr/bin/env python3
"""
nnunet_task_builder_black.py
----------------------------
Create 14 nnU-Net Task folders where selected modalities are replaced by a
'black' (all-zero) scan.  Geometry, header and naming follow nnU-Net specs.
"""

import csv, json, os, random, shutil, sys
from pathlib import Path

# ───────────── USER SETTINGS (updated) ──────────────────────────────
REAL_DIR   = Path(r"C:\Users\P095789\Downloads\seq2seqOGreal\nnSeq2Seq_raw\Dataset520_NeckTumour\imagesTs")
GEN_DIR    = Path(r"C:\Users\P095789\Downloads\nnunet_black_imagesTr")          # ← black images here
LABELS_DIR = Path(r"C:\Users\P095789\Downloads\seq2seqOGreal\nnSeq2Seq_raw\Dataset520_NeckTumour\labelsTs")
DST_ROOT   = Path(r"C:\Users\P095789\Downloads\nnunet_black_imagesTr\nnunet_black_combos")            # ← new output root

LINK       = shutil.copy2   # or os.link / os.symlink
SEED       = 42
# ────────────────────────────────────────────────────────────────────

CHANS = {0:"T1",1:"T1C",2:"T2",3:"STIR",4:"DWI",5:"ADC"}

for p in (REAL_DIR, GEN_DIR, LABELS_DIR):
    if not p.exists():
        sys.exit(f"❌  Source folder not found: {p}")

CASES = sorted(p.name[:3] for p in REAL_DIR.glob("*_0000.nii*"))
if not CASES:
    sys.exit("❌  No cases found in REAL_DIR")

random.seed(SEED)
def pick(frac): return set(random.sample(CASES, round(frac*len(CASES))))
T2_30, STIR_30 = pick(0.30), pick(0.30)
T1_30, T1C_30  = pick(0.30), pick(0.30)
DWI_60         = pick(0.60)

# ---------------- 14 black-scan experiments (IDs 641-656) ----------------
exp = [
(641, "allBlack",
      lambda c,k: "gen"),
(642, "T1Black",
      lambda c,k: "gen" if k==0 else "real"),
(643, "T1CBlack",
      lambda c,k: "gen" if k==1 else "real"),
(644, "T2Black",
      lambda c,k: "gen" if k==2 else "real"),
(645, "STIRBlack",
      lambda c,k: "gen" if k==3 else "real"),
(646, "DWIBlack",
      lambda c,k: "gen" if k==4 else "real"),
(647, "ADCBLack",
      lambda c,k: "gen" if k==5 else "real"),
(648, "DWI_ADC_Black",
      lambda c,k: "gen" if k in (4,5) else "real"),
(651, "30T2_30STIR_Black",
      lambda c,k: "gen" if (k==2 and c in T2_30) or (k==3 and c in STIR_30) else "real"),
(652, "30T2_30STIR_ALLDWIADC_Black",
      lambda c,k: "gen" if (k in (4,5) or
                             (k==2 and c in T2_30) or
                             (k==3 and c in STIR_30)) else "real"),
(653, "30T2_30STIR_60DWIADC_Black",
      lambda c,k: "gen" if ((k in (4,5) and c in DWI_60) or
                             (k==2 and c in T2_30) or
                             (k==3 and c in STIR_30)) else "real"),
(654, "30T1_30T1C_Black",
      lambda c,k: "gen" if (k==0 and c in T1_30) or (k==1 and c in T1C_30) else "real"),
(655, "30T1_30T1C_60DWIADC_Black",
      lambda c,k: "gen" if ((k in (4,5) and c in DWI_60) or
                             (k==0 and c in T1_30)  or
                             (k==1 and c in T1C_30)) else "real"),
(656, "30T1_30T1C_60DWIADC_30T2_30STIR_Black",
      lambda c,k: "gen" if ((k in (4,5) and c in DWI_60) or
                             (k==0 and c in T1_30)  or
                             (k==1 and c in T1C_30) or
                             (k==2 and c in T2_30)  or
                             (k==3 and c in STIR_30)) else "real")
]

# -------------- helper functions stay unchanged -------------------
def link_or_copy(src: Path, dst: Path):
    if LINK is shutil.copy2:
        shutil.copy2(src, dst)
    else:
        if dst.exists(): dst.unlink()
        LINK(src, dst)

def make_task(task_id:int, tag:str, rule):
    task_dir = DST_ROOT / f"Task{task_id}_{tag}"
    imgT, lblT = task_dir/"imagesTr", task_dir/"labelsTr"
    if task_dir.exists(): shutil.rmtree(task_dir)
    imgT.mkdir(parents=True); lblT.mkdir()

    for lbl in LABELS_DIR.glob("*.nii*"):
        link_or_copy(lbl, lblT / lbl.name)

    prov=[("caseID","channel","source")]
    for case in CASES:
        for ch in range(6):
            src_type = rule(case, ch)
            src = (REAL_DIR if src_type=="real" else GEN_DIR) / f"{case}_{ch:04d}.nii.gz"
            if not src.exists(): raise FileNotFoundError(src)
            link_or_copy(src, imgT / f"{case}_{ch:04d}.nii.gz")
            prov.append((case,ch,src_type))

    ds={"channel_names":{str(i):n for i,n in CHANS.items()},
        "labels":{"0":"background","1":"lesion"},
        "file_ending":".nii.gz"}
    (task_dir/"dataset.json").write_text(json.dumps(ds,indent=2))
    with open(task_dir/"provenance.tsv","w",newline="") as f:
        csv.writer(f,delimiter="\t").writerows(prov)
    print(f"✓ built {task_dir.name}")

# ------------------------------------------------------------------
if __name__=="__main__":
    for tid, tag, rule in exp:
        make_task(tid, tag, rule)

    print("\nAll BLACK Task folders ready in", DST_ROOT)
