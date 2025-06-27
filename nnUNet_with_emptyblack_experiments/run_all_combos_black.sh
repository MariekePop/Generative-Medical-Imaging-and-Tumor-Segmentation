#!/usr/bin/env bash
set -euo pipefail

# ─── model & true masks (Dataset 520) ───────────────────────────────
BASE_ID=520
CONFIG=3d_fullres
FOLD=0
RAW_REF=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset${BASE_ID}_NeckTumour
GT_REAL=$RAW_REF/labelsTsreal

PREP_REF=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset${BASE_ID}_NeckTumour
PLANS=$PREP_REF/nnUNetPlans.json
PP_PKL=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/nnUNet_results/Dataset${BASE_ID}_NeckTumour/nnUNetTrainer__nnUNetPlans__${CONFIG}/fold_${FOLD}/validation/postprocessing.pkl
CHK=checkpoint_best.pth

# ─── NEW: black-combo test sets ─────────────────────────────────────
COMB_ROOT=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnunet_black_combos
DATASETS=(641 642 643 644 645 646 647 648 651 652 653 654 655 656)

OUT_ROOT=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions_black_combos
EVAL_ROOT=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results_black

mkdir -p "$OUT_ROOT" "$EVAL_ROOT"

for ID in "${DATASETS[@]}"; do
    COMB=$(readlink -f "$COMB_ROOT"/Task${ID}_*)
    IMGS=$COMB/imagesTr
    TAG=${COMB##*/}

    RAW_OUT=$OUT_ROOT/${ID}
    POST_OUT=${RAW_OUT}_post
    mkdir -p "$RAW_OUT" "$POST_OUT"

    echo "────────  predicting $TAG  ────────"
    nnUNetv2_predict \
        -i "$IMGS" -o "$RAW_OUT" \
        -d $BASE_ID -c $CONFIG -f $FOLD \
        -chk "$CHK" \
        --disable_tta --save_probabilities

    echo "────────  evaluating  (raw) $TAG  ────────"
    nnUNetv2_evaluate_folder \
        -djfile "$RAW_REF/dataset.json" -pfile "$PLANS" \
        -o "$EVAL_ROOT/${ID}_raw_GTreal.json" \
        "$GT_REAL" "$RAW_OUT"

    echo "────────  post-processing $TAG  ────────"
    nnUNetv2_apply_postprocessing \
        -i "$RAW_OUT" -o "$POST_OUT" \
        -pp_pkl_file "$PP_PKL" \
        -plans_json  "$PLANS" \
        -dataset_json "$RAW_REF/dataset.json"

    echo "────────  evaluating  (post) $TAG  ────────"
    nnUNetv2_evaluate_folder \
        -djfile "$RAW_REF/dataset.json" -pfile "$PLANS" \
        -o "$EVAL_ROOT/${ID}_post_GTreal.json" \
        "$GT_REAL" "$POST_OUT"

    echo
done
echo "✅  all BLACK combinations finished using labelsTsreal"

