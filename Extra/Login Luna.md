Login Luna:
ssh jdekok@luna.amc.nl

Put files from local computer to Luna:
scp -r "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\NestedFormer-main/utils/data_utils.py" jdekok@luna:~/my-scratch/NestedFormer-main/utils
scp -r "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only/mri_seq2seq-main/nnseq2seq/training/nnSeq2SeqTrainer/nnSeq2SeqTrainer.py" jdekok@luna.amc.nl:~/my-scratch/seq2seq/mri_seq2seq-main/nnseq2seq/training/nnSeq2SeqTrainer/nnSeq2SeqTrainer.py

C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only\mri_seq2seq-main\nnseq2seq\training\nnSeq2SeqTrainer\nnSeq2SeqTrainer.py

scp -r "C:\Users\P095789\Downloads\nnunet_imagesTr2\nnUNet_combinations" jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet
scp -r "C:\Users\P095789\Downloads\nnunet_black_imagesTr\nnunet_black_combos" jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet


python eval_segmentation.py ^
  --pred_dir  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\fold_0_only\predictions" ^
  --ref_dir   "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\fold_0_only_extra_data\predictions" ^
  --out_json  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\metrics.json"


scp -r jdekok@luna.amc.nl:/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\luna\luna_last"

scp -r jdekok@luna.amc.nl:/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\luna"

scp -r jdekok@luna.amc.nl:/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results_black  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results"
s
scp -r jdekok@luna.amc.nl:/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results_black_trained  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results"


scp -r jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/imagesTs  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_raw\Dataset520_NeckTumour"
scp -r jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_raw\Dataset520_NeckTumour"
scp -r jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTr  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_raw\Dataset520_NeckTumour"


scp -r jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet/predictions  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\fold_0_only_extra_data"


scp -r jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans_3d_fullres  "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\extra_data_npy"

scp "C:\Users\P095789\Downloads\mr_subset_completed\imagesTr\*" jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/imagesTr/
scp -r "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\nnUNet_results\fold_0_only" jdekok@luna.amc.nl:~/my-scratch/nnunetv2/nnUNet/

start environment:
source NestedFormer/bin/activate

module load Anaconda3/2024.02-1
conda activate nnunet_env

run nnunet:
nnUNetv2_train 520 3d_fullres 0

squeue/view your own jobs only:
squeue -u jdekok

install tensorboard:
python3.10 -m pip install tensorboard
conda
Look a tensorboard logfiles without tensorboard (you see te training loss values):
python3.10 read_tensorboard_logs.py





module load Anaconda3/2024.02-1
conda activate nnunet_env2

export nnUNet_raw=/home/rth/jdekok/my_scratch/nnunetv2/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed
export nnUNet_results=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_results

nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities

nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/resultsGT2.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions

nnUNetv2_apply_postprocessing  -i "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions" -o "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions_postprocessed" -pp_pkl_file "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_results/Dataset520_NeckTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/postprocessing.pkl" -plans_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json" -dataset_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/dataset.json"

nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/resultspostextra.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions_postprocessed







preprocess nnunetdata:
module load Anaconda3/2024.02-1
conda activate nnunet_env
export nnUNet_raw=/home/rth/jdekok/thesis_folder/nnunetv2/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/rth/jdekok/thesis_folder/nnunetv2/nnUNet/nnUNet_preprocessed
export nnUNet_results=/home/rth/jdekok/thesis_folder/nnunetv2/nnUNet/nnUNet_results
nnUNetv2_plan_and_preprocess -d 520 --verify_dataset_integrity
train nnunet: 
run jobfile

nnUNetv2_find_best_configuration 520 -c 3d_fullres

Test nnunet:
run jobfile with predict from find_best_config?

module load Anaconda3/2024.02-1
conda activate nnunet_env2
export nnUNet_raw=/home/rth/jdekok/my_scratch/nnunetv2/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed
export nnUNet_results=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_results

nnUNetv2_apply_postprocessing  -i "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions" -o "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions_postprocessed" -pp_pkl_file "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_results/Dataset520_NeckTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/postprocessing.pkl" -plans_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json" -dataset_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/dataset.json"

python -m nnunetv2.evaluation.evaluate_predictions \
    --dataset_json     $nnUNet_raw/Dataset520_NeckTumour/dataset.json \
    --plans_json       $nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json \
    --output_file      $nnUNet_results/resultspostextra.json \
    --np \
    --reference_folder $nnUNet_raw/Dataset520_NeckTumour/labelsTs \
    --prediction_folder /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions_postprocessed

nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/resultspostextra.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions_postprocessed

nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/resultsGT2.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions


nnUNetv2_determine_postprocessing \
  -i "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_results/Dataset520_NeckTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation" \
  -ref "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTr" \
  -plans_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json" \
  -dataset_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/dataset.json"

nnUNetv2_determine_postprocessing \
  -i "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/nnUNet_results/Dataset520_NeckTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0"/validation \
  -ref "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTr" \
  -plans_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/plans.json" \
  -dataset_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/dataset.json"

nnUNetv2_apply_postprocessing  -i "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions" -o "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions_postprocessed" -pp_pkl_file "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/nnUNet_results/Dataset520_NeckTumour/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation/postprocessing.pkl" -plans_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/plans.json" -dataset_json "/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/dataset.json"

nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/plans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/resultsorgpostGT2.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions_postprocessed

nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/plans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/resultsorgGT2.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions

nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions/plans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/resultsorgpost.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/lablesTsreal /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/fold_0_only/predictions_postprocessed

lablesTsreal

If changes in files in cache:
pip install -e .


Not Luna:
Start environment:
PS C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\MEPDNet-main> .\MEPDNet\Scripts\activate


cd "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten"

install nibabel:
py -m pip install nibabel


export nnUNet_results=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_results
export nnUNet_raw=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw
export nnUNet_preprocessed=/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed

conda activate nnseq2seq
$env:nnSeq2Seq_results="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\mri_seq2seq-main\nnSeq2Seq_results"
$env:nnSeq2Seq_raw="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\mri_seq2seq-main\nnSeq2Seq_raw"
$env:nnSeq2Seq_preprocessed="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\mri_seq2seq-main\nnSeq2Seq_preprocessed"
python nnseq2seq/experiment_planning/plan_and_preprocess_entrypoints.py -d 520


nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions/nnUNetPlans.json   -o   /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/results.json \
  /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs \
  /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions

python -m nnunetv2.evaluation.evaluate_predictions \
    --dataset_json     $nnUNet_raw/Dataset520_NeckTumour/dataset.json \
    --plans_json       $nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json \
    --output_file      $nnUNet_results/results.json \
    --reference_folder $nnUNet_raw/Dataset520_NeckTumour/labelsTs \
    --prediction_folder /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions



$env:nnSeq2Seq_results="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only\mri_seq2seq-main\nnSeq2Seq_results"
$env:nnSeq2Seq_raw="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only\mri_seq2seq-main\nnSeq2Seq_raw"
$env:nnSeq2Seq_preprocessed="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only\mri_seq2seq-main\nnSeq2Seq_preprocessed"

"C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\AVL_OPSCC"


# Maybe OneDrive is the problem, so we moved the files to Downloads as well

conda activate nnseq2seq
$env:nnSeq2Seq_results="C:\Users\P095789\Downloads\mri_seq2seq-main\nnSeq2Seq_results"
$env:nnSeq2Seq_raw="C:\Users\P095789\Downloads\mri_seq2seq-main\nnSeq2Seq_raw"
$env:nnSeq2Seq_preprocessed="C:\Users\P095789\Downloads\mri_seq2seq-main\nnSeq2Seq_preprocessed"
python nnseq2seq/experiment_planning/plan_and_preprocess_entrypoints.py -d 520

python nnseq2seq/run/run_training.py \
    -dataset_name_or_id 520 \
    -configuration 3d \
    -fold 0

pip install -e .
python nnseq2seq/run/run_training.py \    -dataset_name_or_id Dataset520_NeckTumour \    -configuration 3d \    -fold 0



scp jdekok@luna.amc.nl:/home/rth/jdekok/thesis_folder/nnunetv2/nnUNet/predictions "C:\Users\P095789\Downloads\"



for file in 016*; do mv "$file" "${file/016/075}"; done
chmod u+w /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/imagesTr
mv 075* /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/imagesTr




scp jdekok@luna.amc.nl:/home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predict_single.job "C:\Users\P095789\Downloads\"


nnUNetv2_evaluate_folder -djfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/dataset.json -pfile /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json -o /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/results.json /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions

python -m nnunetv2.evaluation.evaluate_predictions \
    --dataset_json    /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/dataset.json \
    --plans_json      /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_preprocessed/Dataset520_NeckTumour/nnUNetPlans.json \
    --reference_folder /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/nnUNet_raw/Dataset520_NeckTumour/labelsTs \
    --prediction_folder /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/predictions \
    --output_file     /home/rth/jdekok/my-scratch/nnunetv2/nnUNet/evaluation_results/results.json



10.145.0.19 -> IP address
CZC5017H2F.workspace1.local


if it cannot find venv:
conda config --prepend envs_dirs /home/rth/jdekok/my-scratch/.conda/envs


$env:nnSeq2Seq_results="C:\Users\P095789\Downloads\mri_seq2seqOG\nnSeq2Seq_results"
$env:nnSeq2Seq_raw="C:\Users\P095789\Downloads\mri_seq2seqOG\nnSeq2Seq_raw"
$env:nnSeq2Seq_preprocessed="C:\Users\P095789\Downloads\mri_seq2seqOG\nnSeq2Seq_preprocessed"



Seq2Seq:
module load Anaconda3/2024.02-1
conda create -n nnseq2seq python=3.10
conda activate nnseq2seq
pip install torch torchvision torchaudio
pip install -e .


export nnSeq2Seq_results=/home/rth/jdekok/my-scratch/seq2seq/mri_seq2seq-main/nnSeq2Seq_results
export nnSeq2Seq_raw=/home/rth/jdekok/my-scratch/seq2seq/mri_seq2seq-main/nnSeq2Seq_raw
export nnSeq2Seq_preprocessed=/home/rth/jdekok/my-scratch/seq2seq/mri_seq2seq-main/nnSeq2Seq_preprocessed


export nnSeq2Seq_results=/home/rth/jdekok/my-scratch/seq2seqOG/nnSeq2Seq_results
export nnSeq2Seq_raw=/home/rth/jdekok/my-scratch/seq2seqOG/nnSeq2Seq_raw
export nnSeq2Seq_preprocessed=/home/rth/jdekok/my-scratch/seq2seqOG/nnSeq2Seq_preprocessed


$env:nnSeq2Seq_results="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_results"
$env:nnSeq2Seq_raw="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_raw"
$env:nnSeq2Seq_preprocessed="C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_preprocessed"



External computer:
conda create -n nnseq2seq python=3.10
conda activate nnseq2seq
pip install torch torchvision torchaudio
pip install -e .
(I then deleted this one: "pip install torch torchvision torchaudio"and installed it again with CUDA working now)





Seq2seq OG \Downloads:
cd "C:\Users\P095789\Downloads\seq2seqOG"
conda activate nnseq2seq
set nnSeq2Seq_results=C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_results
set nnSeq2Seq_raw=C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_raw
set nnSeq2Seq_preprocessed=C:\Users\P095789\Downloads\seq2seqOG\nnSeq2Seq_preprocessed
python nnseq2seq/run/run_training.py \    -dataset_name_or_id Dataset520_NeckTumour \    -configuration 2d \    -fold 0 \    --c


Seq2seq OG:
cd "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG"
conda activate nnseq2seq
set nnSeq2Seq_results=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_results
set nnSeq2Seq_raw=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_raw
set nnSeq2Seq_preprocessed=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_preprocessed
python nnseq2seq/run/run_training.py \    -dataset_name_or_id Dataset520_NeckTumour \    -configuration 2d \    -fold 0 \    --c

OG:
set nnSeq2Seq_results=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_results
set nnSeq2Seq_raw=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_raw
set nnSeq2Seq_preprocessed=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seqOG\nnSeq2Seq_preprocessed

python nnseq2seq/experiment_planning/plan_and_preprocess_entrypoints.py -d 520

python nnseq2seq/run/run_training.py \    -dataset_name_or_id Dataset520_NeckTumour \    -configuration 2d \    -fold 0



GEN_ONLY:
set nnSeq2Seq_results=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only\mri_seq2seq-main\nnSeq2Seq_results
set nnSeq2Seq_raw=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only\mri_seq2seq-main\nnSeq2Seq_raw
set nnSeq2Seq_preprocessed=C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\seq2seq_gen_only\mri_seq2seq-main\nnSeq2Seq_preprocessed

python nnseq2seq/experiment_planning/plan_and_preprocess_entrypoints.py -d 520

python nnseq2seq/run/run_training.py \    -dataset_name_or_id Dataset520_NeckTumour \    -configuration 2d \    -fold 0

python nnseq2seq/run/run_training.py \    -dataset_name_or_id Dataset520_NeckTumour \    -configuration 2d \    -fold 0 \    --c

python nnseq2seq/inference/predict_from_raw_data.py -i nnSeq2Seq_raw/Dataset520_NeckTumour/imagesTs -o nnSeq2Seq_predictions -d Dataset520_NeckTumour -c 2d -f 0 -chk checkpoint_best.pth --infer_translate


python nnseq2seq/inference/predict_from_raw_data.py -i C:/Users/P095789/Downloads/mr_subset/imagesTr -o nnSeq2Seq_pred_extra_data -d Dataset520_NeckTumour -c 2d -f 0 -chk checkpoint_best.pth --infer_translate

python nnseq2seq/inference/predict_from_raw_data.py -i C:/Users/P095789/Downloads/mr_subset_resampled/imagesTr -o nnSeq2Seq_pred_extra_data -d Dataset520_NeckTumour -c 2d -f 0 -chk checkpoint_best.pth --infer_translate




python main.py --logdir=log_train_nestedformer --fold=0 --json_list=./brats2020_datajson.json --max_epochs=10 --lrschedule=warmup_cosine --val_every=1 --data_dir=C:/Users/P095789/Downloads/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/ --out_channels=3 --batch_size=1 --infer_overlap=0.5


External NestedFormer:
cd "C:\Users\P095789\OneDrive - Amsterdam UMC\Documenten\NestedFormer-main"
conda activate nestedformer

cd "C:\Users\P095789\Downloads\NestedFormer-main"
conda activate nestedformer
python main.py --logdir=log_train_nestedformer --fold=0 --json_list=./headneck_datajson.json --max_epochs=100 --lrschedule=warmup_cosine --val_every=10 --data_dir="C:/Users/P095789/Downloads/NestedFormer-main/NestedFormer_preprocessed/" --out_channels=1  --in_channels=6 --batch_size=2 --infer_overlap=0.5 --warmup_epochs=5 --optim_lr=2e-4

python main.py --logdir=log_train_nestedformer_spacing --fold=0 --json_list=./headneck_datajson.json --max_epochs=1000 --lrschedule=None --val_every=20 --data_dir="C:/Users/P095789/Downloads/NestedFormer-main/NestedFormer_preprocessed/" --out_channels=1  --in_channels=6 --batch_size=1 --infer_overlap=0.5 --warmup_epochs=0 --optim_lr=2e-4




python main.py --logdir=log_train_nestedformer_spacing --fold=0 --json_list=./brats2020_datajson.json --max_epochs=1000 --lrschedule=None --val_every=2 --data_dir="C:\Users\P095789\Downloads\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData" --out_channels=3  --in_channels=4 --batch_size=1 --infer_overlap=0.5 --warmup_epochs=0 --optim_lr=2e-4



python main.py --logdir=log_train_nestedformer --fold=0 --json_list=./headneck_datajson.json --max_epochs=100 --lrschedule=warmup_cosine --val_every=50 --data_dir="C:/Users/P095789/OneDrive - Amsterdam UMC/Documenten/NestedFormer-main/NestedFormer_preprocessed/" --out_channels=1  --in_channels=6 --batch_size=1 --infer_overlap=0.5


if test_only:
python main.py --logdir=log_train_nestedformer_spacing_next --fold=0 --json_list=./headneck_datajson.json --max_epochs=1000 --lrschedule=warmup_cosine --val_every=1 --data_dir="C:/Users/P095789/Downloads/NestedFormer-main/NestedFormer_preprocessed/" --out_channels=1  --in_channels=6 --batch_size=1 --infer_overlap=0.5 --test_only




module load Anaconda3/2024.02-1
conda activate nnFormer
cd /home/rth/jdekok/my-scratch/nnFormer/nnformer

python get_json.py --data-dir /home/rth/jdekok/my-scratch/nnFormer/nnformer/nnformer_raw --output-dir /home/rth/jdekok/my-scratch/nnFormer/nnformer/nnformer_raw --file-ending .nii.gz --num-val 15 --seed 44

export nnFormer_raw_data_base=/home/rth/jdekok/my-scratch/nnFormer/nnformer/nnformer_raw/Task04_Dataset520_NeckTumour
export nnFormer_preprocessed=/home/rth/jdekok/my-scratch/nnFormer/nnformer/nnformer_preprocessed
export RESULTS_FOLDER=/home/rth/jdekok/my-scratch/nnFormer/nnformer/nnformer_results

nnFormer_convert_decathlon_task \
  -i /home/rth/jdekok/my-scratch/nnFormer/nnformer/nnformer_raw/Task04_Dataset520_NeckTumour 

nnFormer_plan_and_preprocess -t 4

bash train_inference.sh -c 0 -n necktumour -t 4



nano nnformer/network_architecture/nnFormer_acdc.py
find . -name '*.pyc'      -delete
find . -type d -name '__pycache__' -exec rm -rf {} +
export PYTHONPATH=/home/rth/jdekok/my-scratch/nnFormer:$PYTHONPATH
bash train_inference.sh -c 0 -n necktumour -t 4