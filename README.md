
## Data conversion

- start with NRRDs for volume and annotation
- could be whole-body (incl. two legs); could be single-leg
- first create split-and-curated versions
- for whole-body, do `crop_whole_body.py`; scans are listed at top of file
  - this chops into two halves, vertically crops somewhat conservatively
  - doesn't change format or labels in any way
- for single-leg, copy CT and segmentation NRRDs manually to relevant subfolder
  - ensure segmentation has 'seg' somewhere in filename, and CT does not

- for nnUNet, we create both binary and multiclass versions
- use `convert_nrrd_to_nnunet.py` on the above split-and-curated scans

- do nnUNet preprocessing and planning
  - `. prepare_nnunet.sh` for environment
  - `nnUNetv2_plan_and_preprocess -d 001 002 --clean --verify_dataset_integrity --verbose`

- create splits json -- currently manual; see `splits_final.json` in this folder, copy to both nnunet
  processed datasets folders
- currently this holds out one fractured scan per fold
- change to nnunet default (five random folds) when we more scans
  - in that case just skip this step


## Training and inference

- train nnunet:
  - `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 001 3d_fullres 0` where 001 or 002 is the dataset and 0 is the fold
  - n.b. nnUNet_raw not required if _processed exists

- predict with nnunet:
  - `nnUNetv2_predict -f 0 -i $nnUNet_raw/Dataset001_Ankle_Binary/imagesTr -o $nnUNet_results/Dataset001_Ankle_Binary/results_fold0_2025-08-05/ -d 001 -c 3d_fullres -chk checkpoint_best.pth`
  
- export the model
  - `nnUNetv2_export_model_to_zip -d 001 -o exported.zip -c 3d_fullres -f 0` where 001 is dataset and 0 is fold
- re-import the model
  - first set the nnUNet paths to somewhere that doesn't overlap the original
  - `nnUNetv2_install_pretrained_model_from_zip exported.zip`


## SageMaker deployment

- to deploy given a trained nnUNet folder:
  - `cd inference`
  - `python deploy.py --stack dev --nnunet-path ~/remote/salina/projects/orthovis/nnUNet_results/Dataset001_Ankle_Binary/nnUNetTrainer__nnUNetPlans__3d_fullres --profile AdministratorAccess-643058308155`
- to undeploy:
  - `python deploy.py --undeploy --stack dev --profile AdministratorAccess-643058308155`
- test locally with `python test_inference_local.py`
