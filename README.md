
## Data conversion

- start with NRRDs for volume and annotation
- could be whole-body (incl. two legs, probably into pelvis/abdomen); could be single-leg
- first create split-and-curated versions
- for whole-body, do `crop_whole_body.py`; scans are listed at top of file
  - this chops into two halves, vertically crops somewhat conservatively
  - doesn't change format or labels in any way
- for single-leg with truncated annotations (e.g. only to just above fracture), created cropped versions
  - do `crop_partial_leg.py`; scans are listed at top of file
- for other single-leg, copy CT and segmentation NRRDs manually to relevant subfolder
  - ensure segmentation has 'seg' somewhere in filename, and CT does not
  - ok if segmentation is missing (becomes test data)

- for nnUNet, we create binary, multiclass, and cut-based versions
- use `convert_nrrd_to_nnunet.py` on the above split-and-curated scans
  - this restricts to tibia, fibula, talus, and converts to multiclass / bone-and-cut representations; it requires one/two nerds per folder and assumes one containing “seg” is segmentation (if no segmentation, becomes test data)

- do nnUNet preprocessing and planning
  - `. prepare_nnunet.sh` for environment
  - `nnUNetv2_plan_and_preprocess -d 003 005 -c 3d_fullres --clean --verify_dataset_integrity --verbose -np 2`

- create splits json -- currently manual; see `splits_final.json` in this folder, copy to both nnunet
  processed datasets folders
- currently this holds out one fractured scan per fold
- change to nnunet default (five random folds) when we have more scans
  - in that case just skip this step


## Training and inference

- train nnunet:
  - `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 003 3d_fullres 0` where 003 is the dataset and 0 is the fold
  - n.b. nnUNet_raw not required if _processed exists

- predict with nnunet:
  - note validation cases for each fold will be done automatically at the end of training
  - for 4-fold ensemble prediction on held-out data:
  `- nnUNetv2_predict -i $nnUNet_raw/Dataset003_Ankle_BoneAndCuts/imagesTs -o $nnUNet_results/Dataset003_Ankle_BoneAndCuts/results_heldout_ensemble_2025-11-05/ -d 003 -c 3d_fullres -f 0 1 2 3 -chk checkpoint_best.pth`
  - in case of an all-folds model (not an ensemble), use `-f all` 
  
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
