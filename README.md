
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

- do nnUNet preprocessing and planning with `prepare_nnunet.sh`

- create splits json -- currently manual; see `splits_final.json` in this folder, copy to both nnunet
  processed datasets folders
- currently this holds out one fractured scan per fold
- change to nnunet default (five random folds) when we more scans
  - in that case just skip this step

- train nnunet: `nnUNetv2_train 001 3d_fullres 0` where 001 or 002 is the dataset and 0 is the fold
