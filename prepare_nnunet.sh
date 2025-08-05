#!/bin/bash

export nnUNet_raw=/home/paul/projects/orthovis/ankle-data/nnUNet_raw
export nnUNet_preprocessed=/home/paul/projects/orthovis/ankle-data/nnUNet_preprocessed
export nnUNet_results=/home/paul/projects/orthovis/nnUNet_results

nnUNetv2_plan_and_preprocess -d 001 002 --clean --verify_dataset_integrity --verbose
