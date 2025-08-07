import os
import json
import logging
import inference

logging.basicConfig(level=logging.DEBUG)

model_path = '/mnt/bcache/projects/orthovis/nnUNet_results/Dataset001_Ankle_Binary/nnUNetTrainer__nnUNetPlans__3d_fullres'
os.symlink(model_path, './model', target_is_directory=True)  # we need a subfolder 'model' to be compatible with model_fn
try:
    predictor = inference.model_fn(os.path.abspath('.'))

    with open('../../cropped-dicom-example.zip', 'rb') as f:
        request_body = f.read()

    result = inference.transform_fn(predictor, request_body, 'application/zip', 'application/json')

    print(json.dumps(result, indent=2))  # sagemaker also coerces to json!

finally:
    os.unlink('./model')
