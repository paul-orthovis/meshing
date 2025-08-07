import io
import os
import zipfile
import logging
import tempfile

import torch
import SimpleITK as sitk
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import meshing


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def model_fn(model_dir):
    predictor = nnUNetPredictor(
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=False,
    )
    files = [os.path.relpath(os.path.join(dp, f), model_dir) for dp, dn, filenames in os.walk(model_dir) for f in filenames]
    logger.debug(f'model_dir: {model_dir}, files: {files}')
    predictor.initialize_from_trained_model_folder(f'{model_dir}/model', use_folds=None, checkpoint_name='checkpoint_best.pth')
    return predictor


def transform_fn(predictor, request_body, content_type, accept):

    with tempfile.TemporaryDirectory(prefix='case_') as temp_dir:

        # Assuming request body is a zip, extract all files to temporary directory
        with zipfile.ZipFile(io.BytesIO(request_body), 'r') as zip_file:
            file_list = zip_file.namelist()
            logger.info(f"Files in zip: {file_list}")
            dicom_dir = os.path.join(temp_dir, 'dicom')
            os.makedirs(dicom_dir, exist_ok=True)
            for file_name in file_list:
                try:
                    zip_file.extract(file_name, dicom_dir)
                except Exception as e:
                    logger.warning(f"Failed to extract {file_name}: {e}")
        
        # Read the DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        size = image.GetSize()
        logger.info(f"Image size: {size[0]} x {size[1]} x {size[2]}")
        spacing = image.GetSpacing()
        logger.info(f"Image spacing: {spacing[0]} x {spacing[1]} x {spacing[2]}")
        
        # Write the image as NIfTI format to temporary directory
        nifti_path = os.path.join(temp_dir, 'volume.nii.gz')
        sitk.WriteImage(image, nifti_path)
        logger.info(f"Written NIfTI image: {nifti_path}")

        # TODO: instead of the above, to remove SimpleITK dependency and avoid disk round-trip, could 
        #  read into memory with pydicom then use predictor.predict_single_npy_array
        #  That'd require our own logic for slice ordering (and ensuring it agrees with that used for training!)

        # Run nnUNet prediction, without writing to disk
        predicted_labels = predictor.predict_from_files_sequential(
            [[nifti_path]],
            None,  # i.e. return in memory, not on disk
        )[0]

    # Apply zmesh to convert segmentation voxels to meshes
    # TODO: make sure spacing has correct axis order (and we understand how vertex coordinates are given)
    meshes = meshing.convert_array_to_meshes(predicted_labels, spacing)
    
    # TODO: isotropic remeshing (and then disable reduction_factor in zmesh)

    return [
        {'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist()}
        for mesh in meshes.values()
    ]
