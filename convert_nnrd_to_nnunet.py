import os
import re
import json
import glob
import fastremap
import numpy as np
import SimpleITK as sitk
from collections import defaultdict


nrrds_dir = "/home/paul/projects/orthovis/ankle-data/split-and-curated"
datasets_dir = f"/home/paul/projects/orthovis/ankle-data/nnUNet_raw"

bone_name_to_label = {
    "tibia": 1,
    "fibula": 2,
    "talus": 3,
    "calcaneus": 4
}


def get_legs():
    leg_dirs = sorted(glob.glob(f'{nrrds_dir}/*'))

    # Extract all leg files
    all_legs = []
    for leg_dir in leg_dirs:
        dir_name = os.path.basename(leg_dir)
        filenames = os.listdir(leg_dir)
        maybe_segmentation = [f for f in filenames if 'seg' in f.lower()]
        maybe_ct = [f for f in filenames if f not in maybe_segmentation]
        assert len(maybe_segmentation) == len(maybe_ct) == 1
        all_legs.append({
            "dir": dir_name,
            "ct_path": os.path.join(leg_dir, maybe_ct[0]),
            "seg_path": os.path.join(leg_dir, maybe_segmentation[0])
        })

    print(f'found {len(all_legs)} legs')
    return all_legs


def relabel(seg_nrrd, binary_or_multiclass):
    seg_arr = sitk.GetArrayFromImage(seg_nrrd)
    ids = [int(key.split('_')[0][7:]) for key in seg_nrrd.GetMetaDataKeys() if re.match(r'Segment\d+_LabelValue', key)]
    assert len(ids) == len(set(ids)), 'Duplicate IDs found'

    label_to_bone = {}
    if seg_nrrd.GetNumberOfComponentsPerPixel() == 1:
        assert seg_arr.ndim == 3
        for N in ids:
            seg_id = seg_nrrd.GetMetaData(f'Segment{N}_LabelValue')
            seg_name = seg_nrrd.GetMetaData(f'Segment{N}_Name').replace(' ', '_')
            print(f'{seg_id}: {seg_name}')
            label_to_bone[int(seg_id)] = seg_name.split('_')[0]
    else:
        # Slicer sometimes uses multiple layers, presumably due to overlaps; we merge these, arbitrarily favouring later layers
        assert seg_arr.ndim == 4
        assert seg_arr.shape[-1] == seg_nrrd.GetNumberOfComponentsPerPixel()
        layer_to_label_to_new_label = defaultdict(lambda: {})
        label_to_bone = {}
        for N in ids:
            layer_id = int(seg_nrrd.GetMetaData(f'Segment{N}_Layer'))
            label_id = int(seg_nrrd.GetMetaData(f'Segment{N}_LabelValue'))
            seg_name = seg_nrrd.GetMetaData(f'Segment{N}_Name').replace(' ', '_')
            new_label = len(label_to_bone) + 1
            layer_to_label_to_new_label[layer_id][label_id] = new_label
            label_to_bone[new_label] = seg_name.split('_')[0]
        new_seg = np.zeros_like(seg_arr[..., 0])
        for layer_id, label_to_new_label in layer_to_label_to_new_label.items():
            orig_layer = seg_arr[..., layer_id]
            new_layer = fastremap.remap(orig_layer, {**label_to_new_label, 0: 0})
            new_seg = np.maximum(new_seg, new_layer)
        seg_arr = new_seg

    if len(label_to_bone) == 0:  # probably a whole-body crop; hopefully labels are already good
        assert seg_arr.min() == 0 and seg_arr.max() <= 4
        label_to_bone = {label: bone for bone, label in bone_name_to_label.items()}

    for label in fastremap.unique(seg_arr):
        if label > 0:
            assert label in label_to_bone, f'Label {label} not found in label_to_bone'

    if binary_or_multiclass == 'binary':
        instance_label_to_class_label = {0: 0, **{label: 1 for label in label_to_bone.keys()}}
    elif binary_or_multiclass == 'multiclass':
        instance_label_to_class_label = {
            **{instance_label: bone_name_to_label[bone_name.lower()] for instance_label, bone_name in label_to_bone.items()},
            0: 0
        }
    else:
        assert False

    labels_remapped = fastremap.remap(seg_arr, instance_label_to_class_label)
    labels_remapped_img = sitk.GetImageFromArray(labels_remapped)
    labels_remapped_img.CopyInformation(seg_nrrd)
    return labels_remapped_img


def convert_legs(training_legs, testing_legs, dataset_dir, binarise):
    # Convert and organize training cases
    os.makedirs(f"{dataset_dir}/imagesTr", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labelsTr", exist_ok=True)
    for i, leg in enumerate(training_legs):
        case_id = f"ankle_{i:04d}"
        ct_img = sitk.ReadImage(leg["ct_path"])
        sitk.WriteImage(ct_img, f"{dataset_dir}/imagesTr/{case_id}_0000.nii.gz")
        seg_img = sitk.ReadImage(leg["seg_path"])
        seg_img = relabel(seg_img, 'binary' if binarise else 'multiclass')
        sitk.WriteImage(seg_img, f"{dataset_dir}/labelsTr/{case_id}.nii.gz")

        print(f"Processed training case {case_id}: {leg['dir']}")

    #
    # # Convert and organize test cases
    # # TODO: check & update
    # os.makedirs(imagesTs, exist_ok=True)
    # os.makedirs(labelsTs, exist_ok=True)
    # for i, leg in enumerate(testing_legs):
    #     case_id = f"ankle_{i+6:04d}"
    #
    #     # Convert CT to NIfTI
    #     ct_img = sitk.ReadImage(leg["ct_path"])
    #     sitk.WriteImage(ct_img, f"{imagesTs}/{case_id}_0000.nii.gz")
    #
    #     # Convert segmentation to NIfTI
    #     seg_img = sitk.ReadImage(leg["seg_path"])
    #     sitk.WriteImage(seg_img, f"{labelsTs}/{case_id}.nii.gz")
    #
    #     print(f"Processed test case {case_id}: {leg['dir']} {leg['leg']}")
    #
    #
    # print(f"Processed {len(training_legs)} training cases and {len(testing_legs)} test cases")


def create_json(datasets_dir, dataset_name, num_training, num_testing, label_name_to_value):

    dataset_dir = f"{datasets_dir}/{dataset_name}"

    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": label_name_to_value,
        "numTraining": num_training,
        "numTest": num_testing,
        "file_ending": ".nii.gz",
    }

    with open(f"{dataset_dir}/dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)


def main():

    all_legs = get_legs()

    training_legs = all_legs[:]
    testing_legs = []

    binary_dataset_name = "Dataset001_Ankle_Binary"
    convert_legs(training_legs, testing_legs, f'{datasets_dir}/{binary_dataset_name}', binarise=True)
    create_json(datasets_dir, binary_dataset_name, len(training_legs), len(testing_legs), {'background': 0, 'bone': 1})

    multiclass_dataset_name = "Dataset002_Ankle_Multiclass"
    convert_legs(training_legs, testing_legs, f'{datasets_dir}/{multiclass_dataset_name}', binarise=False)
    create_json(datasets_dir, multiclass_dataset_name, len(training_legs), len(testing_legs), {'background': 0, **bone_name_to_label})


if __name__ == "__main__":
    main()
