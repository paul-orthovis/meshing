import os
import re
import json
import glob
import warnings
import cc3d
import fastremap
import numpy as np
import SimpleITK as sitk
from collections import defaultdict
from scipy.ndimage import binary_dilation


nrrds_dir = "/home/paul/projects/orthovis/ankle-data/split-and-curated_cortical-only_2026-02"
datasets_dir = f"/home/paul/projects/orthovis/ankle-data/nnUNet_raw_2026-02-11"

bone_name_to_label = {
    "tibia": 1,
    "fibula": 2,
    "talus": 3,
}


def get_legs():
    leg_dirs = sorted(glob.glob(f'{nrrds_dir}/*'))

    # Extract all leg files
    annotated_legs = []
    unannotated_legs = []
    for leg_dir in leg_dirs:
        dir_name = os.path.basename(leg_dir)
        filenames = os.listdir(leg_dir)
        maybe_segmentation = [f for f in filenames if 'seg' in f.lower()]
        maybe_ct = [f for f in filenames if f not in maybe_segmentation]
        assert len(maybe_ct) == 1 and len(maybe_segmentation) in [0, 1]
        ct_path = os.path.join(leg_dir, maybe_ct[0])
        if len(maybe_segmentation) == 1:
            annotated_legs.append({
                "dir": dir_name,
                "ct_path": ct_path,
                "seg_path": os.path.join(leg_dir, maybe_segmentation[0])
            })
        else:
            unannotated_legs.append({
                "dir": dir_name,
                "ct_path": ct_path,
            })

    print(f'found {len(annotated_legs)} annotated legs and {len(unannotated_legs)} unannotated legs')
    return annotated_legs, unannotated_legs


def preprocess_segments(seg_arr, spacing_mm, min_extent_mm=3.0, max_small_cc_relative=0.01):
    """Clean up each segment by keeping only the largest connected component.
    
    For each segment:
    1. Keep only the largest CC
    2. Warn if other CCs exceed relative size threshold AND min_extent_mm in all dimensions
    3. Drop segment entirely if largest CC extent < min_extent_mm in any dimension
    """
    cleaned = np.zeros_like(seg_arr)
    
    for label in fastremap.unique(seg_arr):
        if label == 0:
            continue
        
        mask = seg_arr == label
        cc_labels, num_cc = cc3d.connected_components(mask, return_N=True)
        
        if num_cc == 0:
            continue
        
        stats = cc3d.statistics(cc_labels)
        voxel_counts = stats['voxel_counts'][1:]  # skip background (index 0)
        bboxes = stats['bounding_boxes'][1:]
        
        largest_idx = np.argmax(voxel_counts)
        largest_count = voxel_counts[largest_idx]
        largest_bbox = bboxes[largest_idx]
        
        # Check extent of largest CC
        extents_vox = [largest_bbox[d].stop - largest_bbox[d].start for d in range(3)]
        extents_mm = [extents_vox[d] * spacing_mm[d] for d in range(3)]
        if min(extents_mm) < min_extent_mm:
            warnings.warn(f"Dropping segment {label}: largest CC extent {min(extents_mm):.1f}mm < {min_extent_mm}mm")
            continue
        
        # Check that other CCs are small
        for i, (count, bbox) in enumerate(zip(voxel_counts, bboxes)):
            if i == largest_idx:
                continue
            rel_size = count / largest_count
            cc_extents_mm = [(bbox[d].stop - bbox[d].start) * spacing_mm[d] for d in range(3)]
            if rel_size > max_small_cc_relative and min(cc_extents_mm) >= min_extent_mm:
                warnings.warn(
                    f"Segment {label}: discarding CC with {rel_size*100:.1f}% of largest "
                    f"(extent {min(cc_extents_mm):.1f}mm >= {min_extent_mm}mm)"
                )
        
        # Keep only largest CC
        cleaned[cc_labels == (largest_idx + 1)] = label
    
    return cleaned


def find_cuts_mask(seg_arr, margin=1):

    if margin > 0:
        # Dilate each label separately into background regions, so we later count near-touching labels as if touching
        seg_dilated = np.zeros_like(seg_arr)
        for label in np.unique(seg_arr):
            if label == 0:
                continue
            label_mask = seg_arr == label
            for _ in range(margin):
                label_mask = binary_dilation(label_mask) & (seg_dilated == 0)
                seg_dilated[label_mask] = label
    else:
        seg_dilated = seg_arr
    
    # Find boundary voxels between different foreground labels
    is_cut = np.zeros_like(seg_arr, dtype=bool)
    for axis in range(3):
        for direction in [-1, 1]:
            slices_current = [slice(None)] * 3
            slices_neighbor = [slice(None)] * 3
            if direction == -1:
                slices_current[axis] = slice(1, None)
                slices_neighbor[axis] = slice(None, -1)
            else:
                slices_current[axis] = slice(None, -1)
                slices_neighbor[axis] = slice(1, None)
            current = seg_dilated[tuple(slices_current)]
            neighbor = seg_dilated[tuple(slices_neighbor)]
            boundary = (current != 0) & (neighbor != 0) & (current != neighbor)
            is_cut[tuple(slices_current)] |= boundary

    is_cut_dilated = binary_dilation(is_cut, iterations=2)
    is_cut = np.where(seg_arr == 0, 0, is_cut_dilated)

    return is_cut


def relabel(seg_nrrd, mode):
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
        assert seg_arr.min() == 0 and seg_arr.max() <= 3
        label_to_bone = {label: bone for bone, label in bone_name_to_label.items()}

    # Preprocess: keep only largest CC per segment, drop tiny segments
    spacing = seg_nrrd.GetSpacing()  # (x, y, z) in mm
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    seg_arr = preprocess_segments(seg_arr, spacing_zyx)

    for label in fastremap.unique(seg_arr):
        if label > 0:
            assert label in label_to_bone, f'Label {label} not found in label_to_bone'

    if mode != 'cuts_only':

        if mode == 'binary' or mode == 'bone_and_cuts':
            instance_label_to_class_label = {0: 0, **{label: 1 for label in label_to_bone.keys()}}
        elif mode == 'multiclass':
            instance_label_to_class_label = {
                **{instance_label: bone_name_to_label[bone_name.lower()] for instance_label, bone_name in label_to_bone.items()},
                0: 0
            }
        elif mode == 'instances':
            instance_labels = [label for label in fastremap.unique(seg_arr) if label > 0]
            instance_label_to_class_label = {0: 0, **{label: i + 1 for i, label in enumerate(instance_labels)}}
        else:
            assert False

        labels_remapped = fastremap.remap(seg_arr, instance_label_to_class_label)

        if mode == 'bone_and_cuts':
            is_cut = find_cuts_mask(seg_arr)
            labels_remapped = np.where(is_cut, 2, labels_remapped)

    else:
        is_cut = find_cuts_mask(seg_arr)
        labels_remapped = np.where(is_cut, 1, 0)

    labels_remapped_img = sitk.GetImageFromArray(labels_remapped)
    labels_remapped_img.CopyInformation(seg_nrrd)
    return labels_remapped_img


def convert_legs(training_legs, testing_legs, dataset_dir, mode):
    # Convert and organize training cases
    os.makedirs(f"{dataset_dir}/imagesTr", exist_ok=True)
    os.makedirs(f"{dataset_dir}/labelsTr", exist_ok=True)
    max_instances = 0
    for i, leg in enumerate(training_legs):
        case_id = leg["dir"]
        print(f"Processing training case {case_id}...")
        ct_img = sitk.ReadImage(leg["ct_path"])
        sitk.WriteImage(ct_img, f"{dataset_dir}/imagesTr/{case_id}_0000.nii.gz")
        seg_img = sitk.ReadImage(leg["seg_path"])
        seg_img = relabel(seg_img, mode)
        sitk.WriteImage(seg_img, f"{dataset_dir}/labelsTr/{case_id}.nii.gz")

        if mode == 'instances':
            seg_arr = sitk.GetArrayFromImage(seg_img)
            num_instances = len([label for label in fastremap.unique(seg_arr) if label > 0])
            max_instances = max(max_instances, num_instances)
            print(f"Processed training case {case_id} ({num_instances} instances)")
        else:
            print(f"Processed training case {case_id}")

    # Convert and organize testing cases
    os.makedirs(f"{dataset_dir}/imagesTs", exist_ok=True)
    for i, leg in enumerate(testing_legs):
        case_id = leg["dir"]
        print(f"Processing testing case {case_id}...")
        ct_img = sitk.ReadImage(leg["ct_path"])
        sitk.WriteImage(ct_img, f"{dataset_dir}/imagesTs/{case_id}_0000.nii.gz")

        print(f"Processed testing case {case_id}")

    if mode == 'instances':
        return max_instances


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

    training_legs, testing_legs = get_legs()

    # binary_dataset_name = "Dataset001_Ankle_Binary"
    # print(f"Creating {binary_dataset_name}...")
    # convert_legs(training_legs, testing_legs, f'{datasets_dir}/{binary_dataset_name}', mode='binary')
    # create_json(datasets_dir, binary_dataset_name, len(training_legs), len(testing_legs), {'background': 0, 'bone': 1})

    # multiclass_dataset_name = "Dataset002_Ankle_Multiclass"
    # print(f"Creating {multiclass_dataset_name}...")
    # convert_legs(training_legs, testing_legs, f'{datasets_dir}/{multiclass_dataset_name}', mode='multiclass')
    # create_json(datasets_dir, multiclass_dataset_name, len(training_legs), len(testing_legs), {'background': 0, **bone_name_to_label})

    cuts_dataset_name = "Dataset003_Ankle_BoneAndCuts"
    print(f"Creating {cuts_dataset_name}...")
    convert_legs(training_legs, testing_legs, f'{datasets_dir}/{cuts_dataset_name}', mode='bone_and_cuts')
    create_json(datasets_dir, cuts_dataset_name, len(training_legs), len(testing_legs), {'background': 0, 'bone': 1, 'cut': 2})

    # cuts_dataset_name = "Dataset004_Ankle_CutsOnly"
    # print(f"Creating {cuts_dataset_name}...")
    # convert_legs(training_legs, testing_legs, f'{datasets_dir}/{cuts_dataset_name}', mode='cuts_only')
    # create_json(datasets_dir, cuts_dataset_name, len(training_legs), len(testing_legs), {'background': 0, 'cut': 1})

    instances_dataset_name = "Dataset005_Ankle_Instances"
    print(f"Creating {instances_dataset_name}...")
    max_instances = convert_legs(training_legs, testing_legs, f'{datasets_dir}/{instances_dataset_name}', mode='instances')
    fragment_labels = {'background': 0, **{f'Fragment{i:02d}': i for i in range(1, max_instances + 1)}}
    create_json(datasets_dir, instances_dataset_name, len(training_legs), len(testing_legs), fragment_labels)
    print(f"Max fragments across scans: {max_instances}")


if __name__ == "__main__":
    main()
