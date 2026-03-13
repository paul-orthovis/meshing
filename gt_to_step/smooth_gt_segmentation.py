"""
Load a ground-truth segmentation NRRD, filter small labels, apply morphological smoothing,
and save the result to disk.
"""

import re
import warnings
from collections import defaultdict
from pathlib import Path

import click
import cc3d
import fastremap
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_closing


def preprocess_segments(seg_arr, spacing_mm, min_extent_mm=3.0, max_small_cc_relative=0.01):
    """Keep only the largest CC per label; drop labels whose largest CC is too small."""
    cleaned = np.zeros_like(seg_arr)

    for label in fastremap.unique(seg_arr):
        if label == 0:
            continue

        mask = seg_arr == label
        cc_labels, num_cc = cc3d.connected_components(mask, return_N=True)

        if num_cc == 0:
            continue

        stats = cc3d.statistics(cc_labels)
        voxel_counts = stats['voxel_counts'][1:]
        bboxes = stats['bounding_boxes'][1:]

        largest_idx = np.argmax(voxel_counts)
        largest_count = voxel_counts[largest_idx]
        largest_bbox = bboxes[largest_idx]

        extents_vox = [largest_bbox[d].stop - largest_bbox[d].start for d in range(3)]
        extents_mm = [extents_vox[d] * spacing_mm[d] for d in range(3)]
        if min(extents_mm) < min_extent_mm:
            warnings.warn(f"Dropping label {label}: largest CC extent {min(extents_mm):.1f}mm < {min_extent_mm}mm")
            continue

        for i, (count, bbox) in enumerate(zip(voxel_counts, bboxes)):
            if i == largest_idx:
                continue
            rel_size = count / largest_count
            cc_extents_mm = [(bbox[d].stop - bbox[d].start) * spacing_mm[d] for d in range(3)]
            if rel_size > max_small_cc_relative and min(cc_extents_mm) >= min_extent_mm:
                warnings.warn(
                    f"Label {label}: discarding CC with {rel_size*100:.1f}% of largest "
                    f"(extent {min(cc_extents_mm):.1f}mm >= {min_extent_mm}mm)"
                )

        cleaned[cc_labels == (largest_idx + 1)] = label

    return cleaned


def make_ball(radius):
    """Full cubic structuring element of side 2*radius+1."""
    side = 2 * radius + 1
    return np.ones((side, side, side), dtype=bool)


def smooth_labels(seg_arr, iterations=2, radius=1):
    """Smooth label boundaries with morphological closing (dilate then erode) per label.

    Each label is closed independently, which fills small concavities and smooths
    boundaries without net shrinkage.  Where closing causes two labels to overlap,
    the original assignment is kept for those contested voxels.
    """
    unique_labels = [int(l) for l in fastremap.unique(seg_arr) if l > 0]
    if not unique_labels:
        return seg_arr.copy()

    structure = make_ball(radius)
    closed_masks = {
        label: binary_closing(seg_arr == label, structure=structure, iterations=iterations)
        for label in unique_labels
    }

    # Count how many labels claim each voxel after closing
    claim_count = np.zeros(seg_arr.shape, dtype=np.uint8)
    for mask in closed_masks.values():
        claim_count += mask.view(np.uint8)

    smoothed = np.zeros_like(seg_arr)
    for label, mask in closed_masks.items():
        # smoothed[mask & (claim_count == 1)] = label
        smoothed[mask] = label

    # Contested voxels fall back to the original assignment
    # contested = claim_count > 1
    # smoothed[contested] = seg_arr[contested]

    return smoothed


def load_seg_array(seg_img):
    """Read the voxel array from a (possibly multi-layer) Slicer segmentation image."""
    seg_arr = sitk.GetArrayFromImage(seg_img)
    ids = [
        int(key.split('_')[0][7:])
        for key in seg_img.GetMetaDataKeys()
        if re.match(r'Segment\d+_LabelValue', key)
    ]

    if seg_arr.ndim == 3:
        # Single-layer: labels are already in the array
        return seg_arr

    # Multi-layer: merge layers
    assert seg_arr.ndim == 4
    assert seg_arr.shape[-1] == seg_img.GetNumberOfComponentsPerPixel()

    layer_to_label_to_new_label = defaultdict(dict)
    label_to_bone = {}
    for N in sorted(ids):
        layer_id = int(seg_img.GetMetaData(f'Segment{N}_Layer'))
        label_id = int(seg_img.GetMetaData(f'Segment{N}_LabelValue'))
        seg_name = seg_img.GetMetaData(f'Segment{N}_Name').replace(' ', '_')
        new_label = len(label_to_bone) + 1
        layer_to_label_to_new_label[layer_id][label_id] = new_label
        label_to_bone[new_label] = seg_name

    new_seg = np.zeros_like(seg_arr[..., 0])
    for layer_id, label_to_new_label in layer_to_label_to_new_label.items():
        orig_layer = seg_arr[..., layer_id]
        new_layer = fastremap.remap(orig_layer, {**label_to_new_label, 0: 0})
        new_seg = np.maximum(new_seg, new_layer)

    return new_seg


@click.command()
@click.argument('labels_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--iterations', default=2, show_default=True,
              help='Morphological closing iterations (dilate+erode).')
@click.option('--radius', default=1, show_default=True,
              help='Structuring element radius in voxels (radius=1 → 3x3x3 cube).')
@click.option('--min-extent-mm', default=3.0, show_default=True,
              help='Minimum extent (mm) for a label to be kept.')
def main(labels_path, output_path, iterations, radius, min_extent_mm):
    """Load a GT segmentation NRRD, filter small labels, smooth, and save."""
    labels_path = Path(labels_path)

    print(f"Loading {labels_path} ...")
    seg_img = sitk.ReadImage(str(labels_path))

    seg_arr = load_seg_array(seg_img)
    print(f"Array shape: {seg_arr.shape}, unique labels: {fastremap.unique(seg_arr).tolist()}")

    spacing = seg_img.GetSpacing()  # (x, y, z) in mm
    spacing_zyx = (spacing[2], spacing[1], spacing[0])

    print("Filtering labels by size ...")
    seg_arr = preprocess_segments(seg_arr, spacing_zyx, min_extent_mm=min_extent_mm)
    print(f"After filtering, unique labels: {fastremap.unique(seg_arr).tolist()}")

    print(f"Smoothing labels (closing iterations={iterations}, radius={radius}) ...")
    smoothed = smooth_labels(seg_arr, iterations=iterations, radius=radius)
    print(f"After smoothing, unique labels: {fastremap.unique(smoothed).tolist()}")

    out_img = sitk.GetImageFromArray(smoothed)
    out_img.CopyInformation(seg_img)
    print(f"Saving to {output_path} ...")
    sitk.WriteImage(out_img, str(output_path))
    print("Done.")


if __name__ == '__main__':
    main()
