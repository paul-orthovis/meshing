import os
import numpy as np
import SimpleITK as sitk


original_folder = '/home/paul/projects/orthovis/ankle-data/from-annotators-and-daniela'
converted_folder = '/home/paul/projects/orthovis/ankle-data/split-and-curated_cortical-only_2026-01'

# For each leg, specify whether to crop "top", "bottom", or "both".
# "top" refers to low z indices; "bottom" refers to high z indices.
scans = {
    'PILON14': {
        'ct_subpath': 'from-ahmed_2025-12-16/3 Unnamed Series.nrrd',
        'seg_subpath': 'cleaned_2026-01-26/Segmentation.nrrd',
        'crop_note': 'top',
    },
    'PILON16': {
        'ct_subpath': 'from-ahmed_2025-12-16/3 Unnamed Series.nrrd',
        'seg_subpath': 'cleaned_2026-01-26/Segmentation.nrrd',
        'crop_note': 'top',
    },
}


def load_nrrd_image(filepath):
    img = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(img)  # shape: (z, y, x)
    return img, array


def get_label_z_bounds(seg_array):
    z_nonzero = np.any(seg_array > 0, axis=(1, 2))
    nonzero_indices = np.where(z_nonzero)[0]
    if len(nonzero_indices) == 0:
        raise ValueError("No annotation found in segmentation.")
    return nonzero_indices[0], nonzero_indices[-1]


def get_crop_indices(seg_array, crop_note):
    z_min, z_max = get_label_z_bounds(seg_array)

    if crop_note not in {"top", "bottom", "both"}:
        raise ValueError(f"Invalid crop_note '{crop_note}'. Use 'top', 'bottom', or 'both'.")

    crop_top = crop_note in {"top", "both"}
    crop_bottom = crop_note in {"bottom", "both"}

    z_start = z_min if crop_top else 0
    z_end = (z_max + 1) if crop_bottom else seg_array.shape[0]

    if z_start >= z_end:
        raise ValueError(f"Invalid crop range: start={z_start}, end={z_end}.")

    return z_start, z_end


def save_cropped(ct_arr, seg_arr, save_dir, original_ct_img, original_seg_img, z_start):
    os.makedirs(save_dir, exist_ok=True)

    ct_img = sitk.GetImageFromArray(ct_arr)
    seg_img = sitk.GetImageFromArray(seg_arr)

    ct_img.SetSpacing(original_ct_img.GetSpacing())
    ct_img.SetDirection(original_ct_img.GetDirection())

    origin = list(original_ct_img.GetOrigin())
    spacing = original_ct_img.GetSpacing()
    origin[2] += z_start * spacing[2]  # adjust z origin after crop
    ct_img.SetOrigin(tuple(origin))

    seg_img.CopyInformation(ct_img)
    for key in original_seg_img.GetMetaDataKeys():
        seg_img.SetMetaData(key, original_seg_img.GetMetaData(key))

    ct_path = os.path.join(save_dir, "ct.nrrd")
    seg_path = os.path.join(save_dir, "seg.nrrd")

    sitk.WriteImage(ct_img, ct_path)
    sitk.WriteImage(seg_img, seg_path)
    print(f"Saved: {ct_path}\n       {seg_path}")


def process_scan(ct_path, seg_path, scan_name, crop_note):
    ct_img, ct_array = load_nrrd_image(ct_path)
    seg_img, seg_array = load_nrrd_image(seg_path)

    print("CT shape:", ct_array.shape)
    print("Segmentation shape:", seg_array.shape)

    z_start, z_end = get_crop_indices(seg_array, crop_note)

    ct_cropped_array = ct_array[z_start:z_end, :, :]
    seg_cropped_array = seg_array[z_start:z_end, :, :]

    print(f"Cropping '{crop_note}': z {z_start} to {z_end} (size {z_end - z_start})")

    save_cropped(
        ct_cropped_array,
        seg_cropped_array,
        os.path.join(converted_folder, scan_name),
        ct_img,
        seg_img,
        z_start,
    )


def main():
    for scan_name, scan_paths in scans.items():
        ct_path = os.path.join(original_folder, scan_name, scan_paths['ct_subpath'])
        seg_path = os.path.join(original_folder, scan_name, scan_paths['seg_subpath'])
        crop_note = scan_paths['crop_note']
        print(f'processing {scan_name} ({crop_note})')
        process_scan(ct_path, seg_path, scan_name, crop_note)


if __name__ == '__main__':
    main()
