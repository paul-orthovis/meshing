
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


original_folder = '/home/paul/projects/orthovis/ankle-data/from-annotators-and-daniela'
converted_folder = f'/home/paul/projects/orthovis/ankle-data/split-and-curated'
scans = {
    'CT1': {'ct_subpath': 'daniela/4 Unnamed Series.nrrd', 'seg_subpath': 'daniela/Segmentation-cortical.nrrd'},
    'CT2': {'ct_subpath': 'daniela/3 Unnamed Series.nrrd', 'seg_subpath': 'daniela/Segmentation-full.nrrd'},
    'CT4': {'ct_subpath': 'daniela-with-fixed-volume/402 Unnamed Series.nrrd', 'seg_subpath': 'daniela-with-fixed-volume/Segmentation.nrrd'},
}


def load_nrrd_image(filepath):
    img = sitk.ReadImage(filepath)
    array = sitk.GetArrayFromImage(img)  # shape: (z, y, x)
    return img, array


def show_slice(volume, slice_idx=None, title=""):
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2  # middle slice
    plt.figure(figsize=(6, 6))
    plt.imshow(volume[slice_idx], cmap='gray')
    plt.title(f"{title} (slice {slice_idx})")
    plt.axis('off')
    plt.show()


def get_centered_z_crop_pow2(mask, z_start_override=None, z_end_override=None):
    if z_start_override is not None and z_end_override is not None:
        crop_size = z_end_override - z_start_override
        return z_start_override, z_end_override, crop_size

    # Auto-detect bounding box if no manual override
    z_nonzero = np.any(mask > 0, axis=(1, 2))
    nonzero_indices = np.where(z_nonzero)[0]

    if len(nonzero_indices) == 0:
        raise ValueError("No annotation found.")

    z_min = nonzero_indices[0]
    z_max = nonzero_indices[-1]
    bbox_size = z_max - z_min + 1

    crop_size = 2 ** int(np.ceil(np.log2(bbox_size)))

    center_z = (z_min + z_max) // 2
    half_crop = crop_size // 2

    z_start = max(center_z - half_crop, 0)
    z_end = min(z_start + crop_size, mask.shape[0])
    z_start = max(z_end - crop_size, 0)

    return z_start, z_end, crop_size


def split_left_right(ct_array, seg_array):
    _, _, x_dim = ct_array.shape
    x_mid = x_dim // 2

    # Left leg (first half), Right leg (second half)
    ct_left = ct_array[:, :, :x_mid]
    ct_right = ct_array[:, :, x_mid:]
    seg_left = seg_array[:, :, :x_mid]
    seg_right = seg_array[:, :, x_mid:]

    return (ct_left, seg_left), (ct_right, seg_right), x_mid


def save_half(ct_arr, seg_arr, save_dir, original_img, x_start):
    os.makedirs(save_dir, exist_ok=True)  # make sure the output directory exists

    # Convert arrays to images
    ct_img = sitk.GetImageFromArray(ct_arr)
    seg_img = sitk.GetImageFromArray(seg_arr)

    # Copy spacing & direction
    ct_img.SetSpacing(original_img.GetSpacing())
    ct_img.SetDirection(original_img.GetDirection())

    # Adjust origin along x-axis
    origin = list(original_img.GetOrigin())
    spacing = original_img.GetSpacing()
    origin[0] += x_start * spacing[0]  # adjust for x split
    ct_img.SetOrigin(tuple(origin))
    seg_img.CopyInformation(ct_img)

    # Output paths
    ct_path = os.path.join(save_dir, f"ct.nrrd")
    seg_path = os.path.join(save_dir, f"seg.nrrd")

    # Save to disk
    sitk.WriteImage(ct_img, ct_path)
    sitk.WriteImage(seg_img, seg_path)
    print(f"Saved: {ct_path}\n       {seg_path}")


def process_scan(ct_path, seg_path, scan_name):

    ct_img, ct_array = load_nrrd_image(ct_path)
    seg_img, seg_array = load_nrrd_image(seg_path)

    print("CT shape:", ct_array.shape)
    print("Segmentation shape:", seg_array.shape)

    show_slice(ct_array, 350, title="Original CT")
    show_slice(seg_array, 350, title="Segmentation Mask")

    # z_start, z_end, used_crop_size = get_centered_z_crop_pow2(seg_array, 0, 256)
    z_start, z_end, used_crop_size = get_centered_z_crop_pow2(seg_array)

    ct_cropped_array = ct_array[z_start : z_end, :, :]
    seg_cropped_array = seg_array[z_start : z_end, :, :]

    print(f"Auto power-of-2 z-crop from slice {z_start} to {z_end} ({used_crop_size} slices)")


    show_slice(ct_cropped_array, title="Cropped CT")
    show_slice(seg_cropped_array, title="Segmentation Mask")


    ct_cropped_img = sitk.GetImageFromArray(ct_cropped_array)
    ct_cropped_img.SetSpacing(ct_img.GetSpacing())
    ct_cropped_img.SetDirection(ct_img.GetDirection())

    original_origin = ct_img.GetOrigin()
    spacing = ct_img.GetSpacing()
    new_origin_z = original_origin[2] + z_start * spacing[2]
    new_origin = (original_origin[0], original_origin[1], new_origin_z)
    ct_cropped_img.SetOrigin(new_origin)

    (left_ct, left_seg), (right_ct, right_seg), x_mid = split_left_right(ct_cropped_array, seg_cropped_array)

    print(f"Split at x={x_mid}, Left shape: {left_ct.shape}, Right shape: {right_ct.shape}")

    show_slice(left_ct, title="Left CT")
    show_slice(left_seg, title="Segmentation Mask")

    save_half(left_ct, left_seg, f'{converted_folder}/{scan_name}_first-leg', ct_cropped_img, x_start=0)
    save_half(right_ct, right_seg, f'{converted_folder}/{scan_name}_second-leg', ct_cropped_img, x_start=x_mid)


def main():
    for scan_name, scan_paths in scans.items():
        ct_path = os.path.join(original_folder, scan_name, scan_paths['ct_subpath'])
        seg_path = os.path.join(original_folder, scan_name, scan_paths['seg_subpath'])
        print(f'processing {scan_name}')
        process_scan(ct_path, seg_path, scan_name)


if __name__ == '__main__':
    main()

