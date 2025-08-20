import cc3d
import numpy as np
import scipy.ndimage
import SimpleITK as sitk

prefix = "/home/paul/remote/salina/projects/orthovis/nnUNet_results/Dataset003_Ankle_BoneAndCuts/results_foldall_2025-08-20/ankle_0011"
pred_img = sitk.ReadImage(f"{prefix}.nii.gz")
pred_arr = sitk.GetArrayFromImage(pred_img)

cut_probs_npz = np.load(f"{prefix}.npz")

cut_probs = cut_probs_npz['probabilities'][2]

bone_or_cut = pred_arr != 0
cut = cut_probs > 0.05

bone_without_cuts = np.where(cut, 0, bone_or_cut)

ccs, num_ccs = cc3d.connected_components(bone_without_cuts, connectivity=6, return_N=True)
print(f'found {num_ccs} connected components before dusting')

# TODO: threshold should depend on physical size of voxels; also choose a 'clinically reasonable' value
ccs, num_ccs = cc3d.dust(ccs, threshold=20, precomputed_ccl=True, return_N=True)
print(f'retained {num_ccs} connected components after dusting')

# TODO: use physical voxel size in edt
_, indices = scipy.ndimage.distance_transform_edt(np.where(ccs > 0, 0, 1), return_indices=True)
ccs = np.where(cut, ccs[*indices], ccs)

ccs_img = sitk.GetImageFromArray(ccs)
ccs_img.CopyInformation(pred_img)

sitk.WriteImage(ccs_img, f"{prefix}_ccs.nii.gz")
