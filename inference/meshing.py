import zmesh
import trimesh


def convert_array_to_meshes(segmentation, spacing, label_to_name):

    # Note zmesh assumes the index ordering of the segmentation volume is the same as the axis
    # ordering of spacing; vertices then have that ordering too

    mesher = zmesh.Mesher(tuple(spacing))
    mesher.mesh(segmentation, close=True)
    meshed_labels = mesher.ids()
    
    print(f"Generating meshes for labels: {meshed_labels}")

    meshes = {}    
    for label in meshed_labels:
        print(f"Processing label {label}...")
        mesh = mesher.get(
            label,
            normals=False,
            reduction_factor=100,  # TODO: use isotropic remeshing instead; set this to zero!
        )
        for cc_idx, cc_mesh in enumerate(trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces).split()):
            meshes[f'{label_to_name[label]}_{cc_idx}'] = cc_mesh

    return meshes
