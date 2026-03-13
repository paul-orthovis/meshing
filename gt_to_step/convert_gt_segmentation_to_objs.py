from pathlib import Path
import re
import subprocess

import click
import numpy as np
import SimpleITK as sitk

try:
    import meshing
except ModuleNotFoundError:
    from inference import meshing


def obj_output_path(segmentation_path: Path) -> Path:
    path_no_suffix = segmentation_path
    while path_no_suffix.suffix:
        path_no_suffix = path_no_suffix.with_suffix("")
    return path_no_suffix.with_suffix(".obj")


def sanitize_component(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "segment"


_REMESH_BIN = Path(__file__).parent / "cpp" / "remesh_obj"


@click.command()
@click.argument("segmentation_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target-edge-length", default=1.2, show_default=True, help="Target edge length for remeshing.")
def main(segmentation_path: Path, target_edge_length: float):
    """Convert a voxel segmentation to OBJ meshes using zmesh via meshing.py."""
    seg_img = sitk.ReadImage(str(segmentation_path))
    spacing_xyz_mm = seg_img.GetSpacing()  # x, y, z
    seg_zyx = sitk.GetArrayFromImage(seg_img)

    if seg_zyx.ndim != 3:
        raise click.ClickException(
            f"Expected a 3D label image; got shape {seg_zyx.shape}."
        )

    seg_xyz = seg_zyx.transpose(2, 1, 0)
    labels = [int(label) for label in np.unique(seg_xyz) if label > 0]
    if not labels:
        raise click.ClickException("No non-zero labels found in segmentation.")

    label_to_name = {label: f"label_{label}" for label in labels}
    meshes = meshing.convert_array_to_meshes(seg_xyz, spacing_xyz_mm, label_to_name)
    if not meshes:
        raise click.ClickException("Meshing produced no output meshes.")

    base_output_path = obj_output_path(segmentation_path)
    output_dir = base_output_path.parent
    base_stem = base_output_path.stem

    for mesh_id, mesh in meshes.items():
        out_name = f"{base_stem}_{sanitize_component(mesh_id)}.obj"
        out_path = output_dir / out_name
        mesh.export(str(out_path))
        click.echo(f"Wrote {out_path}")

        if _REMESH_BIN.exists():
            remeshed_path = out_path.with_stem(out_path.stem + "-remeshed")
            subprocess.run(
                [str(_REMESH_BIN), str(out_path), str(remeshed_path), str(target_edge_length)],
                check=True,
            )
            click.echo(f"Wrote {remeshed_path}")


if __name__ == "__main__":
    main()
