from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import glob

import click
import numpy as np
import trimesh
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.GC import GC_MakePlane
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopoDS import TopoDS_Shell, topods
from OCC.Core.gp import gp_Pnt


@dataclass(frozen=True)
class ObjMesh:
    vertices: np.ndarray
    faces: np.ndarray


def load_obj_mesh(obj_path: Path) -> ObjMesh:
    mesh = trimesh.load_mesh(str(obj_path), file_type="obj", process=False, maintain_order=True)

    assert isinstance(mesh, trimesh.Trimesh)  # i.e. don't support multi-object meshes loaded as trimesh.Scene

    vertices_array = np.asarray(mesh.vertices)
    faces_array = np.asarray(mesh.faces)

    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise click.ClickException(f"{obj_path}: expected Nx3 vertices, got shape {vertices_array.shape}.")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise click.ClickException(f"{obj_path}: expected triangulated faces, got shape {faces_array.shape}.")

    for face in faces_array:
        if len({int(face[0]), int(face[1]), int(face[2])}) != 3:
            raise click.ClickException(f"{obj_path}: degenerate face with repeated vertices.")

    triangle_vectors_1 = vertices_array[faces_array[:, 1]] - vertices_array[faces_array[:, 0]]
    triangle_vectors_2 = vertices_array[faces_array[:, 2]] - vertices_array[faces_array[:, 0]]
    double_areas = np.linalg.norm(np.cross(triangle_vectors_1, triangle_vectors_2), axis=1)
    if np.any(double_areas <= 0.0):
        raise click.ClickException(f"{obj_path}: mesh contains zero-area triangles.")

    if not mesh.is_watertight:
        raise click.ClickException(f"{obj_path}: mesh is not watertight.")
    if not mesh.is_winding_consistent:
        raise click.ClickException(f"{obj_path}: mesh has inconsistent face winding.")

    return ObjMesh(vertices=vertices_array, faces=faces_array)


def build_occ_shape(mesh: ObjMesh):
    points = [gp_Pnt(float(vertex[0]), float(vertex[1]), float(vertex[2])) for vertex in mesh.vertices]
    shared_edges = {}
    edge_counts: Counter[tuple[int, int]] = Counter()
    vertex_shapes = [BRepBuilderAPI_MakeVertex(point).Vertex() for point in points]

    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)

    for face_indices in mesh.faces:
        a, b, c = (int(face_indices[0]), int(face_indices[1]), int(face_indices[2]))
        directed_pairs = ((a, b), (b, c), (c, a))
        wire_builder = BRepBuilderAPI_MakeWire()

        for start, end in directed_pairs:
            edge_key = (min(start, end), max(start, end))
            edge_counts[edge_key] += 1
            edge = shared_edges.get(edge_key)
            if edge is None:
                edge = BRepBuilderAPI_MakeEdge(
                    vertex_shapes[start],
                    vertex_shapes[end],
                ).Edge()
                shared_edges[edge_key] = edge

            directed_edge = edge if (start, end) == edge_key else topods.Edge(edge.Reversed())
            wire_builder.Add(directed_edge)

        if not wire_builder.IsDone():
            raise click.ClickException(f"Failed to build wire for face {face_indices}.")

        plane = GC_MakePlane(points[a], points[b], points[c]).Value()
        face_builder = BRepBuilderAPI_MakeFace(plane, wire_builder.Wire(), True)
        if not face_builder.IsDone():
            raise click.ClickException(f"Failed to build face for triangle {face_indices}.")
        face = face_builder.Face()
        builder.Add(shell, face)

    invalid_edges = [edge for edge, count in edge_counts.items() if count != 2]
    if invalid_edges:
        raise click.ClickException(
            "Mesh is not a closed 2-manifold; "
            f"{len(invalid_edges)} edges do not have exactly two incident triangles."
        )

    if not BRepCheck_Analyzer(shell, True).IsValid():
        raise click.ClickException("Constructed shell is invalid; cannot form a solid.")

    solid_builder = BRepBuilderAPI_MakeSolid(shell)
    if not solid_builder.IsDone():
        raise click.ClickException("Failed to convert closed shell into a solid.")

    solid = solid_builder.Solid()
    if not BRepCheck_Analyzer(solid, True).IsValid():
        raise click.ClickException("Constructed solid is invalid.")
    return solid


def write_step(shape, output_path: Path) -> None:
    Interface_Static.SetCVal("write.step.schema", "AP242DIS")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(output_path))
    if status != IFSelect_RetDone:
        raise click.ClickException(f"Failed to write STEP file: {output_path}")


def iter_matching_objs(pattern: str) -> list[Path]:
    matches = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    files = [path for path in matches if path.is_file()]
    if not files:
        raise click.ClickException(f"No OBJ files matched pattern: {pattern}")
    return files


@click.command()
@click.argument("pattern", type=str)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for STEP output. Defaults to each OBJ's parent directory.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing STEP files.")
def main(pattern: str, output_dir: Path | None, overwrite: bool) -> None:
    """Convert triangulated watertight OBJ meshes matching PATTERN into STEP files."""
    obj_paths = iter_matching_objs(pattern)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for obj_path in obj_paths:
        target_dir = output_dir if output_dir is not None else obj_path.parent
        step_path = target_dir / f"{obj_path.stem}.step"
        if step_path.exists() and not overwrite:
            raise click.ClickException(
                f"Refusing to overwrite existing STEP file: {step_path}"
            )

        mesh = load_obj_mesh(obj_path)
        shape = build_occ_shape(mesh)
        write_step(shape, step_path)
        click.echo(f"Wrote {step_path}")


if __name__ == "__main__":
    main()
