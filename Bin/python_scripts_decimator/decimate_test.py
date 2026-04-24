from pathlib import Path

import numpy as np
import meshio
from typing import Tuple

import broxy_decimate


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(str(path))
    triangles = mesh.cells_dict.get("triangle")
    if triangles is None:
        raise ValueError("Expected triangle faces in the OFF file.")

    vertices = np.asarray(mesh.points, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    return vertices, faces


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    out_mesh = meshio.Mesh(points=vertices, cells=[("triangle", faces)])
    meshio.write(str(path), out_mesh)


def main() -> None:
    input_path = Path("data/decimate_mesh_input.off")
    output_path = Path("decimated_mesh.ply")
    ref_path = Path("data/decimate_mesh_ref.ply")

    vertices, faces = load_mesh(input_path)

    params = broxy_decimate.DecimationParams()
    params.target_error = 5.0
    params.target_num_faces = 20
    params.target_num_vertices = 150
    params.max_edge_length_alpha = 2.5
    params.global_scale = 0.0629838

    decimator = broxy_decimate.Decimator(params, vertices, faces)
    iteration = 0
    while True:
        if decimator.is_done():
            break
        decimator.step()
        iteration += 1
        print("iteration:", iteration, "-", "current error:", decimator.get_current_error())
        if iteration % 10 == 0:
            print("test: running updating mesh")
            decimator.update_mesh(decimator.get_current_vertices())

    current_vertices = decimator.get_current_vertices()
    current_faces = decimator.get_current_faces()

    write_mesh(output_path, current_vertices, current_faces)

    print(
        "Feature-Aware Decimator error :",
        decimator.get_current_error(),
        "and number of faces :",
        len(current_faces),
        "and number of vertices :",
        len(current_vertices),
    )

    if ref_path.exists():
        ref_vertices, ref_faces = load_mesh(ref_path)
        assert ref_vertices.shape == current_vertices.shape, "Vertex count mismatch with reference mesh."
        assert np.array_equal(ref_faces, current_faces), "Face mismatch with reference mesh."
        diffs = current_vertices - ref_vertices
        dists = np.linalg.norm(diffs, axis=1)
        print(
            "Vertex distance to reference (mean/max):",
            float(np.mean(dists)),
            float(np.max(dists)),
        )
    else:
        print("Reference mesh not found; skipping comparison.")


if __name__ == "__main__":
    main()
