"""Compare mesh components and self-intersections across cage meshes."""
import _hack_fix_pymesh_dep  # noqa: F401  (pre-load libboost for pymesh)

from pathlib import Path
from typing import List, Tuple

import numpy as np
import meshio
import pymesh
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / ".." / "experiments-caging-results"

CAGES = [  # type: List[Tuple[str, str, Path]]
    ("lego",      "baseline", RESULTS_DIR / "lego-baseline-cage.ply"),
    ("lego",      "ours",     RESULTS_DIR / "lego-ours-cage.ply"),
    ("vase",      "baseline", RESULTS_DIR / "vase-baseline-cage.ply"),
    ("vase",      "ours",     RESULTS_DIR / "vase-ours-cage.ply"),
    ("vase_2dgs", "baseline", RESULTS_DIR / "vase_2dgs-baseline-cage.ply"),
    ("vase_2dgs", "ours",     RESULTS_DIR / "vase_2dgs-ours-cage.ply"),
]


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(str(path))
    verts = np.asarray(mesh.points, dtype=np.float64)
    faces = np.asarray(mesh.cells_dict["triangle"], dtype=np.int32)
    return verts, faces


def compute_mesh_components(verts: np.ndarray, faces: np.ndarray) -> int:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts[:, :3])
    mesh.triangles = o3d.utility.Vector3iVector(faces[:, :3])
    cluster_ids, _, _ = mesh.cluster_connected_triangles()
    return len(set(cluster_ids))


def compute_self_intersection(verts: np.ndarray, faces: np.ndarray) -> Tuple[int, int]:
    pm = pymesh.form_mesh(verts[:, :3], faces[:, :3])
    si_pairs = pymesh.detect_self_intersection(pm)
    n_pairs = len(si_pairs)
    n_faces = len(set(si_pairs.flatten().tolist())) if n_pairs > 0 else 0
    return n_pairs, n_faces


def main() -> None:
    print("{:<12} {:<10} {:>10} {:>12} {:>10}".format(
        "scene", "method", "components", "si_pairs", "si_faces"))
    print("-" * 58)

    for scene, method, path in CAGES:
        if not path.is_file():
            print("{:<12} {:<10} MISSING".format(scene, method))
            continue
        verts, faces = load_mesh(path)
        nc = compute_mesh_components(verts, faces)
        sip, sif = compute_self_intersection(verts, faces)
        print("{:<12} {:<10} {:>10} {:>12} {:>10}".format(
            scene, method, nc, sip, sif))


if __name__ == "__main__":
    main()
