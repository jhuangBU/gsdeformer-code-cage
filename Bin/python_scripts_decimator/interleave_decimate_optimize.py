"""Interleave Broxy decimation with cage optimization in fixed-size chunks.

Workflow per round:
1) Decimate source mesh for N steps (or until decimator reports done).
2) Treat the current decimated mesh as the cage candidate and optimize it for M epochs.
3) Feed optimized cage vertices back into decimator state.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import meshio
import numpy as np
import pymesh
import broxy_decimate
import torch
from tqdm.auto import tqdm

from pytorch_points.network.geo_operations import mean_value_coordinates_3D
from scene.gaussian_model import GaussianModel
from scene.axes_pts import compute_axes_pts
from optimize_cage_cagenerf import (
    CageTrainer,
    CageTrainerConfig,
    set_reproducibility,
)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = meshio.read(str(path))
    triangles = mesh.cells_dict.get("triangle")
    if triangles is None:
        raise ValueError(f"Expected triangle faces in mesh: {path}")

    vertices = np.asarray(mesh.points, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    return vertices, faces


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_mesh = meshio.Mesh(points=vertices, cells=[("triangle", faces)])
    meshio.write(str(path), out_mesh)


def save_cage_mesh(path: Path, cage_v, cage_f) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = cage_v.detach().cpu().numpy().squeeze(0)
    faces = cage_f.detach().cpu().numpy().squeeze(0)
    mesh = pymesh.form_mesh(vertices, faces)
    pymesh.save_mesh(str(path), mesh)


def build_decimation_params(args: argparse.Namespace) -> broxy_decimate.DecimationParams:
    params = broxy_decimate.DecimationParams()
    params.target_error = float(args.target_error)
    params.target_num_faces = int(args.target_num_faces)
    params.target_num_vertices = int(args.target_num_vertices)
    params.max_edge_length_alpha = float(args.max_edge_length_alpha)
    params.global_scale = float(args.global_scale)
    return params


OPTIMIZE_VERTEX_LIMIT = 1000


class InterleavedTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        
        self.source_mesh_path = Path(self.args.source_mesh)
        self.optimize_source_mesh_path = Path(self.args.optimize_source_mesh or self.args.source_mesh)
        
        self.output_root = Path(self.args.output_folder)
        self.decimate_dir = self.output_root / "decimated_meshes"
        self.optimize_dir = self.output_root / "optimize_rounds"
        self.cage_dir = self.output_root / "cages"

        vertices, faces = load_mesh(self.source_mesh_path)
        self.decimator = broxy_decimate.Decimator(build_decimation_params(self.args), vertices, faces)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        source_mesh = pymesh.load_mesh(str(self.optimize_source_mesh_path))
        self.source_vertices_full = torch.from_numpy(source_mesh.vertices).unsqueeze(0).float().to(self.device)

        self.axes_pts_query: Optional[torch.Tensor] = None
        if self.args.gaussian_ply:
            model = GaussianModel(sh_degree=3)
            model.load_ply(self.args.gaussian_ply)
            axes_pts = compute_axes_pts(model, center_only=True)  # (N, 1, 3)
            N, P, _ = axes_pts.shape
            self.axes_pts_query = axes_pts.reshape(1, N * P, 3).to(self.device)

        self.round_idx = 0
        self.total_decimation_steps = 0
        self.total_optimization_epochs = 0
        self.final_cage_path: Optional[Path] = None

    def _prepare_output_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.decimate_dir.mkdir(parents=True, exist_ok=True)
        self.optimize_dir.mkdir(parents=True, exist_ok=True)
        self.cage_dir.mkdir(parents=True, exist_ok=True)

    def _save_current_decimated_mesh(self) -> tuple[Path, np.ndarray, np.ndarray]:
        current_vertices = self.decimator.get_current_vertices()
        current_faces = self.decimator.get_current_faces()
        decimated_mesh_path = self.decimate_dir / f"round_{self.round_idx:04d}_preopt.ply"
        write_mesh(decimated_mesh_path, current_vertices, current_faces)
        return decimated_mesh_path, current_vertices, current_faces

    def _report_axes_pts_mvc(self, tag: str) -> dict:
        """Compute MVC stats for axes-pt centers against the current cage.

        Reports: number of vertices with negative MVC, min negative MVC, mean negative MVC.
        """
        cage_vertices = self.decimator.get_current_vertices()
        cage_faces = self.decimator.get_current_faces()
        cage_v = torch.from_numpy(cage_vertices).unsqueeze(0).float().to(self.device)
        cage_f = torch.from_numpy(cage_faces).unsqueeze(0).long().to(self.device)

        query_pts = self.axes_pts_query
        _, N, _ = query_pts.shape
        bs = max(1, int(self.args.mvc_batch))

        neg_count = 0
        neg_min = float("inf")
        neg_sum = 0.0
        neg_total = 0
        total_entries = 0

        with torch.no_grad():
            for start in range(0, N, bs):
                end = min(start + bs, N)
                pts = query_pts[:, start:end, :].contiguous()
                w = mean_value_coordinates_3D(pts, cage_v, cage_f, verbose=False)
                total_entries += int(w.numel())
                neg_flags = (w < 0).any(dim=-1)
                neg_count += int(neg_flags.sum().item())
                negative_weights = w[w < 0]
                if negative_weights.numel() > 0:
                    chunk_min = float(negative_weights.min().item())
                    if chunk_min < neg_min:
                        neg_min = chunk_min
                    neg_sum += float(negative_weights.sum().item())
                    neg_total += int(negative_weights.numel())

        min_neg = neg_min if neg_total > 0 else float("nan")
        mean_neg = (neg_sum / neg_total) if neg_total > 0 else float("nan")
        neg_entry_pct = (neg_total / total_entries * 100) if total_entries > 0 else float("nan")

        label = f"[{tag}] " if tag else ""
        print(f"[MVC] {label}Negative-MVC vertices: {neg_count} / {N}")
        print(f"[MVC] {label}Negative-MVC entries: {neg_total} / {total_entries} ({neg_entry_pct:.2f}%)")
        print(f"[MVC] {label}Min negative MVC: {min_neg:.8e}")
        print(f"[MVC] {label}Mean negative MVC: {mean_neg:.8e}")

        report_path = self.output_root / f"mvc_axes_pts_report_{tag}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"negative_vertices={neg_count}\n")
            f.write(f"total_vertices={N}\n")
            f.write(f"negative_entries={neg_total}\n")
            f.write(f"total_entries={total_entries}\n")
            f.write(f"negative_entry_pct={neg_entry_pct:.2f}\n")
            f.write(f"min_negative_mvc={min_neg:.8e}\n")
            f.write(f"mean_negative_mvc={mean_neg:.8e}\n")
        print(f"[MVC] {label}Report saved: {report_path}")

        return {"neg_count": neg_count, "total": N, "neg_total": neg_total, "total_entries": total_entries, "neg_entry_pct": neg_entry_pct, "min_negative_mvc": min_neg, "mean_negative_mvc": mean_neg}

    def _write_summary(self) -> Path:
        summary_path = self.output_root / "interleave_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"rounds={self.round_idx}\n")
            f.write(f"total_decimation_steps={self.total_decimation_steps}\n")
            f.write(f"total_optimization_epochs={self.total_optimization_epochs}\n")
            f.write(f"final_cage={self.final_cage_path if self.final_cage_path is not None else 'N/A'}\n")
            f.write(f"final_decimation_error={self.decimator.get_current_error()}\n")
            f.write(f"final_num_faces={len(self.decimator.get_current_faces())}\n")
            f.write(f"final_num_vertices={len(self.decimator.get_current_vertices())}\n")
            f.write(
                "decimation_params="
                f"target_error:{self.args.target_error},"
                f"target_num_faces:{self.args.target_num_faces},"
                f"target_num_vertices:{self.args.target_num_vertices},"
                f"max_edge_length_alpha:{self.args.max_edge_length_alpha},"
                f"global_scale:{self.args.global_scale}\n"
            )
        return summary_path

    def run(self) -> None:
        self._prepare_output_dirs()
        if self.axes_pts_query is not None:
            self._report_axes_pts_mvc("process_pre")
        pbar = tqdm(desc="Interleave rounds", unit="round")

        preloop_steps = int(self.decimator.step_until_vertex_threshold(OPTIMIZE_VERTEX_LIMIT))
        self.total_decimation_steps += preloop_steps

        while not self.decimator.is_done():
            self.round_idx += 1

            decimation_steps_this_round = 0
            for _ in range(self.args.decimate_steps):
                if self.decimator.is_done():
                    break
                self.decimator.step()
                decimation_steps_this_round += 1
                self.total_decimation_steps += 1

            decimated_mesh_path, current_vertices, current_faces = self._save_current_decimated_mesh()
            current_error = self.decimator.get_current_error()
            pbar.update(1)
            pbar.set_postfix(vertices=len(current_vertices), error=f"{current_error:.6f}")

            if len(current_vertices) > OPTIMIZE_VERTEX_LIMIT:
                continue

            print(
                f"[Round {self.round_idx}] decimation steps={decimation_steps_this_round}, "
                f"error={current_error:.6f}, "
                f"faces={len(current_faces)}, vertices={len(current_vertices)}"
            )

            trainer_cfg = CageTrainerConfig(
                source_cage=str(self.optimize_source_mesh_path),
                template_cage=str(decimated_mesh_path),
                lr=self.args.lr,
                nepochs=self.args.optimize_steps,
                l2_weight=self.args.l2_weight,
                mvc_weight=self.args.mvc_weight,
                output_folder=str(self.optimize_dir / f"round_{self.round_idx:04d}"),
                n_sample=self.args.n_sample,
                mvc_batch=self.args.mvc_batch,
                seed=self.args.seed,
                enable_mvc_report=False,
                axes_pts_query=self.axes_pts_query,
            )
            trainer = CageTrainer(trainer_cfg)
            cage_v, cage_f = trainer.run()
            self.total_optimization_epochs += self.args.optimize_steps

            cage_path = self.cage_dir / f"round_{self.round_idx:04d}.ply"
            save_cage_mesh(cage_path, cage_v, cage_f)
            if self.args.optimize_steps != 0:
                self.decimator.update_mesh(cage_v.detach().cpu().numpy().squeeze(0).astype(np.float32))
            self.final_cage_path = cage_path
            print(f"[Round {self.round_idx}] optimized cage saved to {cage_path}")

        pbar.close()
        if self.axes_pts_query is not None:
            self._report_axes_pts_mvc("process_post")
        summary_path = self._write_summary()
        print(f"Interleaving complete after {self.round_idx} rounds.")
        print(f"Summary saved to {summary_path}")
        if self.final_cage_path is not None:
            print(f"Final cage: {self.final_cage_path}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interleave mesh decimation and cage optimization."
    )
    parser.add_argument("--source_mesh", type=str, required=True, help="Input mesh to decimate.")
    parser.add_argument(
        "--optimize_source_mesh",
        type=str,
        default=None,
        help="Mesh used as optimization supervision. Defaults to --source_mesh.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output root for all round artifacts.",
    )

    parser.add_argument(
        "--gaussian_ply",
        type=str,
        default=None,
        help="3DGS PLY file. When provided, computes before/after MVC stats for Gaussian centers.",
    )

    parser.add_argument("--decimate_steps", type=int, default=10, help="Decimation steps per round.")
    parser.add_argument("--optimize_steps", type=int, default=10, help="Optimization epochs per round.")

    parser.add_argument("--target_error", type=float, default=5.0, help="Decimation target error.")
    parser.add_argument("--target_num_faces", type=int, default=20, help="Decimation target face count.")
    parser.add_argument("--target_num_vertices", type=int, default=150, help="Decimation target vertex count.")
    parser.add_argument("--max_edge_length_alpha", type=float, default=2.5, help="Decimation max edge alpha.")
    parser.add_argument("--global_scale", type=float, default=0.0629838, help="Decimation global scale.")

    parser.add_argument("--lr", type=float, default=1e-3, help="Optimization learning rate.")
    parser.add_argument("--l2_weight", type=float, default=0.0, help="L2 regularization weight.")
    parser.add_argument("--mvc_weight", type=float, default=0.0, help="MVC regularization weight.")
    parser.add_argument("--n_sample", type=int, default=256, help="Source mesh samples per optimization step.")
    parser.add_argument("--mvc_batch", type=int, default=256, help="MVC eval batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    set_reproducibility(args.seed)
    InterleavedTrainer(args).run()


if __name__ == "__main__":
    main()
