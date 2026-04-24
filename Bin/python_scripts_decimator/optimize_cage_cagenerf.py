""" Optimize the initial cage for a new source shape """
import shutil
import random
import torch
import os
import numpy as np
import pymesh
import argparse
import open3d as o3d
from dataclasses import dataclass, field
from typing import Optional
from common import loadInitCage
from pytorch_points.network.geo_operations import mean_value_coordinates_3D
from losses import MVCRegularizer
from torch.utils.tensorboard import SummaryWriter


def set_reproducibility(seed):
    """Set RNG seeds and deterministic flags for reproducible optimization."""
    curr_seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(curr_seed)
    # Needed by some CUDA kernels for deterministic behavior.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(curr_seed)
    np.random.seed(curr_seed)
    torch.manual_seed(curr_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(curr_seed)
        torch.cuda.manual_seed_all(curr_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    if hasattr(o3d, "utility") and hasattr(o3d.utility, "random"):
        try:
            o3d.utility.random.seed(curr_seed)
        except Exception:
            pass


def _mvc_negative_stats(source_vertices_full: torch.Tensor,
                        cage_v: torch.Tensor,
                        cage_f: torch.Tensor,
                        batch_size: int = 65536):
    """Compute negative MVC statistics over all mesh vertices in batches.

    Returns dict: {neg_count, total, neg_ratio, highest_negative_mvc, mean_negative_mvc}
    """
    results = {
        "neg_count": 0,
        "total": 0,
        "neg_ratio": 0.0,
        "highest_negative_mvc": float("nan"),
        "mean_negative_mvc": float("nan"),
    }
    with torch.no_grad():
        B, Nv, _ = source_vertices_full.shape
        total = B * Nv
        results["total"] = int(total)
        bs = max(1, int(batch_size))
        neg_count = 0
        neg_sum = 0.0
        neg_total = 0
        highest_negative_mvc = float("nan")
        for start in range(0, Nv, bs):
            end = min(start + bs, Nv)
            pts = source_vertices_full[:, start:end, :].contiguous()
            w = mean_value_coordinates_3D(pts, cage_v, cage_f, verbose=False)
            neg_flags = (w < 0).any(dim=-1)  # (B, end-start)
            neg_count += int(neg_flags.sum().item())
            negative_weights = w[w < 0]
            if negative_weights.numel() > 0:
                chunk_highest = float(negative_weights.max().item())
                if np.isnan(highest_negative_mvc) or chunk_highest > highest_negative_mvc:
                    highest_negative_mvc = chunk_highest
                neg_sum += float(negative_weights.sum().item())
                neg_total += int(negative_weights.numel())
        results["neg_count"] = int(neg_count)
        results["neg_ratio"] = (neg_count / total) if total > 0 else 0.0
        if neg_total > 0:
            results["highest_negative_mvc"] = highest_negative_mvc
            results["mean_negative_mvc"] = neg_sum / neg_total
    return results


def _mvc_negative_report(source_vertices_full: torch.Tensor,
                         cage_v: torch.Tensor,
                         cage_f: torch.Tensor,
                         batch_size: int = 65536,
                         output_folder: str = None,
                         tag: str = None):
    """Compute MVC negative stats and optionally print/save a report."""
    results = {
        "neg_count": 0,
        "total": 0,
        "neg_ratio": 0.0,
        "highest_negative_mvc": float("nan"),
        "mean_negative_mvc": float("nan"),
    }
    try:
        results = _mvc_negative_stats(source_vertices_full, cage_v, cage_f, batch_size)
        label = f"[{tag}] " if tag else ""
        print(f"[MVC] {label}Negative-weights vertices: {results['neg_count']} / {results['total']} ({results['neg_ratio']*100:.2f}%)")
        print(
            f"[MVC] {label}highest_negative_mvc={results['highest_negative_mvc']:.8e}, "
            f"mean_negative_mvc={results['mean_negative_mvc']:.8e}"
        )

        if output_folder is not None:
            try:
                os.makedirs(output_folder, exist_ok=True)
                suffix = f"_{tag}" if tag else ""
                report_path = os.path.join(output_folder, f"mvc_negative_report{suffix}.txt")
                with open(report_path, 'w') as f:
                    f.write(f"negative_vertices={results['neg_count']}\n")
                    f.write(f"total_vertices={results['total']}\n")
                    f.write(f"negative_ratio={results['neg_ratio']:.6f}\n")
                    f.write(f"highest_negative_mvc={results['highest_negative_mvc']:.8e}\n")
                    f.write(f"mean_negative_mvc={results['mean_negative_mvc']:.8e}\n")
                print(f"[MVC] {label}Report saved: {report_path}")
            except Exception as e:
                print(f"[MVC] {label}Warning: failed to save report: {e}")
    except Exception as e:
        print(f"[MVC] Warning: failed to compute MVC {f'({tag})' if tag else ''}: {e}")

    return results


@dataclass
class CageTrainerConfig:
    source_cage: str
    template_cage: str
    lr: float = 1e-3
    nepochs: int = 1000
    l2_weight: float = 0.0
    mvc_weight: float = 0.0
    output_folder: str = None
    n_sample: int = 1024
    mvc_batch: int = 65536
    seed: int = 42
    enable_mvc_report: bool = True
    axes_pts_query: Optional[torch.Tensor] = None


class CageTrainer:
    
    # init #
    
    def __init__(self, cfg: CageTrainerConfig):
        self.cfg = cfg

        # setup logging
        os.makedirs(self.cfg.output_folder, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.cfg.output_folder)

        # load cage & model
        self.cage_v, self.cage_f = self._load_cage()
        self.o3d_source_mesh, self.source_vertices_full = self._load_source_mesh()
        if self.cfg.axes_pts_query is not None:
            self.source_vertices_full = self.cfg.axes_pts_query.to(device=self.cage_v.device)
        self.cage_v.requires_grad_(True)
        self.cage_init = self.cage_v.clone().detach()

        # create losses used in run.sh
        self.mvc_reg_loss = MVCRegularizer(threshold=50, beta=1.0, alpha=0.0)
        
        # create optimizer and LR
        self.optimizer = torch.optim.Adam([self.cage_v], lr=self.cfg.lr, betas=(0.5, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, int(self.cfg.nepochs * 0.4), gamma=0.5, last_epoch=-1
        )

    def _load_cage(self):
        init_cage_V, init_cage_Fs = loadInitCage([self.cfg.template_cage], preprocess=False)
        return init_cage_V, init_cage_Fs[0]

    def _load_source_mesh(self):
        o3d_source_mesh = o3d.io.read_triangle_mesh(self.cfg.source_cage)
        source_mesh = pymesh.load_mesh(self.cfg.source_cage)
        source_vertices_full = torch.from_numpy(source_mesh.vertices).unsqueeze(0).float()
        if self.cfg.enable_mvc_report:
            source_vertices_full = source_vertices_full.to(device=self.cage_v.device)
        return o3d_source_mesh, source_vertices_full
    
    # lifecycle #

    def run(self):
        self._cb_training_start()

        try:
            for t in range(self.cfg.nepochs):
                self._train_step(t)
        finally:
            self._cb_training_end()

        return self.cage_v, self.cage_f
    
    
    def _cb_training_start(self):
        self._save_initial_artifacts()
        if not self.cfg.enable_mvc_report:
            return
        _mvc_negative_report(
            source_vertices_full=self.source_vertices_full,
            cage_v=self.cage_init,
            cage_f=self.cage_f,
            batch_size=self.cfg.mvc_batch,
            output_folder=self.cfg.output_folder,
            tag='pre',
        )

    def _cb_training_end(self):
        if not self.cfg.enable_mvc_report:
            if self.writer is not None:
                self.writer.close()
            return
        _mvc_negative_report(
            source_vertices_full=self.source_vertices_full,
            cage_v=self.cage_v.detach(),
            cage_f=self.cage_f,
            batch_size=self.cfg.mvc_batch,
            output_folder=self.cfg.output_folder,
            tag='post',
        )
        if self.writer is not None:
            self.writer.close()

    def _save_initial_artifacts(self):
        try:
            init_vertices = self.cage_init.detach().cpu().numpy().squeeze(0)
            init_faces = self.cage_f.cpu().numpy().squeeze(0)
            init_mesh = pymesh.form_mesh(init_vertices, init_faces)
            init_cage_path = os.path.join(self.cfg.output_folder, "init_cage.ply")
            pymesh.save_mesh(init_cage_path, init_mesh)

            src_ext = os.path.splitext(self.cfg.source_cage)[1] or ".ply"
            dst_src_path = os.path.join(self.cfg.output_folder, f"init_src_mesh{src_ext}")
            shutil.copyfile(self.cfg.source_cage, dst_src_path)
            print(f"Saved initial cage to {init_cage_path} and source mesh to {dst_src_path}")
        except Exception as e:
            print(f"Warning: failed to save initial meshes: {e}")

    # training #
    
    def _train_step(self, t):
        self.optimizer.zero_grad(set_to_none=True)
        self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], t)

        sampled_points = self._sample_source_points()
        weights = mean_value_coordinates_3D(sampled_points, self.cage_v, self.cage_f, verbose=False)
        reg = torch.zeros(1, device=self.cage_v.device)

        l2_reg_val = torch.sum((self.cage_init - self.cage_v) ** 2)
        mvc_reg_val = self.mvc_reg_loss(weights)

        self.writer.add_scalar('loss/l2_reg', l2_reg_val.item(), t)
        if self.cfg.l2_weight > 0:
            reg += self.cfg.l2_weight * l2_reg_val

        self.writer.add_scalar('loss/mvc_reg', mvc_reg_val.item(), t)
        if self.cfg.mvc_weight > 0:
            reg += self.cfg.mvc_weight * mvc_reg_val

        loss = reg
        self.writer.add_scalar('loss/total_reg', reg.item(), t)
        self.writer.add_scalar('loss/total', loss.item(), t)

        self._cb_log_full_mvc(t)
        self._cb_print_progress(t, loss, reg)
        self._cb_save_checkpoint(t)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
    
    def _sample_source_points(self):
        sampled_pcd = self.o3d_source_mesh.sample_points_uniformly(number_of_points=self.cfg.n_sample)
        sampled_np = np.asarray(sampled_pcd.points, dtype=np.float32)
        return torch.from_numpy(sampled_np).unsqueeze(0).to(device=self.cage_v.device)

    def _cb_log_full_mvc(self, t):
        if not self.cfg.enable_mvc_report:
            return
        if (t + 1) % 100 != 0:
            return
        stats_full = _mvc_negative_stats(
            source_vertices_full=self.source_vertices_full,
            cage_v=self.cage_v.detach(),
            cage_f=self.cage_f,
            batch_size=self.cfg.mvc_batch,
        )
        self.writer.add_scalar('mvc/neg_count_full', stats_full['neg_count'], t)
        self.writer.add_scalar('mvc/neg_count_ratio', stats_full['neg_ratio'], t)

    def _cb_print_progress(self, t, loss, reg):
        if (t + 1) % 100 != 0:
            return

        print("t {}/{} mvc_loss: {} reg: {}".format(t, self.cfg.nepochs, loss.item(), reg.item()))

    def _cb_save_checkpoint(self, t):
        if (t + 1) % 100 != 0:
            return

        save_path = os.path.join(self.cfg.output_folder, f"iter_{t+1}.ply")
        cage_vertices = self.cage_v.detach().cpu().numpy().squeeze(0)
        cage_faces = self.cage_f.cpu().numpy().squeeze(0)
        save_mesh = pymesh.form_mesh(cage_vertices, cage_faces)
        pymesh.save_mesh(save_path, save_mesh)


def main():
    parser = argparse.ArgumentParser(description='Optimize cage for source shape')
    parser.add_argument('--source_cage', type=str, required=True,
                        help='Path to source cage file')
    parser.add_argument('--template_cage', type=str, required=True,
                        help='Path to template cage file')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--nepochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--l2_weight', type=float, default=0.0,
                        help='L2 regularization weight')
    parser.add_argument('--mvc_weight', type=float, default=0.0,
                        help='MVC regularization weight')
    parser.add_argument('--output_folder', type=str, default=None,
                        help='Folder to save mesh checkpoints during optimization')
    parser.add_argument('--n_sample', type=int, default=1024,
                        help='Number of points sampled uniformly from source mesh per iteration (Open3D)')
    parser.add_argument('--mvc_batch', type=int, default=65536,
                        help='Batch size for post-optimization MVC evaluation over all source vertices')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global random seed for reproducible runs')

    opt = parser.parse_args()
    set_reproducibility(opt.seed)

    cfg = CageTrainerConfig(
        source_cage=opt.source_cage,
        template_cage=opt.template_cage,
        lr=opt.lr,
        nepochs=opt.nepochs,
        l2_weight=opt.l2_weight,
        mvc_weight=opt.mvc_weight,
        output_folder=opt.output_folder,
        n_sample=opt.n_sample,
        mvc_batch=opt.mvc_batch,
        seed=opt.seed,
    )
    CageTrainer(cfg).run()


if __name__ == "__main__":
    main()
