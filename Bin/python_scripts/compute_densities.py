import argparse
from pathlib import Path
from typing import Optional

import numpy
import numpy as np
import torch
import tqdm

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.ops import knn_points


def get_gaussians_closest_to_samples(model: GaussianModel, x: torch.Tensor, n_closest_gaussian: int = 16) -> torch.Tensor:
    points = model.get_xyz.detach().float().cuda()
    closest_gaussians_idx = knn_points(x[None], points[None], K=n_closest_gaussian).idx[0]
    return closest_gaussians_idx

def get_covariance(model: GaussianModel, return_full_matrix=False, return_sqrt=False, inverse_scales=False):
    scaling = model.get_scaling
    if inverse_scales:
        scaling = 1. / scaling.clamp(min=1e-8)
    scaled_rotation = quaternion_to_matrix(model.get_rotation) * scaling[:, None]
    if return_sqrt:
        return scaled_rotation

    cov3Dmatrix = scaled_rotation @ scaled_rotation.transpose(-1, -2)
    if return_full_matrix:
        return cov3Dmatrix

    cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=cov3Dmatrix.device)
    cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
    cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
    cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
    cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
    cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
    cov3D[:, 5] = cov3Dmatrix[:, 2, 2]

    return cov3D

def prune_model(smodel: GaussianModel, selector: torch.Tensor) -> GaussianModel:
    """prune gaussian model & related segmentation based on selector mask"""
    new_smodel = GaussianModel(smodel.max_sh_degree)
    new_smodel.active_sh_degree = smodel.active_sh_degree
    new_smodel.max_sh_degree = smodel.max_sh_degree
    # TODO: do not replace parameter with tensor
    new_smodel._xyz = smodel._xyz[selector, ...]
    new_smodel._features_dc = smodel._features_dc[selector, ...]
    new_smodel._features_rest = smodel._features_rest[selector, ...]
    new_smodel._scaling = smodel._scaling[selector, ...]
    new_smodel._rotation = smodel._rotation[selector, ...]
    new_smodel._opacity = smodel._opacity[selector, ...]
    new_smodel.max_radii2D = smodel.max_radii2D
    new_smodel.xyz_gradient_accum = smodel.xyz_gradient_accum
    new_smodel.denom = smodel.denom
    new_smodel.optimizer = smodel.optimizer
    new_smodel.percent_dense = smodel.percent_dense
    new_smodel.spatial_lr_scale = smodel.spatial_lr_scale
    return new_smodel


def compute_density(
        model: GaussianModel, x: torch.Tensor,
        closest_gaussians_idx: Optional[torch.Tensor]=None,
        density_factor: float=1.,
        return_closest_gaussian_opacities:bool=False
):
    if closest_gaussians_idx is None:
        closest_gaussians_idx = get_gaussians_closest_to_samples(model, x)

    # Gather gaussian parameters
    points = model.get_xyz.detach().float().cuda()
    close_gaussian_centers = points[closest_gaussians_idx]
    close_gaussian_inv_scaled_rotation = get_covariance(
        model,
        return_full_matrix=True, return_sqrt=True, inverse_scales=True
    )[closest_gaussians_idx]
    close_gaussian_strengths = model.get_opacity[closest_gaussians_idx]

    # Compute the density field as a sum of local gaussian opacities
    shift = (x[:, None] - close_gaussian_centers)
    warped_shift = close_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
    neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
    neighbor_opacities = density_factor * close_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
    densities = neighbor_opacities.sum(dim=-1)

    if return_closest_gaussian_opacities:
        return densities, neighbor_opacities
    else:
        return densities  # Shape is (n_points, )


def load_model(model_folder: Path, iteration: int = -1):
    if iteration == -1:
        iteration = searchForMaxIteration(str(model_folder / "point_cloud"))
    print(f"Loading trained model at iteration {iteration}")
    model = GaussianModel(sh_degree=3)
    model.load_ply(str(model_folder / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"))
    return model


@torch.no_grad()
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=Path)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--coord_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--batch_size", type=int, default=200_000)
    args = parser.parse_args(args)

    model_folder: Path = args.model_folder
    iteration: int = args.iteration
    coord_file: Path = args.coord_file
    output_file: Path = args.output_file
    batch_size: int = args.batch_size

    model = load_model(model_folder, iteration)
    model = prune_model(model, (model.get_opacity >= 0.01)[:,0])
    coords = torch.tensor(np.load(str(coord_file))).cuda()

    densities = []
    for batch in tqdm.tqdm(torch.split(coords, batch_size), desc="computing densities"):
        d = compute_density(model, batch)
        densities.append(d)
    densities = torch.cat(densities)

    np.save(str(output_file), densities.cpu().numpy())


if __name__ == '__main__':
    main()
