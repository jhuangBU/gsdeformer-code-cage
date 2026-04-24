import torch
import numpy as np
from scene.gaussian_model import GaussianModel
from utils.rotation_conversions import quaternion_to_matrix

def deform_compute_ellipsoid_pts_to_deform(old_xyz:torch.Tensor, ep_axes: torch.Tensor, ep_scale: torch.Tensor) -> torch.Tensor:
    ep_mean = old_xyz
    pts = (ep_mean[:, ..., np.newaxis] + ep_axes * ep_scale[:, np.newaxis, ...]).permute(0, 2, 1)
    pts2 = (ep_mean[:, ..., np.newaxis] - ep_axes * ep_scale[:, np.newaxis, ...]).permute(0, 2, 1)
    axes_pts = torch.concat([ep_mean[:, np.newaxis, ...], pts, pts2], dim=1)
    axes_pts = axes_pts.to(torch.float32)
    return axes_pts

def compute_axes_pts(model: GaussianModel, center_only: bool = False):
    old_xyz = model._xyz.detach().clone()
    old_rot = model.rotation_activation(model._rotation.detach())
    old_scaling = model.scaling_activation(model._scaling.detach())

    ep_mean = old_xyz
    ep_axes = quaternion_to_matrix(old_rot)
    ep_scale = old_scaling
    axes_pts = deform_compute_ellipsoid_pts_to_deform(ep_mean, ep_axes, ep_scale)

    if center_only:
        return axes_pts[:, :1, :]  # (N_gaussians, 1, 3)

    return axes_pts  # (N_gaussians, 7, 3)