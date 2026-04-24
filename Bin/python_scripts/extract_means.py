import argparse
from pathlib import Path

import numpy

from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration


def load_model(model_folder: Path, iteration: int = -1):
    if iteration == -1:
        iteration = searchForMaxIteration(str(model_folder / "point_cloud"))
    print(f"Loading trained model at iteration {iteration}")
    model = GaussianModel(sh_degree=3)
    model.load_ply(str(model_folder / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"))
    return model


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=Path)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output_file", type=Path)
    args = parser.parse_args(args)

    model_folder: Path = args.model_folder
    iteration: int = args.iteration
    output_file: Path = args.output_file

    model = load_model(model_folder, iteration)

    xyz = model.get_xyz.detach().cpu().numpy()
    numpy.save(str(output_file), xyz)


if __name__ == '__main__':
    main()