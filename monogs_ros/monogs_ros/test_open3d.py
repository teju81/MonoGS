from utils.dataset import load_dataset
from utils.camera_utils import Camera
from munch import munchify
import yaml
import sys
from argparse import ArgumentParser
import torch
import numpy as np
from gaussian_splatting.scene.gaussian_model import GaussianModel
from utils.config_utils import load_config
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2

def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    args = parser.parse_args(sys.argv[1:])
    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    model_params = munchify(config["model_params"])
    dataset = load_dataset(model_params, model_params.source_path, config=config)
    idx = 3
    print(dataset.color_paths[idx])
    print(dataset.depth_paths[idx])
    gt_color, gt_depth, gt_pose = dataset[idx]
    print(gt_color)
    print(gt_depth)
    print(gt_pose)

    gaussians = GaussianModel(model_params.sh_degree, config=config)

    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=dataset.fx,
        fy=dataset.fy,
        cx=dataset.cx,
        cy=dataset.cy,
        W=dataset.width,
        H=dataset.height,
    ).transpose(0, 1)
    projection_matrix = projection_matrix.to(device="cuda:0")
    viewpoint = Camera.init_from_dataset(dataset, idx, projection_matrix)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_img = viewpoint.original_image.cuda()
    valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

    # use the observed depth
    initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0).contiguous()
    initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
    depth_map = initial_depth[0].numpy()

    print("1")
    print(gt_img)
    print(depth_map)

    gaussians.extend_from_pcd_seq(viewpoint, kf_id=0, init=True, scale=2.0, depthmap=depth_map)
    print("2")


if __name__ == "__main__":
    main()