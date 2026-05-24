# Cage Building Code Dump for GSDeformer

cage building code dump for [GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting](https://arxiv.org/abs/2405.15491)

For main deformation code, please check [gsdeformer-code](https://github.com/jhuangBU/gsdeformer-code) 

Based on [superboubek/Broxy](https://github.com/superboubek/Broxy) (Bounding Proxies for Shape Approximation, SIGGRAPH 2017).

## Prerequisites

* An NVIDIA GPU with a recent driver installed on the host
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (required for the `--device nvidia.com/gpu=all` flag used by `setup-distrobox.sh create`)
* [distrobox](https://distrobox.it/#installation) installed on the host

If distrobox is not available, check setup-distrobox.sh for all the deps installed and reproduce them manually.

## Setup Environment

```bash
# Step 1: Run this script on the HOST to create the distrobox
./setup-distrobox.sh create

# Step 2: Enter the distrobox and run the install step
distrobox enter broxy
cd ~/Broxy
./setup-distrobox.sh install

# Step 3 (still inside distrobox): Build the project
./setup-distrobox.sh build
```

Step 4 (still inside distrobox): install mamba (via miniforge) and create the two Python envs the app invokes at runtime:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash "Miniforge3-$(uname)-$(uname -m).sh"
# re-open the shell so `mamba` is on PATH

mamba env create -f Bin/python_scripts/environment.yaml
mamba env create -f Bin/python_scripts_decimator/environment.yaml
```

## Usage

1. Start the GUI
```bash
# (inside distrobox): Run
./setup-distrobox.sh run
```
2. Load 3DGS

The folder to select is the training result folder from 3DGS — e.g. if the ply file is `output/nerf_lego/point_cloud/iteration_30000/point_cloud.ply` then the folder is `gs3d/output/nerf_lego`.

* in the opened window:
    * click "File" Button on the Menu Bar
    * click "Open 3DGS"
    * select the folder containing the 3DGS training result
    * set a value for Density Threshold, or keep default (only used in Conservative Conversion, TSDF ignores it)
    * Select TSDF in dialog (Conservative for opacity field based baseline)
    * wait till script completes and loaded

3. Adjust voxel grid under Broxy
* check the "Asymmetric" box in "Morphological Proxy" Group
* click "Increase Base" / "Decrease Base" button for desired level of detail

4. To Cage
* Go to "Proxy Mesh" Group, click "Generate" Button
* In the dialog, select interleaved for the interleaved decimation, or standard for standard CQEM
* wait for decimation process to complete

5. Save Result
* click "File" Button on the Menu Bar
* click "Save Proxy" to save ply

## Citation

```
@ARTICLE{11494421,
  author={Huang, Jiajun and Xu, Shuolin and Yu, Hongchuan and Lee, Tong-Yee},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting}, 
  year={2026},
  volume={},
  number={},
  pages={1-16},
  keywords={Videos;Video equipment;Radio access networks;Regional area networks;Graphical user interfaces;MISO;Protocols;Neural radiance field;Deep learning;Artificial intelligence;3D Gaussian Splatting;Cage-based Deformation;3D Representation Editing;Animation},
  doi={10.1109/TVCG.2026.3687062}}
```
