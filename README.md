# KinectFusion

![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/mremilien/KinectFusion.svg?style=social)](https://github.com/mremilien/KinectFusion/stargazers)

This is an implementation of KinectFusion by python, based on *Newcombe, Richard A., et al.* **KinectFusion: Real-time dense surface mapping and tracking.** This code is just for learning the processing of the KinectFusion, so I do not implement it real time. In the original paper, the author use GPU heavily for accelerating, for python you can use numba model to do this. And more details can be found in my [report](https://github.com/mremilien/KinectFusion/blob/main/docs/report.pdf).

## Dependencies
I implement it on python 3.6.And the main module I used is Open3d (version 0.12.0). And other modules are:
* numpy==1.19.2
* opencv==3.4.2
* trimesh==3.9.1
* coloredlogs==15.0
* scipy==1.5.4
* pyrender==0.1.39

## File structure
* `data/` the data for testing our algorithm, [TUM rgbd dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download).
* `docs/` the report of my experiment and the reference papers.
* `output/` the runing results will be saved in this folder.
* `src/` includes all of the source code.
   * `data_loader/` the code for getting the input rgbd images.
   * `kinect_fusion/` implementation for kinect fusion.
   * `utils/` some tool function for showing mesh.
   * `extension/` The novelty what I did for KinectFusion.
   * `main.py` main script to run experiment
   * `showmesh.py` to show 3d object file (.obj /.pcd/.off)

## How to run
For testing the performance, I use [TUM rgbd dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download) (please download dataset firstly). However, The Kinect provides the color and depth images in an un-synchronized way. This means that the set of time stamps from the color images do not intersect with those of the depth images. Therefore, we need some way of associating color images to depth images. For this purpose, you can use `associate.py` script which has been provided by TUM group. And details show [here](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools).

Directly run ```python main.py```, you will get the results in `ouput/`. And results are saved as `.obj` and `.pcd` formats. Besides, the configuration for the experiment will be written into `configuration.json` file.

Then, for showing the results, you can run ```python showmesh.py file_name``` to obtain the visualization reults of `.obj` or `.pcd` file.

Appart from this, ```cd extension``` and then do ```python fusion_by_point_cloud.py```, you can see the fusion result by point cloud method which extends by myself. And more details about this part is shown in my report.

### Easy usage
``` python
import matplotlib.pyplot as plt
from os.path import join as opj
import os
import json
import open3d as o3d

from data_loader import *
from kinect_fusion import *
from utils.pyrender_show import showMesh
from utils.open3d_show import showPointCloud

camera = Camera('../data/camera_config.txt')
dataset = Dataset('../data/rgbd_dataset_freiburg2_xyz/')
dataloader = DataLoader(dataset=dataset)
    
kinect = KinectFusion()

for (i, img) in enumerate(dataloader):
  depth_img = img['depth']
  rgb_img = img['rgb']
  frame = Frame(depth_img, rgb_img, camera.parameter)
  
  kinect.run(frame)
  
  if i >= 10:
    mesh = kinect.getMesh()
    showMesh(mesh)

```


