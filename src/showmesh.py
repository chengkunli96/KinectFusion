from utils.pyrender_show import showMesh
from utils.open3d_show import showPointCloud

import numpy as np
import open3d as o3d
import trimesh
import argparse
import os


if __name__ == '__main__':
    # arg
    parser = argparse.ArgumentParser(description='for showing the experiment result.')
    parser.add_argument('data', type=str, help='data file path.')
    args = parser.parse_args()

    assert os.path.exists(args.data), 'cannot found:' + args.data

    # the file format
    suffix = os.path.splitext(args.data)[-1]
    if suffix == '.obj':
        mesh = trimesh.load(args.data)
        print(mesh.visual.vertex_colors[mesh.visual.vertex_colors[:, 3] == 0, :])
        showMesh(mesh)
    elif suffix == '.pcd':
        pcd = o3d.io.read_point_cloud(args.data)
        showPointCloud(vertices=pcd.points, colors=pcd.colors)
    elif suffix == '.off':
        mesh = o3d.io.read_triangle_mesh(args.data)
        print(np.array(mesh.vertices))
        print(np.array(mesh.vertex_colors))
        colors = vex_color_rgb = np.tile(np.array([0.5, 0, 0]), (np.array(mesh.vertices).shape[0], 1))
        showPointCloud(vertices=mesh.vertices, colors=colors)
    else:
        raise ValueError('the file should be ".obj" or ".pcd"')

