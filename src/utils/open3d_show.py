import numpy as np
import argparse
import open3d as o3d
import matplotlib as mpl
import matplotlib.cm as cm
import trimesh
import pyrender


def toO3d(tm, color):
    """put trimesh object into open3d object"""
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(tm.faces)
    mesh_o3d.compute_vertex_normals()
    vex_color_rgb = np.array(color)
    if vex_color_rgb.ndim < 2:
        vex_color_rgb = np.tile(vex_color_rgb, (tm.vertices.shape[0], 1))
    mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vex_color_rgb)
    return mesh_o3d


def showMesh(tm: trimesh, color=None):
    if color is None:
        color = np.array(tm.face_normals)
    # show the object by open3d
    mesh_o3d = toO3d(tm, color=color)
    o3d.visualization.draw_geometries([mesh_o3d])


def showPointCloud(vertices, colors):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])


