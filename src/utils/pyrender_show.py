import numpy as np
import argparse
import open3d as o3d
import matplotlib as mpl
import matplotlib.cm as cm
import trimesh
import pyrender


def scene_factory(render_list, return_nodes=False):
    scene = pyrender.Scene(ambient_light=0.5 * np.array([1.0, 1.0, 1.0, 1.0]))

    nd_list = []
    for m in render_list:
        nd = scene.add(m)
        nd_list.append(nd)

    if return_nodes:
        return scene, nd_list
    else:
        return scene


def show_mesh_gui(rdobj):
    scene = scene_factory([rdobj])
    v = pyrender.Viewer(scene, use_raymond_lighting=True)
    del v


def showMesh(mesh: trimesh):
    mesh_rd = pyrender.Mesh.from_trimesh(mesh)
    show_mesh_gui(mesh_rd)