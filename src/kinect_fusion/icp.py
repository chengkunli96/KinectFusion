import numpy as np
import open3d as o3d


def ICP(v1, n1, v2, n2, max_iter=5, sample_rate=1, trans_init=np.eye(4), threshold=0.1):
    """Transformation matrix from v2 to v1"""
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(v1.reshape(-1, 3))
    source.normals = o3d.utility.Vector3dVector(n1.reshape(-1, 3))

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(v2.reshape(-1, 3))
    target.normals = o3d.utility.Vector3dVector(n2.reshape(-1, 3))

    # down sample
    source_vertex_num = np.array(source.points).shape[0]
    if int(source_vertex_num * sample_rate) < 100:
        sample_num = np.minimum(100, source_vertex_num)
        source_sample_rate = sample_num / source_vertex_num
    else:
        source_sample_rate = sample_rate
    source_samples = source.voxel_down_sample(voxel_size=source_sample_rate)

    target_vertex_num = np.array(target.points).shape[0]
    if int(source_vertex_num * sample_rate) < 100:
        sample_num = np.minimum(100, target_vertex_num)
        target_sample_rate = sample_num / source_vertex_num
    else:
        target_sample_rate = sample_rate
    target_samples = target.voxel_down_sample(voxel_size=target_sample_rate)

    # # outlier
    # processed_source, outlier_index = source_samples.remove_radius_outlier(nb_points=16, radius=0.5)
    #     # processed_target, outlier_index = target_samples.remove_radius_outlier(nb_points=16, radius=0.5)
    #     # processed_source_vertex_num = np.array(processed_source.points).shape[0]
    #     # print(source_vertex_num)
    #     # if processed_source_vertex_num < 100:
    #     #     processed_source = source_samples
    #     # processed_target_vertex_num = np.array(processed_target.points).shape[0]
    #     # if processed_target_vertex_num < 100:
    #     #     processed_target = target_samples

    #  icp
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_samples, target_samples, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

    return reg_p2p.transformation

