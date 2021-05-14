import numpy as np


def TsdfFusion(GlobalV_tsdf_weight, GlobalV_tsdf_value,
               GlobalV, GlobalN, GlobalV_color, Vk, Nk,
               depth_img, rgb_img, camera_intrinsic_matrix,
               Tkg, truncation_distance=1):
    img = depth_img
    Tgk = np.linalg.inv(Tkg)

    cam_coord_GlobalV = GlobalV @ Tgk[:3, :3].T + Tgk[:3, 3].reshape(1, 3)  # n * 3
    img_coord_GlobalV = cam_coord_GlobalV @ camera_intrinsic_matrix.T  # n * 3
    img_coord_GlobalV = img_coord_GlobalV / img_coord_GlobalV[:, 2].reshape(-1, 1)
    uv_coord_GlobalV = np.round(img_coord_GlobalV[:, :2]).astype(int)

    cam_coord_Vk = Vk
    cam_coord_Nk = Nk
    img_coord_Vk = cam_coord_Vk @ camera_intrinsic_matrix.T  # n * 3
    img_coord_Vk = img_coord_Vk / img_coord_Vk[:, 2].reshape(-1, 1)
    uv_coord_Vk = np.round(img_coord_Vk[:, :2]).astype(int)

    new_GlobalV = GlobalV.copy()
    new_GlobalN = GlobalN.copy()
    new_GlobalV_tsdf_value = GlobalV_tsdf_value.copy()
    new_GlobalV_tsdf_weight = GlobalV_tsdf_weight.copy()
    new_GlobalV_color = GlobalV_color.copy()

    delete_repeat_ids = np.array([])

    h, w = img.shape
    for v in range(img.shape[0] - 1):
        for u in range(img.shape[1] - 1):
            if img[v, u] <= 1e-5:
                continue

            # find the vertices which project on the pixel (u, v)
            u_ids = np.where(uv_coord_GlobalV[:, 0] == u)
            v_ids = np.where(uv_coord_GlobalV[:, 1] == v)
            GlobalV_uv_ids = np.intersect1d(u_ids[0], v_ids[0])
            u_ids = np.where(uv_coord_Vk[:, 0] == u)
            v_ids = np.where(uv_coord_Vk[:, 1] == v)
            Vk_uv_ids = np.intersect1d(u_ids[0], v_ids[0])

            # compute tsdf
            raw_depth = img[v, u]
            Lambda = np.array([u, v, 1]) @ np.linalg.inv(camera_intrinsic_matrix.T)
            Lambda = np.linalg.norm(Lambda.reshape(-1))
            TSDFs = {}
            TSDFs['GlobalV'] = []
            GlobalV_repeat_ids = []
            TSDFs['Vk'] = []
            Vk_repeat_ids = []

            for id in GlobalV_uv_ids:
                vertex = cam_coord_GlobalV[id, :]
                sdf = 1 / Lambda * np.linalg.norm(vertex) - raw_depth
                if -truncation_distance < sdf < truncation_distance:
                    TSDFs['GlobalV'].append(sdf)
                    # only the value of sdf is between truncation distance, the GlobalV will be the same vertex
                    GlobalV_repeat_ids.append(id)
                    delete_repeat_ids = np.union1d(delete_repeat_ids, np.array(GlobalV_repeat_ids))

            for id in Vk_uv_ids:
                vertex = cam_coord_Vk[id, :]
                sdf = 1 / Lambda * np.linalg.norm(vertex) - raw_depth
                if -truncation_distance < sdf < truncation_distance:
                    Vk_repeat_ids.append(id)

            if len(GlobalV_repeat_ids) + len(Vk_repeat_ids) == 0:
                continue
            else:
                # # delete the repeat vertices
                # new_GlobalV[GlobalV_repeat_ids, :] = [0, 0, 0]
                # new_GlobalN[GlobalV_repeat_ids, :] = [0, 0, 0]

                # fusion
                if len(GlobalV_repeat_ids) == 0:
                    # then add vk (unoverlap part of frame k-1 and frame k)
                    new_vertex_list = []  # for mean
                    new_normal_list = []
                    for id in Vk_repeat_ids:
                        new_cam_vertex = cam_coord_Vk[id, :]
                        new_cam_normal = cam_coord_Nk[id, :]
                        new_glo_vertex = (new_cam_vertex @ Tkg[:3, :3].T + Tkg[:3, 3].reshape(1, 3)).reshape(-1)
                        new_glo_normal = (new_cam_normal @ Tkg[:3, :3].T).reshape(-1)
                        new_vertex_list.append(new_glo_vertex)
                        new_normal_list.append(new_glo_normal)
                    new_vertex = np.mean(np.array(new_vertex_list), axis=0)
                    new_normal = np.mean(np.array(new_normal_list), axis=0)
                    new_color = rgb_img[v, u]
                    new_GlobalV = np.vstack((new_GlobalV, new_vertex.reshape(1, -1)))
                    new_GlobalN = np.vstack((new_GlobalN, new_normal.reshape(1, -1)))
                    new_GlobalV_tsdf_value = np.vstack((new_GlobalV_tsdf_value, [0]))
                    new_GlobalV_tsdf_weight = np.vstack((new_GlobalV_tsdf_weight, len(Vk_repeat_ids) * np.array([1])))
                    new_GlobalV_color = np.vstack((new_GlobalV_color, new_color.reshape(1, -1)))
                else:
                    # then fusion global v
                    new_vertex = np.zeros(3)  # for mean
                    new_normal = np.zeros(3)
                    new_tsdf = 0
                    new_weight = 0
                    new_color = np.zeros(3)
                    for id in GlobalV_repeat_ids:
                        color = GlobalV_color[id, :]
                        tsdf = GlobalV_tsdf_value[id, :]
                        weight = GlobalV_tsdf_weight[id, :]
                        glo_normal = GlobalN[id, :]
                        glo_vertex = GlobalV[id, :]
                        new_vertex += weight * glo_vertex
                        new_normal += weight * glo_normal
                        new_color += weight * color
                        new_tsdf += weight * tsdf
                        new_weight += weight
                    # add the point of (u, v)
                    depth = img[v, u]
                    if depth == 0:
                        new_tsdf /= new_weight
                        new_normal /= new_weight
                        new_normal = new_normal / np.linalg.norm(new_normal)
                        new_vertex /= new_weight
                        new_color /= new_weight
                        new_GlobalV = np.vstack((new_GlobalV, new_vertex.reshape(1, -1)))
                        new_GlobalN = np.vstack((new_GlobalN, new_normal.reshape(1, -1)))
                        new_GlobalV_tsdf_value = np.vstack((new_GlobalV_tsdf_value, [new_tsdf]))
                        new_GlobalV_tsdf_weight = np.vstack((new_GlobalV_tsdf_value, [new_weight]))
                        new_GlobalV_color = np.vstack((new_GlobalV_color, new_color.reshape(1, -1)))
                    else:
                        local_vertex = DepthImgVertex2CameraCoordVertex(u, v, depth, camera_intrinsic_matrix)
                        global_vertex = (local_vertex @ Tkg[:3, :3].T + Tkg[:3, 3].reshape(1, 3)).reshape(-1)
                        local_normal = GetNormalOfUv(u, v, depth_img, camera_intrinsic_matrix)
                        global_normal = (local_normal @ Tkg[:3, :3].T).reshape(-1)
                        local_color = rgb_img[v, u]
                        if np.linalg.norm(local_normal) == 0:
                            new_tsdf /= new_weight
                            new_normal /= new_weight
                            new_normal = new_normal / np.linalg.norm(new_normal)
                            new_vertex /= new_weight
                            new_color /= new_weight
                            new_GlobalV = np.vstack((new_GlobalV, new_vertex.reshape(1, -1)))
                            new_GlobalN = np.vstack((new_GlobalN, new_normal.reshape(1, -1)))
                            new_GlobalV_tsdf_value = np.vstack((new_GlobalV_tsdf_value, [new_tsdf]))
                            new_GlobalV_tsdf_weight = np.vstack((new_GlobalV_tsdf_weight, [new_weight]))
                            new_GlobalV_color = np.vstack((new_GlobalV_color, new_color.reshape(1, -1)))
                        else:
                            new_weight += 1
                            new_vertex += global_vertex
                            new_normal += global_normal
                            new_color += local_color
                            new_tsdf /= new_weight
                            new_normal /= new_weight
                            new_normal = new_normal / np.linalg.norm(new_normal)
                            new_vertex /= new_weight
                            new_color /= new_weight
                            new_GlobalV = np.vstack((new_GlobalV, new_vertex.reshape(1, -1)))
                            new_GlobalN = np.vstack((new_GlobalN, new_normal.reshape(1, -1)))
                            new_GlobalV_tsdf_value = np.vstack((new_GlobalV_tsdf_value, [new_tsdf]))
                            new_GlobalV_tsdf_weight = np.vstack((new_GlobalV_tsdf_weight, [new_weight]))
                            new_GlobalV_color = np.vstack((new_GlobalV_color, new_color.reshape(1, -1)))

    new_GlobalV = np.delete(new_GlobalV, delete_repeat_ids.astype(int), axis=0)
    new_GlobalN = np.delete(new_GlobalN, delete_repeat_ids.astype(int), axis=0)
    new_GlobalV_tsdf_value = np.delete(new_GlobalV_tsdf_value, delete_repeat_ids.astype(int), axis=0)
    new_GlobalV_tsdf_weight = np.delete(new_GlobalV_tsdf_weight, delete_repeat_ids.astype(int), axis=0)
    new_GlobalV_color = np.delete(new_GlobalV_color, delete_repeat_ids.astype(int), axis=0)
    return new_GlobalV, new_GlobalN, new_GlobalV_color, new_GlobalV_tsdf_value, new_GlobalV_tsdf_weight


def DepthImgVertex2CameraCoordVertex(u, v, depth, camera_intrinsic_matrix):
    camera_fx = camera_intrinsic_matrix[0, 0]
    camera_fy = camera_intrinsic_matrix[1, 1]
    camera_cx = camera_intrinsic_matrix[0, 2]
    camera_cy = camera_intrinsic_matrix[1, 2]

    z = depth
    x = (u - camera_cx) * z / camera_fx
    y = (v - camera_cy) * z / camera_fy
    local_new_vertex = np.array([x, y, z])
    return local_new_vertex

def GetNormalOfUv(u, v, depth_img, camera_intrinsic_matrix):
    pt_uv = DepthImgVertex2CameraCoordVertex(u, v, depth_img[v, u], camera_intrinsic_matrix)
    pt_u1v = DepthImgVertex2CameraCoordVertex(u + 1, v, depth_img[v, u + 1], camera_intrinsic_matrix)
    pt_uv1 = DepthImgVertex2CameraCoordVertex(u, v + 1, depth_img[v + 1, u], camera_intrinsic_matrix)
    n = np.cross(pt_uv1 - pt_uv, pt_u1v - pt_uv)
    return n








