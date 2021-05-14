import numpy as np
import cv2
import logging
import coloredlogs
from skimage import measure
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d
from numba import njit, prange
from numba.experimental import jitclass
from numba import int32, float32
import matplotlib.cm as cm

from data_loader import *
from .frame import Frame
from .icp import ICP
from .volume import Volume

# logger
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('OpenGL.acceleratesupport').disabled = True
logging.getLogger('OpenGL.GL.shaders').disabled = True
logger = logging.getLogger()
coloredlogs.install(
    datefmt='%d/%m %H:%M',
    fmt='%(asctime)s %(levelname)s %(message)s',
    level='INFO',
)


class KinectFusion(object):
    def __init__(self):
        # the frame which has been processed
        self.frame_id = -1
        self.frame = None
        self.frame_pyramid = {}
        # transform matrix (local->global, k2g means local frame k to global frame).
        self.T_k2g = np.eye(4)
        # icp config
        self.sample_rate = 0.02  # TODO: (initially, this part can be writen into icp function)
        self.pose_estimation_mode = 'frame2frame'
        self.icp_threshold = 1
        # volume config
        self.volume = Volume(scale=(5, 5, 5), shape=(256, 256, 256), origin=(0, 0, 0))
        # tsdf config
        self.truncation_distance = 0.1
        # for computing mesh color
        self.mesh_color_method = 'volume'
        self.frame_list = []  # save every frame for computing colors
        self.Transformation_loc2glo_list = []  # for saving every transform matrix
        self.pose_estimation_method = 'icp'

    def configPosEstimationMethod(self, method):
        if method == 'icp' or method == 'Park':
            self.pose_estimation_method = method
        else:
            raise ValueError('Parameter "pose_estimation_mode" should only be "icp" or "Park"')

    def configIcp(self, sample_rate=0.01, threshold=1, pose_estimation_mode='frame2frame'):
        self.sample_rate = sample_rate
        self.icp_threshold = threshold
        if pose_estimation_mode == 'frame2frame' or pose_estimation_mode == 'frame2model':
            self.pose_estimation_mode = pose_estimation_mode
        else:
            raise ValueError('Parameter "pose_estimation_mode" should only be "frame2frame" or "frame2model"')

    def configVolume(self, scale, shape, origin):
        self.volume = Volume(scale=scale, shape=shape, origin=origin)

    def configTsdf(self, truncation_distance):
        self.truncation_distance = truncation_distance

    def configMeshColorComputingMethod(self, mode='by volume'):
        if mode == 'by volume':
            self.mesh_color_method = 'volume'
        elif mode == 'by frame':
            self.mesh_color_method = 'frame'
        else:
            raise ValueError('Parameter "mode" should only be "by volume" or "by frame".')

    def run(self, frame):
        if self.frame_id == -1:
            self.initialize(frame)
        else:
            self.process(frame)

    def initialize(self, frame: Frame):
        """frame = Frame(depth_img, rgb_img, camera_parameter)"""
        logger.info('Initializing with Frame 0')
        # some info of current frame
        raw_depth_img = frame.getKinectUnscaledDepthImg()
        rgb_img = frame.getRbgImg()
        camera_parameter = frame.getCameraParameters()
        # the frame has been processed
        self.frame_id = 0

        # =====Mesurement=====
        logger.info('Measurement')
        # frame pyramid
        h, w = raw_depth_img.shape
        level1_depth = raw_depth_img
        level1_rgb = rgb_img
        level2_depth = cv2.resize(raw_depth_img, (int(w / 2), int(h / 2)))
        level2_rgb = cv2.resize(rgb_img, (int(w / 2), int(h / 2)))
        level3_depth = cv2.resize(raw_depth_img, (int(w / 4), int(h / 4)))
        level3_rgb = cv2.resize(rgb_img, (int(w / 4), int(h / 4)))
        curr_frame_pyramid = {'l1': Frame(level1_depth, level1_rgb, camera_parameter),
                              'l2': Frame(level2_depth, level2_rgb, camera_parameter),
                              'l3': Frame(level3_depth, level3_rgb, camera_parameter)}
        curr_frame = curr_frame_pyramid['l1']

        # =====Pose Estimation=====
        logger.info('Pose Estimation')
        # set the first frame's camera position is the original of the global frame
        T_curr2glo = np.eye(4)

        # =====Surface Reconstruction=====
        logger.info('Surface Reconstruction')
        pre_volume = self.volume
        curr_volume = self.surfaceReconstrut(pre_volume, curr_frame, T_curr2glo, self.truncation_distance)

        # =====Update=====
        self.T_k2g = T_curr2glo
        self.frame_pyramid = curr_frame_pyramid
        self.frame = curr_frame
        self.volume = curr_volume
        self.Transformation_loc2glo_list.append(self.T_k2g)
        self.frame_list.append(self.frame)
        pass

    def process(self, frame: Frame):
        # some info of frame
        raw_depth_img = frame.getKinectUnscaledDepthImg()
        rgb_img = frame.getRbgImg()
        camera_parameter = frame.getCameraParameters()

        # update frame id
        self.frame_id += 1
        logger.info('Processing Frame {:d}'.format(self.frame_id))

        # =====Measurement, compute surface vertices and normal maps=====
        logger.info('Measurement')
        # the previous frame pyramid
        pre_frame_pyramid = self.frame_pyramid
        # get current frame pyramid
        h, w = raw_depth_img.shape
        level1_depth = raw_depth_img
        level1_rgb = rgb_img
        level2_depth = cv2.resize(raw_depth_img, (int(w / 2), int(h / 2)))
        level2_rgb = cv2.resize(rgb_img, (int(w / 2), int(h / 2)))
        level3_depth = cv2.resize(raw_depth_img, (int(w / 4), int(h / 4)))
        level3_rgb = cv2.resize(rgb_img, (int(w / 4), int(h / 4)))
        # get the local vertices and normals
        # you can run the method of class fame to get the vertices and normals
        # like frame.getLocalVertices() or frame.getLocalNormals
        curr_frame_pyramid = {'l1': Frame(level1_depth, level1_rgb, camera_parameter),
                              'l2': Frame(level2_depth, level2_rgb, camera_parameter),
                              'l3': Frame(level3_depth, level3_rgb, camera_parameter)}
        curr_frame = curr_frame_pyramid['l1']

        # =====Pose Estimation=====
        logger.info('Pose Estimation')
        T_pre2glo = self.T_k2g
        if self.pose_estimation_mode == 'frame2model':
            pre_frame_pyramid['l1'] = self._updateFrameByVolume(self.volume, pre_frame_pyramid['l1'], T_pre2glo,
                                                                truncation_distance=self.truncation_distance)
        # get the transformation matrix which corresponds current frame to previous frame
        if self.pose_estimation_method == 'icp':
            T_curr2pre = self.poseEstimate1(pre_frame_pyramid, curr_frame_pyramid, self.sample_rate)
        else:
            T_curr2pre = self.poseEstimate2(pre_frame_pyramid, curr_frame_pyramid)
        # get the transformation matrix (curr -> global)
        T_curr2glo = T_pre2glo @ T_curr2pre

        # =====Surface Reconstruction=====
        logger.info('Surface Reconstruction')
        pre_volum = self.volume
        curr_volum = self.surfaceReconstrut(pre_volum, curr_frame, T_curr2glo, self.truncation_distance)

        # =====Update=====
        self.T_k2g = T_curr2glo
        self.frame_pyramid = curr_frame_pyramid
        self.frame = curr_frame
        self.volume = curr_volum
        self.Transformation_loc2glo_list.append(self.T_k2g)
        self.frame_list.append(self.frame)

    def poseEstimate2(self, pre_frame_pyramid, curr_frame_pyramid):
        """Colored Point Cloud Registration. Park's method"""
        pre_frame = pre_frame_pyramid['l1']
        depth_o3d_img = o3d.geometry.Image((pre_frame.raw_depth_img).astype(np.float32))
        color_o3d_img = o3d.geometry.Image((pre_frame.rgb_img).astype(np.float32))
        pre_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_o3d_img, depth_o3d_img)

        curr_frame = curr_frame_pyramid['l1']
        depth_o3d_img = o3d.geometry.Image((curr_frame.raw_depth_img).astype(np.float32))
        color_o3d_img = o3d.geometry.Image((curr_frame.rgb_img).astype(np.float32))
        curr_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(color_o3d_img, depth_o3d_img)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=curr_frame.raw_depth_img.shape[1],
            height=curr_frame.raw_depth_img.shape[0],
            fx=curr_frame.parameter['fx'],
            fy=curr_frame.parameter['fx'],
            cx=curr_frame.parameter['cx'],
            cy=curr_frame.parameter['cy'],
        )

        odo_init = np.identity(4)
        option = o3d.pipelines.odometry.OdometryOption()

        [success_hybrid_term, trans_hybrid_term,
         info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            curr_rgbd_image, pre_rgbd_image, pinhole_camera_intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

        transform = np.array(trans_hybrid_term)
        return transform

    def poseEstimate1(self, pre_frame_pyramid, curr_frame_pyramid, sample_rate=0.1):
        """Follow the instruction of the paper, ICP method"""
        V3 = curr_frame_pyramid['l3'].getLocalVertices()
        N3 = curr_frame_pyramid['l3'].getLocalNormals()
        V2 = curr_frame_pyramid['l2'].getLocalVertices()
        N2 = curr_frame_pyramid['l2'].getLocalNormals()
        V1 = curr_frame_pyramid['l1'].getLocalVertices()
        N1 = curr_frame_pyramid['l1'].getLocalNormals()

        pre_V3 = pre_frame_pyramid['l3'].getLocalVertices()
        pre_N3 = pre_frame_pyramid['l3'].getLocalNormals()
        pre_V2 = pre_frame_pyramid['l2'].getLocalVertices()
        pre_N2 = pre_frame_pyramid['l2'].getLocalNormals()
        pre_V1 = pre_frame_pyramid['l1'].getLocalVertices()
        pre_N1 = pre_frame_pyramid['l1'].getLocalNormals()

        # ICP in level3
        T3 = ICP(pre_V3, pre_N3, V3, N3, max_iter=5, sample_rate=sample_rate,
                 threshold=self.icp_threshold)
        T2 = ICP(pre_V2, pre_N2, V2, N2, max_iter=10, sample_rate=sample_rate,
                 trans_init=T3, threshold=self.icp_threshold)  # 5
        T1 = ICP(pre_V1, pre_N1, V1, N1, max_iter=50, sample_rate=sample_rate,
                 trans_init=T2, threshold=self.icp_threshold)  # 10

        return T1

    def _updateFrameByVolume(self, volume: Volume, frame: Frame, T_cam2glo, truncation_distance):
        # marching cubes
        tsdf_volume = volume.tsdf_mt
        tsdf_volume[tsdf_volume == None] = 1
        verts, faces, normals, values = measure.marching_cubes(tsdf_volume, 0)
        vertices = volume.volumeCoord2globalCoordArray(verts)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        vertices = np.array(mesh.vertices)
        nomals = np.array(mesh.vertex_normals)
        camera_intrinsic_matrix = frame.parameter['K']
        T_glo2cam = np.linalg.inv(T_cam2glo)
        frame_coord_vertices = []
        frame_coord_normals = []
        for k in range(vertices.shape[0]):
            glo_coord_vertex = vertices[k, :]
            glo_coord_normal = nomals[k, :]
            cam_coord_vertex = (glo_coord_vertex @ T_glo2cam[:3, :3].T + T_glo2cam[:3, 3].reshape(1, 3)).reshape(-1)
            cam_coord_normal = (glo_coord_normal @ T_glo2cam[:3, :3].T).reshape(-1)
            if cam_coord_vertex[2] <= 1e-5:
                # the situation can not covert into uv_coord
                continue
            uv_coord_voxel = self._cameraCoord2uvCoord(cam_coord_vertex, camera_intrinsic_matrix)
            if not frame.contain(uv_coord_voxel):
                continue
            # compute tsdf
            u, v = uv_coord_voxel
            depth_img = frame.getDepthImg()
            Rk = depth_img[v, u]  # the raw depth value of Kinect image
            Lambda = self._getLambda(u, v, camera_intrinsic_matrix)
            sdf = self._getSdf(Lambda, Rk, cam_coord_vertex)
            # update volume color
            if -truncation_distance < sdf < truncation_distance:
                frame_coord_vertices.append(cam_coord_vertex)
                frame_coord_normals.append(cam_coord_normal)
        frame_coord_vertices = np.array(frame_coord_vertices).reshape(-1, 3)
        frame_coord_normals = np.array(frame_coord_normals).reshape(-1, 3)

        new_frame = frame
        new_frame.local_vertices = frame_coord_vertices
        new_frame.local_normals = frame_coord_normals
        return new_frame

    def _downSampleVerticesAndNormals(self, vertices, normals, sample_rate=1.0):
        vertex_num = vertices.shape[0]
        if int(vertex_num * sample_rate) < 100:
            sample_num = np.minimum(100, vertex_num)
        else:
            sample_num = int(vertex_num * sample_rate)
        mask = np.random.randint(0, vertex_num, sample_num)
        vertex_samples = vertices[mask, :]
        normal_samples = normals[mask, :]
        return vertex_samples, normal_samples

    def _downSample(self, vertices, sample_rate=1.0):
        vertex_num = vertices.shape[0]
        if int(vertex_num * sample_rate) < 100:
            sample_num = np.min(100, vertex_num)
        else:
            sample_num = int(vertex_num * sample_rate)
        mask = np.random.randint(0, vertex_num, sample_num)
        vertex_samples = vertices[mask, :]
        return vertex_samples

    def surfaceReconstrut(self, volume: Volume, curr_frame: Frame, T_curr2glo, truncation_distance):
        T_glo2curr = np.linalg.inv(T_curr2glo)
        camera_parameter = curr_frame.getCameraParameters()
        camera_intrinsic_matrix = camera_parameter['K']
        # TODO：this part maybe can be convert into numpy computation
        count = 0
        for i in prange(volume.shape[0]):
            for j in prange(volume.shape[1]):
                for k in prange(volume.shape[2]):
                    count += 1
                    if count % (volume.voxel_num // 100) == 0 or count == volume.voxel_num:
                        print('\rComputing TSDF {:.2f}% ...'.format(100 * count / volume.voxel_num), end='', flush=True)
                    glo_coord_voxel = volume.volumeCoord2globalCoord(i, j, k)
                    cam_coord_voxel = (glo_coord_voxel @ T_glo2curr[:3, :3].T + T_glo2curr[:3, 3].reshape(1, 3)).reshape(-1)
                    if cam_coord_voxel[2] <= 1e-5:
                        # the situation can not covert into uv_coord
                        continue
                    uv_coord_voxel = self._cameraCoord2uvCoord(cam_coord_voxel, camera_intrinsic_matrix)

                    # whether this vertex (u, v) has a depth value,
                    # which determines whether the tsdf could be compute
                    if not curr_frame.contain(uv_coord_voxel):
                        continue

                    # compute tsdf
                    u, v = uv_coord_voxel
                    depth_img = curr_frame.getDepthImg()
                    Rk = depth_img[v, u]  # the raw depth value of Kinect image
                    Lambda = self._getLambda(u, v, camera_intrinsic_matrix)
                    sdf = self._getSdf(Lambda, Rk, cam_coord_voxel)
                    tsdf = self._sdf2tsdf(sdf, truncation_distance)

                    # do not update tsdf if point (u, v) is not near the surface defined by the depth img
                    if tsdf is None:
                        continue
                    # update fusion
                    curr_sdf = sdf
                    curr_tsdf = tsdf
                    curr_weight = 1
                    pre_tsdf, pre_weight = volume.getTsdf(i, j, k)
                    if pre_tsdf is None:
                        new_tsdf = curr_tsdf
                        new_weight = curr_weight
                    else:
                        new_tsdf = (pre_weight * pre_tsdf + curr_weight * curr_tsdf) / (pre_weight + curr_weight)
                        new_weight = pre_weight + curr_weight
                    volume.setTsdf(i, j, k, new_tsdf, new_weight)

                    # update volume color
                    rgb_img = curr_frame.getRbgImg()
                    curr_color = rgb_img[v, u, :]
                    pre_color = volume.getColor(i, j, k)
                    if -truncation_distance < curr_sdf < truncation_distance:
                        if pre_color[0] is None or pre_color[1] is None or pre_color[2] is None:
                            new_color = curr_color
                        else:
                            new_color = (pre_weight * pre_color + curr_weight * curr_color) / (pre_weight + curr_weight)
                        volume.setColor(i, j, k, new_color)
        print('')
        return volume

    def _cameraCoord2uvCoord(self, cam_coord_pos, camera_intrinsic_matrix):
        img_coord_pos = cam_coord_pos @ camera_intrinsic_matrix.T
        img_coord_pos = img_coord_pos.reshape(-1)
        img_coord_pos = img_coord_pos / img_coord_pos[2]
        uv_coord_pos = np.around(img_coord_pos[:2]).astype(int)
        return uv_coord_pos

    def _getLambda(self, u, v, camera_intrinsic_matrix):
        Lambda = np.array([u, v, 1]) @ np.linalg.inv(camera_intrinsic_matrix.T)
        Lambda = np.linalg.norm(Lambda.reshape(-1))
        return Lambda

    def _getSdf(self, Lambda, raw_depth, cam_coord_pos):
        sdf = 1 / Lambda * np.linalg.norm(cam_coord_pos) - raw_depth
        sdf = -1 * sdf
        return sdf

    def _sdf2tsdf(self, sdf, truncation_distance):
        if sdf > -truncation_distance:
            tsdf = np.minimum(1, sdf / truncation_distance)
            return tsdf
        else:
            return -1

    def getMesh(self):
        mesh = self._marchingCubes(self.volume)
        return mesh

    def _marchingCubes(self, volume: Volume):
        tsdf_volume = volume.tsdf_mt
        tsdf_volume[tsdf_volume == None] = 1
        verts, faces, normals, values = measure.marching_cubes(tsdf_volume, 0)
        # convert from volume coord to global coord
        vertices = volume.volumeCoord2globalCoordArray(verts)
        if self.mesh_color_method == 'frame':
            colors = self._getColorsByFrames(vertices)
        else:
            colors = self._getColorsByVolumeCube(volume, verts)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        mesh.visual.vertex_colors = colors
        return mesh

    def getPointClould(self, sample_rate=0.2):
        """:return pcd open3d.geometry.PointCloud()"""
        vertices = []
        normals = []
        for i in prange(self.frame_id + 1):
            frame = self.frame_list[i]
            Trans = self.Transformation_loc2glo_list[i]
            local_vertices = frame.getLocalVertices()
            local_normals = frame.getLocalNormals()
            local_vertex_samples, local_normal_samples = self._downSampleVerticesAndNormals(
                local_vertices, local_normals, sample_rate=sample_rate)
            global_vertex_samples = local_vertex_samples @ Trans[:3, :3].T + Trans[:3, 3].reshape(1, 3)
            global_normal_samples = local_normal_samples @ Trans[:3, :3].T
            if i == 0:
                vertices = global_vertex_samples
                normals = global_normal_samples
            else:
                vertices = np.vstack((vertices, global_vertex_samples))
                normals = np.vstack((normals, global_normal_samples))
        colors = self._getColorsByFrames(vertices)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        return pcd

    def _getColorsByVolumeCube(self, volume: Volume, voxels):
        voxels = np.array(voxels)
        colors = np.zeros((voxels.shape[0], 4))
        for i in prange(voxels.shape[0]):
            voxel = voxels[i, :]
            # voxel_cube = None
            m, n, k = np.floor(voxel).astype(int)
            # on the point
            if voxel[0] - m == 0 and voxel[1] - n == 0 and voxel[2] - k == 0:
                voxel_cube = np.zeros((1, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
            # in the cube
            elif voxel[0] - m == 0 and voxel[1] - n == 0 and voxel[2] - k == 0:
                voxel_cube = np.zeros((8, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
                voxel_cube[1, :] = [m + 1, n, k]
                voxel_cube[2, :] = [m, n + 1, k]
                voxel_cube[3, :] = [m, n, k + 1]
                voxel_cube[4, :] = [m + 1, n + 1, k]
                voxel_cube[5, :] = [m + 1, n, k + 1]
                voxel_cube[6, :] = [m, n + 1, k + 1]
                voxel_cube[7, :] = [m + 1, n + 1, k + 1]
            # in the face
            elif voxel[0] - m == 0 and voxel[1] - n != 0 and voxel[2] - k != 0:
                voxel_cube = np.zeros((4, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
                voxel_cube[1, :] = [m, n + 1, k]
                voxel_cube[2, :] = [m, n, k + 1]
                voxel_cube[3, :] = [m, n + 1, k + 1]
            elif voxel[0] - m != 0 and voxel[1] - n == 0 and voxel[2] - k != 0:
                voxel_cube = np.zeros((4, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
                voxel_cube[1, :] = [m + 1, n, k]
                voxel_cube[2, :] = [m, n, k + 1]
                voxel_cube[3, :] = [m + 1, n, k + 1]
            elif voxel[0] - m != 0 and voxel[1] - n != 0 and voxel[2] - k == 0:
                voxel_cube = np.zeros((4, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
                voxel_cube[1, :] = [m + 1, n, k]
                voxel_cube[2, :] = [m, n + 1, k]
                voxel_cube[3, :] = [m + 1, n + 1, k]
            # in the line
            elif voxel[0] - m == 0 and voxel[1] - n == 0 and voxel[2] - k != 0:
                voxel_cube = np.zeros((2, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
                voxel_cube[1, :] = [m, n, k + 1]
            elif voxel[0] - m == 0 and voxel[1] - n != 0 and voxel[2] - k == 0:
                voxel_cube = np.zeros((2, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
                voxel_cube[1, :] = [m, n + 1, k]
            elif voxel[0] - m != 0 and voxel[1] - n == 0 and voxel[2] - k == 0:
                voxel_cube = np.zeros((2, 3)).astype(int)
                voxel_cube[0, :] = [m, n, k]
                voxel_cube[1, :] = [m + 1, n, k]
            rgb = np.zeros(3)
            weight = 0
            for id in range(voxel_cube.shape[0]):
                cube_coner = voxel_cube[id, :]
                x, y, z = cube_coner
                distance = np.linalg.norm(voxel - cube_coner)
                cube_coner_color = volume.getColor(x, y, z)
                if distance == 0:
                    if cube_coner_color[0] is not None \
                            and cube_coner_color[1] is not None \
                            and cube_coner_color[2] is not None:
                        rgb = cube_coner_color
                    weight = 1
                    break
                if cube_coner_color[0] is not None \
                        and cube_coner_color[1] is not None \
                        and cube_coner_color[2] is not None:
                    rgb = rgb + 1 / distance * cube_coner_color
                    weight += 1 / distance
            if weight == 0:
                weight = 1
            rgb /= weight
            if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
                rgba = [1, 1, 1, 0]
            else:
                rgba = [rgb[0], rgb[1], rgb[2], 1]
            colors[i, :] = rgba
        return colors

    def _getColorsByFrames(self, vertices):
        truncation_distance = self.truncation_distance
        vertices_colors = np.ones((vertices.shape[0], 4))
        for k in prange(vertices.shape[0]):
            glo_coord_voxel = vertices[k, :]
            colors = []
            for i in prange(self.frame_id + 1):
                frame = self.frame_list[i]
                camera_intrinsic_matrix = frame.parameter['K']
                T_cam2glo = self.Transformation_loc2glo_list[i]
                T_glo2cam = np.linalg.inv(T_cam2glo)
                cam_coord_voxel = (glo_coord_voxel @ T_glo2cam[:3, :3].T + T_glo2cam[:3, 3].reshape(1, 3)).reshape(-1)
                if cam_coord_voxel[2] <= 1e-5:
                    # the situation can not covert into uv_coord
                    continue
                uv_coord_voxel = self._cameraCoord2uvCoord(cam_coord_voxel, camera_intrinsic_matrix)
                if not frame.contain(uv_coord_voxel):
                    continue
                # compute tsdf
                u, v = uv_coord_voxel
                depth_img = frame.getDepthImg()
                Rk = depth_img[v, u]  # the raw depth value of Kinect image
                Lambda = self._getLambda(u, v, camera_intrinsic_matrix)
                sdf = self._getSdf(Lambda, Rk, cam_coord_voxel)
                # update volume color
                rgb_img = frame.getRbgImg()
                r, g, b = rgb_img[v, u, :]
                color = [r, g, b, 1]
                if -truncation_distance / 2 < sdf < truncation_distance / 2:
                    colors.append(color)
            if len(colors) == 0:
                vertices_colors[k, :] = [1, 1, 1, 0]
            else:
                vertices_colors[k, :] = np.mean(np.array(colors), axis=0)
        return vertices_colors

    def checkPositionOfVolumeAndFrame(self, frame):
        """only be used to check the first frame with the volume.
        If you want to check with other frame, please run KinectFusion.run() fistly
        to update self.T_k2g."""
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('[x]')
        ax.set_ylabel('[y]')
        ax.set_zlabel('[z]')
        self._pltShowFrameVertices(frame, ax)
        self._pltShowVolumCube(self.volume, ax)

    def _pltShowVolumCube(self, volume: Volume, ax: plt.axes):
        cube = np.zeros((8, 3))
        l, w, h = list(volume.scale)
        cube[0, :] = np.array([0, 0, 0]) + volume.origin
        cube[1, :] = np.array([l, 0, 0]) + volume.origin
        cube[2, :] = np.array([l, w, 0]) + volume.origin
        cube[3, :] = np.array([0, w, 0]) + volume.origin
        cube[4, :] = np.array([0, 0, h]) + volume.origin
        cube[5, :] = np.array([l, 0, h]) + volume.origin
        cube[6, :] = np.array([l, w, h]) + volume.origin
        cube[7, :] = np.array([0, w, h]) + volume.origin

        linewideths = 1
        ax.plot(xs=cube[0:4, 0], ys=cube[0:4, 1], zs=cube[0:4, 2], linewidth=linewideths, color='r')
        ax.plot(xs=[cube[0, 0], cube[3, 0]],
                ys=[cube[0, 1], cube[3, 1]],
                zs=[cube[0, 2], cube[3, 2]], linewidth=linewideths, color='r')
        ax.plot(xs=cube[4:8, 0], ys=cube[4:8, 1], zs=cube[4:8, 2], linewidth=linewideths, color='r')
        ax.plot(xs=[cube[4, 0], cube[7, 0]],
                ys=[cube[4, 1], cube[7, 1]],
                zs=[cube[4, 2], cube[7, 2]], linewidth=linewideths, color='r')

        ax.plot(xs=[cube[0, 0], cube[4, 0]],
                ys=[cube[0, 1], cube[4, 1]],
                zs=[cube[0, 2], cube[4, 2]], linewidth=linewideths, color='r')
        ax.plot(xs=[cube[1, 0], cube[5, 0]],
                ys=[cube[1, 1], cube[5, 1]],
                zs=[cube[1, 2], cube[5, 2]], linewidth=linewideths, color='r')
        ax.plot(xs=[cube[2, 0], cube[6, 0]],
                ys=[cube[2, 1], cube[6, 1]],
                zs=[cube[2, 2], cube[6, 2]], linewidth=linewideths, color='r')
        ax.plot(xs=[cube[3, 0], cube[7, 0]],
                ys=[cube[3, 1], cube[7, 1]],
                zs=[cube[3, 2], cube[7, 2]], linewidth=linewideths, color='r')

    def _pltShowFrameVertices(self, frame: Frame, ax: plt.axes):
        local_vertices = frame.getLocalVertices()
        global_vertices = local_vertices @ self.T_k2g[:3, :3].T + self.T_k2g[:3, 3].reshape(1, 3)
        global_vertex_samples = self._downSample(global_vertices, sample_rate=0.002)
        ax.scatter(xs=global_vertex_samples[:, 0],
                   ys=global_vertex_samples[:, 1],
                   zs=global_vertex_samples[:, 2], color='b')

    def showVolume(self):
        volume = self.volume
        vertices = []
        colors = []
        count = 0
        for i in prange(volume.shape[0]):
            for j in range(volume.shape[1]):
                for k in range(volume.shape[2]):
                    count += 1
                    if count % (volume.voxel_num // 100) == 0 or count == volume.voxel_num:
                        print('\rComputing for showing volume {:.2f}% ...'.format(100 * count / volume.voxel_num), end='', flush=True)
                    vertex = [i, j, k]
                    vertices.append(vertex)
                    color = volume.getColor(i, j, k)
                    if color[0] is None:
                        color = [1, 1, 1]
                    else:
                        color = [color[0], color[1], color[2]]
                    colors.append(color)
        vertices = np.array(vertices)
        colors = np.array(colors)
        self._visualizePointCloud(vertices, colors)

    def _visualizePointCloud(self, vertices, colors):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(vertices)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([point_cloud])