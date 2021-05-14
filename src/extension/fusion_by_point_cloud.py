import cv2
import time
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
from .tsdf_fusion import TsdfFusion


class KinectFusion:
    def __init__(self):
        # timestep
        self.timestep = 0
        # vertex map
        # normal vectors
        self.VNdict = {}
        # global vertex
        self.GlobalV = None
        self.GlobalN = None
        self.GVsize = 0
        # transformation local->global
        self.Tkg = np.eye(4)
        # TSDF
        self.TSDF = {}
        self.colors = None

    def Process(self, parameter, rgbImg, depthImg):
        # depth image -> three resolutions
        print('================================')
        print('[INFO] timestep = %d'%(self.timestep))
        startTime = time.time()
        h, w = depthImg.shape
        level1 = depthImg
        level2 = cv2.resize(depthImg, (int(w/2), int(h/2)))
        level3 = cv2.resize(depthImg, (int(w/4), int(h/4)))
        print('[INFO] Get DepthImg Pyramids')
        # get vertex maps and normal maps 
        Vk1, Nk1, M1, Dk1 = self.Measurement(parameter, level1)
        Vk2, Nk2, M2, Dk2 = self.Measurement(parameter, level2)
        Vk3, Nk3, M3, Dk3 = self.Measurement(parameter, level3)
        # plt.figure()
        # plt.subplot(2, 3, 1)
        # plt.imshow(Dk1, cmap=plt.get_cmap('gray'))
        # plt.subplot(2, 3, 2)
        # plt.imshow(Dk2, cmap=plt.get_cmap('gray'))
        # plt.subplot(2, 3, 3)
        # plt.imshow(Dk3, cmap=plt.get_cmap('gray'))
        # plt.subplot(2, 3, 4)
        # plt.imshow(M1, cmap=plt.get_cmap('gray'))
        # plt.subplot(2, 3, 5)
        # plt.imshow(M2, cmap=plt.get_cmap('gray'))
        # plt.subplot(2, 3, 6)
        # plt.imshow(M3, cmap=plt.get_cmap('gray'))
        # plt.show()
        print('[INFO] Get VKs and Nks')
        # print(Vk1.shape, Vk2.shape, Vk3.shape)
        # downsample
        Vk1sampled, Nk1sampled, colorMask = self.Downsample(Vk1, Nk1, M1, sample_rate=0.02)# 0.01
        Vk2sampled, Nk2sampled, _    = self.Downsample(Vk2, Nk2, M2, sample_rate=0.02)# 0.04
        Vk3sampled, Nk3sampled, _    = self.Downsample(Vk3, Nk3, M3, sample_rate=0.02)# 0.16
        print('[INFO] Get Downsampled VKs and Nks')
        
        # print(Vk1sampled.shape, Vk2sampled.shape, Vk3sampled.shape)
    
        if self.timestep == 0:
            print('[INFO] Initialize Global V, N')
            self.GlobalV = Vk1sampled
            self.GlobalN = Nk1sampled
            self.colors = rgbImg[M1 == 1].reshape(-1, 3)[colorMask]/255.0
            self.TSDF['F'] = np.zeros((self.GlobalV.shape[0], 1))
            self.TSDF['W'] = np.ones((self.GlobalV.shape[0], 1))
            print('[INFO] Global size(V, N) = ', self.GlobalV.shape, self.GlobalN.shape)
        else:
            print('[INFO] Compute Transformation T(k->global)')
            Tkkpre = self.PoseEstimation(
                Vk1sampled, Nk1sampled,
                Vk2sampled, Nk2sampled,
                Vk3sampled, Nk3sampled)
            self.Tkg = self.Tkg @ Tkkpre

            # ============ TSDF Fusion ============
            print('[INFO] Fusion')
            # print(self.GlobalV[10:20, :])
            depth_img = depthImg / parameter['ds']
            self.GlobalV, self.GlobalN, self.colors, self.TSDF['F'], self.TSDF['W'] = TsdfFusion(
                self.TSDF['W'], self.TSDF['F'],
                self.GlobalV, self.GlobalN, self.colors,
                Vk1sampled, Nk1sampled, depth_img, rgbImg/255.0, parameter['K'],
                self.Tkg, truncation_distance=1
            )

        print("++++++++++++")
        print(self.GlobalV.shape[0])
        if self.timestep >= 10:
            self.VisualizationPC(self.GlobalV, self.colors)
        self.VNdict['V1'] = Vk1sampled
        self.VNdict['N1'] = Nk1sampled
        self.VNdict['V2'] = Vk2sampled
        self.VNdict['N2'] = Nk2sampled
        self.VNdict['V3'] = Vk3sampled
        self.VNdict['N3'] = Nk3sampled
        self.GVsize = self.GlobalV.shape[0]

        endTime = time.time()
        print('[INFO] Time = %.3f'%(endTime - startTime))
        self.timestep += 1

    def Measurement(self, parameter, img):
        h, w = img.shape
        Dk = cv2.bilateralFilter(img, 20, 20, 20)
        # Vk = np.zeros((h, w, 3))
        Nk = np.zeros((h, w, 3))
        M = np.zeros((h, w))
        X_range = range(img.shape[1])
        Y_range = range(img.shape[0])
        X, Y = np.meshgrid(X_range, Y_range)
        Z = (Dk / parameter['ds']).reshape(-1, 1)
        X = (X.reshape(-1, 1) - parameter['cx']) * Z / parameter['fx']
        Y = (Y.reshape(-1, 1) - parameter['cy']) * Z / parameter['fy']
        Vk = np.hstack([X, Y, Z]).reshape(h, w, 3)
        for v in range(img.shape[0]-1):
            for u in range(img.shape[1]-1):
                n = np.zeros(3,)
                if 16 <= Vk[v, u, 2] <= 24:
                    n = np.cross(Vk[v+1, u, :]-Vk[v, u, :], Vk[v, u+1, :]-Vk[v, u, :])
                if np.linalg.norm(n) != 0:
                    M[v, u] = 1
                    Nk[v, u, :] = n / np.linalg.norm(n)
        Vk = Vk[M == 1].reshape(-1, 3)
        Nk = Nk[M == 1].reshape(-1, 3)
        return Vk, Nk, M, Dk

    def Downsample(self, V, N, M, sample_rate=0.01):
        mask = np.random.randint(0, np.sum(M), int(np.sum(M)*sample_rate))
        return V[mask], N[mask], mask

    def PoseEstimation(self, V1, N1, V2, N2, V3, N3):
        # ICP in level3
        T3, loss3 = self.P2PICP(self.VNdict['V3'], self.VNdict['N3'], V3, N3, epoch=4) # 4
        print(loss3[-1])
        # self.VisualizationError(loss3, name='loss3')
        # ICP in level2
        tmpV2 = V2 @ T3[:3, :3].T + T3[:3, 3].reshape(1, 3)
        tmpN2 = N2 @ T3[:3, :3].T
        T2, loss2 = self.P2PICP(self.VNdict['V2'], self.VNdict['N2'], tmpV2, tmpN2, epoch=5)# 5
        print(loss2[-1])
        T = T2 @ T3
        # self.VisualizationError(loss2, name='loss2')
        # ICP in level1
        tmpV1 = V1 @ T[:3, :3].T + T[:3, 3].reshape(1, 3)
        tmpN1 = N1 @ T[:3, :3].T
        # self.VisualizationMesh(self.VNdict['V1'], tmpV1)
        T1, loss1 = self.P2PICP(self.VNdict['V1'], self.VNdict['N1'], tmpV1, tmpN1, epoch=10) # 10
        print(loss1[-1])
        T = T1 @ T
        # self.VisualizationError(loss1, name='loss1')
        return T

    def UpdateReconstruction(self, parameter, img, mu=0.1):
        Tgk = np.linalg.inv(self.Tkg)
        x = (self.GlobalV @ Tgk[:3, :3].T + Tgk[:3, 3].reshape(1, 3)) @ parameter['K'].T
        x = (x[:, :2] / x[:, 2].reshape(-1, 1)).astype(int)
        Lambda = np.linalg.norm(np.hstack([x, np.ones((x.shape[0], 1))]) @ np.linalg.inv(parameter['K']).T, axis=1)
        xmask = (x[:, 0] < img.shape[1]) & (x[:, 1] < img.shape[0])
        xflip = np.fliplr(x[xmask])
        F = 1/Lambda[xmask]*np.linalg.norm(self.Tkg[:3, 3].reshape(1, 3)-self.GlobalV[xmask], axis=1) - img[xflip[:, 0], xflip[:, 1]]/parameter['ds']
        phimask = (F >= -mu)
        F[phimask] = np.minimum(1, F[phimask]/mu) * np.sign(F[phimask])
        F[1-phimask] = -1
        ssss = np.hstack([x[xmask][phimask], F[phimask].reshape(-1, 1)])
        self.VisualizationMesh(ssss, np.zeros((1,)))
        W = 1
        return F, W, xmask, phimask

    def SurfacePrediction(self):
        pass

    def P2PICP(self, v1, n1, v2, n2, epoch=5, sample_rate=1, threshold=[0.2, np.sqrt(3)/2]):
        # transformation: v2 -> v1
        T = np.eye(4)
        loss = []
        v1_KDTree = KDTree(v1)
        for _ in range(epoch):
            # random sampling 
            sample_num = int(v2.shape[0]*sample_rate)
            sample = np.random.randint(0, v2.shape[0], sample_num)
            s = v2[sample]
            # find the closest points in v1
            dis, idx = v1_KDTree.query(s, k=1)
            dis_mean = np.mean(dis)
            dis_std = np.std(dis)
            d = v1[idx]
            n = n1[idx]
            
            # Ax = b, construct A and b
            A = np.zeros((len(idx), 6))
            b = np.zeros((len(idx), 1))
            A[:, 0] = n[:, 2]*s[:, 1] - n[:, 1]*s[:, 2]
            A[:, 1] = n[:, 0]*s[:, 2] - n[:, 2]*s[:, 0]
            A[:, 2] = n[:, 1]*s[:, 0] - n[:, 0]*s[:, 1]
            A[:, 3:6] = n
            b[:, 0] = np.sum(n*(d-s), axis=1)
            # compute the pseudo inverse of A
            # print(A)
            U, W, VT = np.linalg.svd(A)
            W_pinv = np.zeros((6, len(idx)))
            W_pinv[:6, :6] = np.diag(1/W)
            A_pinv = VT.T @ W_pinv @ U.T
            # compute transformation
            x = A_pinv @ b
            t = x[3:6]
            r = x[0:3]
            T_tmp = T.copy()
            t_matrix = np.eye(4)
            t_matrix[:3, 3] = np.squeeze(t)
            gamma = np.eye(4)
            gamma[0, 0] = np.cos(r[2])
            gamma[1, 1] = np.cos(r[2])
            gamma[0, 1] = -np.sin(r[2])
            gamma[1, 0] = np.sin(r[2])
            beta = np.eye(4)
            beta[0, 0] = np.cos(r[1])
            beta[2, 2] = np.cos(r[1])
            beta[0, 2] = np.sin(r[1])
            beta[2, 0] = -np.sin(r[1])
            alpha = np.eye(4)
            alpha[1, 1] = np.cos(r[0])
            alpha[2, 2] = np.cos(r[0])
            alpha[1, 2] = -np.sin(r[0])
            alpha[2, 1] = np.sin(r[0])
            r_matrix = gamma @ beta @ alpha
            T_tmp = t_matrix @ r_matrix

            v2tmp = (T_tmp[:3, :3] @ v2.T).T + T_tmp[:3, 3].reshape(1, -1)
            n2tmp = n2 @ T_tmp[:3, :3].T
            dis, _ = v1_KDTree.query(v2tmp, k=1)
            loss.append(np.mean(dis))
            if len(loss) > 1 and loss[-1] <= loss[-2]:
                # update v2
                v2 = v2tmp
                n2 = n2tmp
                # update T
                T = T_tmp @ T
        return T, loss

    def VisualizationMesh(self, v1, v2, name='mesh'):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(v1[:, 0], -v1[:, 2], v1[:, 1], c='b', s=0.5, label='M1')
        if v2.size > 1:
            ax.scatter(v2[:, 0], -v2[:, 2], v2[:, 1], c='r', s=0.5, label='M2')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        ax.set_title(name)
        # plt.show()

    def VisualizationPC(self, pointCloud, colors):
        # vis = o3d.Visualizer()
        # vis.create_window(window_name="Global Vertex Map")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pointCloud)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([point_cloud])

    def VisualizationError(self, loss, name='loss'):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(loss)), loss)
        ax.set_title(name)
        # plt.show()

class Dataset:
    def __init__(self, dataPath):
        self.rgbPaths = []
        self.depthPaths = []
        with open(dataPath + '/associate.txt') as f:
            lines = f.readlines()
            for line in lines:
                # _, rgbPath, _, depthPath = line.split(' ', 4)
                line = line.split(' ')
                self.rgbPaths.append(dataPath + line[1])# rgbPath)
                self.depthPaths.append(dataPath + line[3][:-1])# depthPath)

class Camera:
    def __init__(self, configPath):
        # self.Kundistort = np.zeros(3)
        self.parameter = {}
        with open(configPath) as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.split('=')
                self.parameter[key] = float(value)
        K = np.eye(3)
        K[0, 0] = self.parameter['fx']
        K[1, 1] = self.parameter['fy']
        K[0, 2] = self.parameter['cx']
        K[1, 2] = self.parameter['cy']
        self.parameter['K'] = K
        distCoef = np.zeros((5, ))
        distCoef[0] = self.parameter['k1']
        distCoef[1] = self.parameter['k2']
        distCoef[2] = self.parameter['p1']
        distCoef[3] = self.parameter['p2']
        distCoef[4] = self.parameter['k3']
        self.parameter['distcoef'] = distCoef

if __name__ == "__main__":

    camera = Camera('../data/camera_config.txt')
    data = Dataset('../data/rgbd_dataset_freiburg2_xyz/')
    kf = KinectFusion()

    for i in range(len(data.rgbPaths)):
        x = 40

        bgrImg = cv2.imread(data.rgbPaths[i])
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_RGB2BGR)
        rgbImg = rgbImg[x:(480-x), x:(600-x), :]
        depthImg = cv2.imread(data.depthPaths[i])
        depthImg = depthImg[:, :, 0]
        depthImg = depthImg[x:(480-x), x:(600-x)]
        
        kf.Process(camera.parameter, rgbImg, depthImg)
