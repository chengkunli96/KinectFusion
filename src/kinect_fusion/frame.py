import numpy as np
import cv2

class Frame(object):
    def __init__(self, depth_img, rgb_img, camera_parameter):
        self.parameter = camera_parameter
        self.raw_depth_img = depth_img.copy()
        self.depth_img = self.raw_depth_img / self.parameter['ds']
        self.rgb_img = rgb_img
        # valid vertices which is filtered by mask
        self.local_vertices = None
        self.local_normals = None
        self.mask = None

        self.computeLocalVerticesAndNormals()

    def getLocalVertices(self):
        return self.local_vertices

    def getLocalNormals(self):
        return self.local_normals

    def getRbgImg(self):
        return self.rgb_img

    def getDepthImg(self):
        return self.depth_img

    def getKinectUnscaledDepthImg(self):
        return self.raw_depth_img

    def getDepthValidMask(self):
        return self.mask

    def getCameraParameters(self):
        return self.parameter

    def computeLocalVerticesAndNormals(self):
        """
        Get the vertices and normal map.

        :param parameter: camera intrinsic parameters
        :param img: 2 dimension np array
        :return: VK: vertices (N*3) in the image frame
                NK: normals (N*3) in the image frame
                M: mask, where the depth image has values
        """
        parameter = self.parameter
        img = self.raw_depth_img
        h, w = img.shape
        Dk = cv2.bilateralFilter(img, 20, 20, 20)
        Dk = img
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
        for v in range(img.shape[0] - 1):
            for u in range(img.shape[1] - 1):
                n = np.zeros(3, )
                if 16 <= Vk[v, u, 2] <= 24:
                    n = np.cross(Vk[v + 1, u, :] - Vk[v, u, :], Vk[v, u + 1, :] - Vk[v, u, :])
                if np.linalg.norm(n) != 0 and img[v, u] != 0:
                    M[v, u] = 1
                    Nk[v, u, :] = n / np.linalg.norm(n)
        Vk = Vk[M == 1].reshape(-1, 3)
        Nk = Nk[M == 1].reshape(-1, 3)
        # return Vk, Nk, M, Dk
        self.local_vertices = Vk
        self.local_normals = Nk
        self.mask = M
        # return Vk, Nk, M

    def contain(self, uv_pos):
        """Check whether uv point is in this image and wether this place
        has valid depth value."""
        h, w = self.depth_img.shape
        u, v = uv_pos
        if 0 <= u < w and 0 <= v < h:
            if self.mask[v, u] == 1:
                return True
            else:
                return False
        else:
            return False

