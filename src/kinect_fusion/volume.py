import numpy as np


class Volume(object):
    def __init__(self, scale=(5, 5, 5), shape=(256, 256, 256), origin=(0, 0, 0)):
        """Volume's scale range unit is cm. Origin means the volume cube should translate
        how many centimeters along xyz-axis (left-down corner is the origin by fault in
        right hand frame)."""
        self.scale = np.array(scale)
        self.shape = np.array(shape)
        self.voxel_num = shape[0] * shape[1] * shape[2]
        self.origin = np.array(origin)
        self.voxel_size = self.getVoxelSize(self.scale, self.shape)

        self.tsdf_mt = np.full(shape, None)
        self.tsdf_weight_mt = np.zeros(shape)
        self.color_mt = np.full((shape[0], shape[1], shape[2], 3), None)

    def getTsdf(self, x, y, z):
        """x, y, z is the coord of the volume coordinate system"""
        return self.tsdf_mt[x, y, z], self.tsdf_weight_mt[x, y, z]

    def setTsdf(self, x, y, z, tsdf, weight):
        self.tsdf_mt[x, y, z] = tsdf
        self.tsdf_weight_mt[x, y, z] = weight

    def getColor(self, x, y, z):
        return self.color_mt[x, y, z, :]

    def setColor(self, x, y, z, color):
        self.color_mt[x, y, z, :] = color

    def getVoxelSize(self, scale, shape):
        voxel_size = np.zeros(3)
        voxel_size = scale / shape
        return voxel_size

    def volumeCoord2globalCoord(self, x, y, z):
        glo_pos = np.zeros(3)
        glo_pos[0] = x * self.voxel_size[0]
        glo_pos[1] = y * self.voxel_size[1]
        glo_pos[2] = z * self.voxel_size[2]
        glo_pos += self.origin
        return glo_pos

    def volumeCoord2globalCoordArray(self, voxels):
        vertices = np.zeros_like(voxels)
        vertices = voxels * np.array(self.voxel_size).reshape(1, -1) + self.origin.reshape(1, 3)
        return vertices


