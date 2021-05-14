import numpy as np
import os


class Camera(object):
    def __init__(self, configPath):
        if not os.path.exists(configPath):
            raise ValueError('Can not find path: {:}'.format(configPath))
        # self.Kundistort = np.zeros(3)
        self.parameter = {}
        with open(configPath) as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.split('=')
                self.parameter[key] = float(value)

        # camera intrinsics
        K = np.eye(3)
        K[0, 0] = self.parameter['fx']
        K[1, 1] = self.parameter['fy']
        K[0, 2] = self.parameter['cx']
        K[1, 2] = self.parameter['cy']
        self.parameter['K'] = K

        # Distortion parameter
        distCoef = np.zeros((5, ))
        distCoef[0] = self.parameter['k1']
        distCoef[1] = self.parameter['k2']
        distCoef[2] = self.parameter['p1']
        distCoef[3] = self.parameter['p2']
        distCoef[4] = self.parameter['k3']
        self.parameter['distcoef'] = distCoef