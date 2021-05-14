import numpy as np
from .camera import Camera
from .dataset import Dataset
import cv2


class DataLoader(object):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    # TODO: this part can be done with Queue
    def __getitem__(self, item):
        x = 30

        bgrImg = cv2.imread(self.dataset.rgbPaths[item])
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_RGB2BGR)
        rgbImg = rgbImg[x:(480 - x), x:(600 - x), :]
        rgbImg = cv2.normalize(rgbImg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        depthImg = cv2.imread(self.dataset.depthPaths[item])
        depthImg = depthImg[:, :, 0]
        depthImg = depthImg[x:(480 - x), x:(600 - x)]

        img = {}
        img['rgb'] = rgbImg
        img['depth'] = depthImg
        return img

    def __len__(self):
        num_rgbImg = len(self.dataset.rgbPaths)
        num_depthImg = len(self.dataset.depthPaths)
        return min(num_depthImg, num_rgbImg)


