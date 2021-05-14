import os


class Dataset(object):
    def __init__(self, dataPath):
        if not os.path.exists(dataPath):
            raise ValueError('Can not find path: {:}'.format(dataPath))
        self.rgbPaths = []
        self.depthPaths = []
        with open(dataPath + '/associate.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                self.rgbPaths.append(dataPath + line[1])# rgbPath)
                self.depthPaths.append(dataPath + line[3][:-1])# depthPath)