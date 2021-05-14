import matplotlib.pyplot as plt
from os.path import join as opj
import os
import json
import open3d as o3d

from data_loader import *
from kinect_fusion import *
from utils.pyrender_show import showMesh
from utils.open3d_show import showPointCloud


# for checking whether your defined volume is suitable
CHECK_VOLUME = True
LOGGING = True
if __name__ == '__main__':
    # argument
    config_dict = {
        'camera_config_path': '../data/camera_config.txt',
        'dataset_path': '../data/rgbd_dataset_freiburg2_xyz/',
        'pose_estimation_method': 'icp method',  # "Park's method",
        'ICP': {
            'sample_rate': 0.2,
            'mode': 'frame2frame',
            'threshold': 0.1,
        },
        'truncation_distance': 1,
        'volume_resolution': 5,
        'color_method': 'by volume',
    }

    # for saving experiment's results
    out_dir = None
    exp_out_dir = None
    if LOGGING:
        out_dir = '../output/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # check the num of dirs of the out_dir
        dirnum = 0
        for lists in os.listdir(out_dir):
            sub_path = opj(out_dir, lists)
            if os.path.isdir(sub_path):
                dirnum = dirnum + 1
        # experiment
        exp_out_dir = opj(out_dir, 'experiment_{:02d}'.format(dirnum))
        if not os.path.exists(exp_out_dir):
            os.makedirs(exp_out_dir)
        # write down the experiment configuration
        json_str = json.dumps(config_dict, indent=4)
        with open(opj(exp_out_dir, 'configuration.json'), 'w') as json_file:
            json_file.write(json_str)

    camera = Camera(config_dict['camera_config_path'])
    dataset = Dataset(config_dict['dataset_path'])
    dataloader = DataLoader(dataset=dataset)

    kinect = KinectFusion()
    if config_dict['pose_estimation_method'] == 'icp method':
        kinect.configPosEstimationMethod('icp')
        kinect.configIcp(sample_rate=config_dict['ICP']['sample_rate'],
                         pose_estimation_mode=config_dict['ICP']['mode'],
                         threshold=config_dict['ICP']['threshold'])
    else:
        kinect.configPosEstimationMethod('Park')
    kinect.configTsdf(truncation_distance=config_dict['truncation_distance'])
    kinect.configVolume(scale=(30, 30, 30),
                        shape=(config_dict['volume_resolution'],
                               config_dict['volume_resolution'],
                               config_dict['volume_resolution']),
                        origin=(-15, -17.5, 5))
    kinect.configMeshColorComputingMethod(config_dict['color_method'])

    if CHECK_VOLUME:
        img = dataloader[0]
        frame = Frame(img['depth'], img['rgb'], camera.parameter)
        kinect.checkPositionOfVolumeAndFrame(frame)
        plt.show()

    for (i, img) in enumerate(dataloader):
        n = 0  # start from which frame
        if i > n:
            depth_img = img['depth']  # the raw depth image from Kinect device which is not scaled
            rgb_img = img['rgb']
            frame = Frame(depth_img, rgb_img, camera.parameter)

            kinect.run(frame)

            if LOGGING:
                # saving for each iteration
                pcd = kinect.getPointClould()
                o3d.io.write_point_cloud(opj(exp_out_dir, 'pointcloud_frame{:02d}.pcd'.format(i + 1)), pcd)
                mesh = kinect.getMesh()
                mesh.export(opj(exp_out_dir, 'mesh_frame{:02d}.obj'.format(i + 1)))
            if i - n >= 20:
                # show the result
                pcd = kinect.getPointClould(sample_rate=0.2)
                showPointCloud(vertices=pcd.points, colors=pcd.colors)
                mesh = kinect.getMesh()
                showMesh(mesh)

