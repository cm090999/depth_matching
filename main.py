import pykitti as pk
import sys

sys.path.insert(1, 'KITTI_Tutorial')

from ST_depth_correspondence import helper_func
from KITTI_Tutorial.kitti_tutorial_func import velo_points_2_pano

if __name__ == "__main__":
    
    ## Load two images from KITTI, 2 LiDAR point clouds and the provided calibration data
    # Data path
    data_path = 'Dataset'
    date = '2011_09_26'
    drive = '0001'
    nframes = 2

    # Extract nframes timestamps
    kitti_raw = pk.raw(data_path, date, drive, frames=range(0, 2, 1))

    # Images in grayscale
    image0 = kitti_raw.get_cam3(0)
    image1 = kitti_raw.get_cam3(1)

    # LiDAR
    velo0 = kitti_raw.get_velo(0)
    velo1 = kitti_raw.get_velo(1)

    # Calibration Data
    K_gt = kitti_raw.calib.K_cam3
    T_gt = kitti_raw.calib.T_cam3_velo

    ## Convert the LiDAR point clouds to range images


