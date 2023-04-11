import pykitti as pk
import matplotlib.pyplot as plt
import torch
import numpy as np

from match_SuperGlue import matchSuperglue

from KITTI_Tutorial.kitti_tutorial_func import velo_to_range

from utils import loadmonodepthModel, convertImageToMonodepth2, evaluateMonodepth2

if __name__ == "__main__":
    # Set if plots should be created (to debug)
    debug = False
    plt.ioff()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ## Load two images from KITTI, 2 LiDAR point clouds and the provided calibration data ##
    # Data path
    data_path = 'Dataset'
    date = '2011_09_26'
    drive = '0001'
    nframes = 5
    upsampleFactor = 1
    smoothing = False
    checkPC = False

    # Extract nframes timestamps
    kitti_raw = pk.raw(data_path, date, drive, frames=range(0, nframes, 1))

    # Calibration Data
    K_gt = kitti_raw.calib.K_cam3
    T_gt = kitti_raw.calib.T_cam3_velo

    ## Convert the LiDAR point clouds to range images ##
    # Set visibility parameters 
    v_fov, h_fov = (2, -24.8), (-180,180) ### START AT TOP LEFT OF IMAGE
    v_res= 0.42 # 0.42
    h_res= 0.35 # 0.35

    # Images and LiDAR in grayscale
    images = []
    velodata = []
    rangeImages = []
    for i in range(nframes):
        images.append(kitti_raw.get_cam3(i))
        velodata.append(kitti_raw.get_velo(i)[:,0:3])
        # rangeImages.append(velo_points_2_pano(velodata[i], v_res, h_res, v_fov, h_fov, depth=True).astype(float))
        rangeImages.append(velo_to_range(velodata[i], v_res=v_res, h_res=h_res, v_fov=v_fov, h_fov=h_fov,recursive=True, scaling = 0.99).astype(float))

    if debug == True:
        # display result image
        figrange, axrange = plt.subplots(1,1, figsize = (13,3) )
        axrange.imshow(rangeImages[0])
        axrange.axis('off')
        plt.show()
        
        print(rangeImages[0].shape)

    ## Create monocular depth image from rgb/grayscale image using monodepth2 ##

    # Load monodepth2 model
    model_name = "mono_640x192"
    encoder, depth_decoder, loaded_dict_enc = loadmonodepthModel(model_name, device = device)

    monodepthImages = []
    for i in range(nframes):
        # Convert image to input to monodepth2
        image_pytorch, original_width, original_height = convertImageToMonodepth2(images[i], loaded_dict_enc)

        # Run monodepth2
        monodepthImages.append(evaluateMonodepth2(image_pytorch,encoder,depth_decoder,original_height,original_width))

    if debug == True:
        # Plot colormapped depth image
        vmax = np.percentile(monodepthImages[0], 95)

        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.imshow(images[0])
        plt.title("Input", fontsize=22)
        plt.axis('off')

        plt.subplot(212)
        plt.imshow(monodepthImages[0], cmap='magma', vmax=vmax)
        plt.title("Disparity prediction", fontsize=22)
        plt.axis('off')
        plt.show()

    print('FINISHED CREATING DEPTH IMAGES AND RANGE MAPS')

    ## Match features using superglue ##

    # Config Options
    nms_radius = 4 # SuperPoint Non Maximum Suppression (NMS) radius (Must be positive), default=4, type = int
    sinkhorn_iterations = 50 # Number of Sinkhorn iterations performed by SuperGlue , default=20, type=int
    match_threshold = 0.4 # SuperGlue match threshold, default=0.2, type=float
    keypoint_threshold = 0.005 # SuperPoint keypoint detector confidence threshold, default=0.005, type=float
    max_keypoints = 1024 # Maximum number of keypoints detected by Superpoint (\'-1\' keeps all keypoints), default=1024, type=int
    superglue = 'outdoor' # SuperGlue weights, choices={'indoor', 'outdoor'}, default='indoor'

    # Load the SuperPoint and SuperGlue models.
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    savePath = 'RES_SuperGlue'

    matchSuperglue(monodepthImages, rangeImages, images, velodata, v_fov, h_fov, v_res, h_res, T_gt, K_gt, config = config, savePath = savePath, device = device, smoothing = smoothing, upsampleFactor = upsampleFactor, checkPC = checkPC)