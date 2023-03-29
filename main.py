import os
from pathlib import Path
import pykitti as pk
import matplotlib.pyplot as plt
import torch
import PIL.Image as pil
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.cm as cm

from monodepth2.utils import download_model_if_doesnt_exist
import monodepth2.networks as networks

from ST_depth_correspondence import helper_func
from KITTI_Tutorial.kitti_tutorial_func import velo_points_2_pano

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot, estimate_pose

from utils import loadmonodepthModel, convertImageToMonodepth2, evaluateMonodepth2, upsampleRangeImage, get3dpointFromRangeimage

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
    nframes = 10
    upsampleFactor = 6

    # Extract nframes timestamps
    kitti_raw = pk.raw(data_path, date, drive, frames=range(0, nframes, 1))

    # Calibration Data
    K_gt = kitti_raw.calib.K_cam3
    T_gt = kitti_raw.calib.T_cam3_velo

    ## Convert the LiDAR point clouds to range images ##
    # Set visibility parameters 
    v_fov, h_fov = (-24.9, 2), (-180,180)
    v_res=0.8#0.42
    h_res=0.8#0.35

    # Images and LiDAR in grayscale
    images = []
    velodata = []
    rangeImages = []
    for i in range(nframes):
        images.append(kitti_raw.get_cam3(i))
        velodata.append(kitti_raw.get_velo(i)[:,0:3])
        rangeImages.append(velo_points_2_pano(velodata[i], v_res, h_res, v_fov, h_fov, depth=True).astype(float))

    if debug == True:
        # display result image
        figrange, axrange = plt.subplots(1,1, figsize = (13,3) )
        # plt.title("Result of Vertical FOV ({} , {}) & Horizontal FOV ({} , {})".format(v_fov[0],v_fov[1],h_fov[0],h_fov[1]))
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
    matching = Matching(config).eval().to(device)

    ## Apply SuperGlue + Pose Estimation for first image with n_time next images

    # Initialize Empty lists to store results
    R_rel_list_SuperGlue = []
    t_rel_list_SuperGlue = []

    R_rel_list_cv2 = []
    t_rel_list_cv2 = []

    # Define directory to save transformations
    output_dir_tf = Path().absolute() / 'KITTI_RES'
    output_dir_tf.mkdir(exist_ok=True, parents=True)

    # Define directory to save matches
    output_dir_matches = output_dir_tf / 'Matches'
    output_dir_matches.mkdir(exist_ok=True, parents=True)

    # Define directory to save range images
    output_dir_range = output_dir_tf / 'Range'
    output_dir_range.mkdir(exist_ok=True, parents=True)

    # Define directory to save monodepth2 images
    output_dir_md2 = output_dir_tf / 'monodepth2'
    output_dir_md2.mkdir(exist_ok=True, parents=True)

    # Define directory to save monodepth2 images
    output_dir_proj = output_dir_tf / 'reprojection'
    output_dir_proj.mkdir(exist_ok=True, parents=True)

    for i in range(nframes):
        # Keep unmodified range image for PnP
        rangeImage_tmp = rangeImages[i]

        # Normalize Images
        monodepthImages[i] = ((monodepthImages[i] - np.min(monodepthImages[i])) / np.max(monodepthImages[i])) * 255
        rangeImages[i] = ((rangeImages[i] - np.min(rangeImages[i])) / np.max(rangeImages[i])) * 255

        # Resize range image
        rangeImages[i] = upsampleRangeImage(rangeImages[i],upsampleFactor)

        # Histogram equalization
        monodepthImages[i] = cv2.equalizeHist(monodepthImages[i].astype(np.uint8))
        rangeImages[i] = cv2.equalizeHist(rangeImages[i].astype(np.uint8))

        fileName = str(i).zfill(3) + '.png'
        savePathMatches = output_dir_matches / fileName
        savePathRange = output_dir_range / fileName
        savePathmd2 = output_dir_md2 / fileName
        savePathproj = output_dir_proj / fileName

        # Save Range Images and monodepth2 images
        cv2.imwrite(str(savePathmd2),monodepthImages[i])
        cv2.imwrite(str(savePathRange),rangeImages[i])

        # Tranform and normalize images
        inp0 = frame2tensor(monodepthImages[i], device)
        inp1 = frame2tensor(rangeImages[i], device)

        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        # Keep the matching keypoints.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]

        # Make Plot
        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']

        make_matching_plot(
            monodepthImages[i], rangeImages[i], kpts0, kpts1, mkpts0, mkpts1, color,
            text, savePathMatches, show_keypoints=True,
            fast_viz=True, opencv_display=True, opencv_title='Matches')
        
        # Get number of matches
        nmatches, _ = np.shape(mkpts0)

        if nmatches >= 4:

        # Get 3d Coordinates from matches
            matches_3d = get3dpointFromRangeimage(rangeImage_tmp,mkpts1,v_fov,h_fov,v_res,h_res, upsampleFactor, depth=True)
            matches_2d = mkpts0

            # Solve PnP problem
            _,Rvec,tvec = cv2.solvePnP(matches_3d,matches_2d,K_gt,np.array([[0],[0],[0],[0]]))  #,flags=cv2.SOLVEPNP_ITERATIVE

            # Append solution to list
            R_rel_list_cv2.append(Rvec)
            t_rel_list_cv2.append(tvec)

            # Reproject LiDAR to image
            R_pnp = helper_func.rtvec_to_matrix(Rvec, tvec)
            velo_tf = helper_func.transformPC(velodata[i],R_pnp)
            
            helper_func.plotOverlay(rgb = images[i],lidar = velo_tf, savepath=savePathproj)

        # Pose Estimation with provided function
        pose = estimate_pose(mkpts0,mkpts1,K_gt,K_gt,1.)
        if pose != None:
            R_rel_list_SuperGlue.append(pose[0])
            t_rel_list_SuperGlue.append(pose[1])
        else:
            R_rel_list_SuperGlue.append(-1)
            t_rel_list_SuperGlue.append(-1)

    # Define dict to extract results
    resDict = {'R_rel_list_SuperGlue': R_rel_list_SuperGlue,
               't_rel_list_SuperGlue': t_rel_list_SuperGlue,
               'R_rel_list_cv2': R_rel_list_cv2,
               't_rel_list_cv2': t_rel_list_cv2}
    
    # Define location to save results
    fout = output_dir_tf / 'Pose.txt'
    fo = open(fout, "w")

    # Safe Ground Truth
    fo.write('T_GroundTruth >>> \n\n')
    fo.write(str(T_gt) + '\n')

    # Safe result dict
    for k, v in resDict.items():
        fo.write(str(k) + ' >>> ' + '\n\n')
        for i in range(len(v)):
            fo.write(str(i) + ': \n')
            fo.write(str(v[i]) + '\n')
            fo.write('\n')

    fo.close()


