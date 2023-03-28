import os
from pathlib import Path
import pykitti as pk
import matplotlib.pyplot as plt
import torch
import PIL.Image as pil
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

if __name__ == "__main__":
    # Set if plots should be created (to debug)
    debug = False
    plt.ioff()
    
    ## Load two images from KITTI, 2 LiDAR point clouds and the provided calibration data ##
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
    velo0 = kitti_raw.get_velo(0)[:,0:3]
    velo1 = kitti_raw.get_velo(1)[:,0:3]

    # Calibration Data
    K_gt = kitti_raw.calib.K_cam3
    T_gt = kitti_raw.calib.T_cam3_velo

    ## Convert the LiDAR point clouds to range images ##
    # Set visibility parameters 
    v_fov, h_fov = (-24.9, 2), (-180,180)
    v_res=0.6#0.42
    h_res=0.6#0.35

    range0 = velo_points_2_pano(velo0, v_res, h_res, v_fov, h_fov, depth=False).astype(float)
    range1 = velo_points_2_pano(velo1, v_res, h_res, v_fov, h_fov, depth=False).astype(float)

    if debug == True:
        # display result image
        figrange, axrange = plt.subplots(1,1, figsize = (13,3) )
        # plt.title("Result of Vertical FOV ({} , {}) & Horizontal FOV ({} , {})".format(v_fov[0],v_fov[1],h_fov[0],h_fov[1]))
        axrange.imshow(range0)
        axrange.axis('off')
        plt.show()
        
        print(range0.shape)

    ## Create monocular depth image from rgb/grayscale image using monodepth2 ##

    # Load monodepth2 model
    model_name = "mono_640x192"

    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()

    # Convert image to input to monodepth2
    original_width0, original_height0 = image0.size
    original_width1, original_height1 = image1.size

    feed_height0 = loaded_dict_enc['height']
    feed_width0 = loaded_dict_enc['width']
    feed_height1 = loaded_dict_enc['height']
    feed_width1 = loaded_dict_enc['width']
    image0_resized = image0.resize((feed_width0, feed_height0), pil.LANCZOS)
    image1_resized = image1.resize((feed_width1, feed_height1), pil.LANCZOS)

    image0_pytorch = transforms.ToTensor()(image0_resized).unsqueeze(0)
    image1_pytorch = transforms.ToTensor()(image1_resized).unsqueeze(0)

    # Monodepth2 inference
    with torch.no_grad():
        features0 = encoder(image0_pytorch)
        features1 = encoder(image1_pytorch)
        outputs0 = depth_decoder(features0)
        outputs1 = depth_decoder(features1)

    disp0 = outputs0[("disp", 0)]
    disp1 = outputs1[("disp", 0)]

    # Resize image to original format
    disp0_resized = torch.nn.functional.interpolate(disp0,
        (original_height0, original_width0), mode="bilinear", align_corners=False)
    disp1_resized = torch.nn.functional.interpolate(disp1,
        (original_height1, original_width1), mode="bilinear", align_corners=False)
    
    # Save images
    depth0_md2 = disp0_resized.squeeze().cpu().numpy()
    depth1_md2 = disp1_resized.squeeze().cpu().numpy()


    if debug == True:
        # Plot colormapped depth image
        disp_resized_np = disp0_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)

        plt.figure(figsize=(10, 10))
        plt.subplot(211)
        plt.imshow(image0)
        plt.title("Input", fontsize=22)
        plt.axis('off')

        plt.subplot(212)
        plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    # Tranform image0 for matching
    inp0 = frame2tensor(range0, device)

    ## Apply SuperGlue + Pose Estimation for first image with n_time next images

    # Initialize Empty lists to store results
    R_rel_list_SuperGlue = []
    t_rel_list_SuperGlue = []

    R_rel_list_cv2 = []
    t_rel_list_cv2 = []

    output_dir = Path().absolute() / 'KITTI_RES'
    output_dir.mkdir(exist_ok=True, parents=True)

    for i in range(1,2):

        fileName = str(i).zfill(3) + '.png'
        savePath = output_dir / fileName

        # Tranform and normalize images
        inp0 = frame2tensor((depth0_md2 - np.min(depth0_md2)) / np.max(depth0_md2) * 255, device)
        inp1 = frame2tensor((range0 - np.min(range0)) / np.max(range0) * 255, device)

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

        ## Make Plot
        # Display extra parameter info.
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']

        make_matching_plot(
            depth0_md2 / np.max(depth0_md2) * 255, range0 / np.max(range0) * 255, kpts0, kpts1, mkpts0, mkpts1, color,
            text, savePath, show_keypoints=True,
            fast_viz=True, opencv_display=True, opencv_title='Matches')

        # Pose Estimation with provided function
        pose = estimate_pose(mkpts0,mkpts1,K_gt,K_gt,1.)
        if pose != None:
            R_rel_list_SuperGlue.append(pose[0])
            t_rel_list_SuperGlue.append(pose[1])

        # essential_matrix, inliers_ess = cv2.findEssentialMat(mkpts0,mkpts1,K_gt,threshold=1.0, prob=0.99999, method=cv2.RANSAC)

        # # Get relative pose of the two images
        # _, R_rel,t_rel, inliers_pose = cv2.recoverPose(essential_matrix,mkpts0,mkpts1,K_gt)
        # R_rel_list_cv2.append(R_rel)
        # t_rel_list_cv2.append(t_rel)

    resDict = {'R_rel_list_SuperGlue': R_rel_list_SuperGlue,
               't_rel_list_SuperGlue': t_rel_list_SuperGlue,
               'R_rel_list_cv2': R_rel_list_cv2,
               't_rel_list_cv2': t_rel_list_cv2}
    


    fout = output_dir / 'Pose.txt'
    fo = open(fout, "w")

    for k, v in resDict.items():
        fo.write(str(k) + ' >>> ' + '\n\n')
        for i in range(len(v)):
            fo.write(str(v[i]) + '\n')
            fo.write('\n')

    fo.close()


