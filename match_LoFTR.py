from pathlib import Path
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from copy import deepcopy


from ST_depth_correspondence import helper_func

from LoFTR.src.loftr import LoFTR, default_cfg

from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot

from utils import upsampleRangeImage, get3dpointFromRangeimage, plot3dPoints, transformationVecLoss

def matchLoFTR(images0, images1, original_images, velodata, v_fov, h_fov, v_res, h_res, T_gt, K_gt, opt, savePath, device = 'cpu', smoothing = False, upsampleFactor = 1, checkPC = False, aggrMatches = 1):

    # Initialize empty matching kpts buffers
    buffer_matches2d = []
    buffer_matches3d = []

    nframes = len(images0)

    # Initialize LoFTR
    _default_cfg = deepcopy(default_cfg)
    # _default_cfg['coarse']['temp_bug_fix'] = True # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load(opt['weight'])['state_dict'])
    matching = matcher.eval().to(device=device)

    ## Apply SuperGlue + Pose Estimation for first image with n_time next images

    # Initialize Empty lists to store results
    numberMatches = []
    T_rel = []

    # Define directory to save transformations
    output_dir_tf = Path().absolute() / savePath
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

    # Define directory to save reprojected images
    output_dir_proj = output_dir_tf / 'reprojection'
    output_dir_proj.mkdir(exist_ok=True, parents=True)

    for i in range(nframes):
        plt.close('all') 
        print('Work on Frame #' + str(i))

        # Keep unmodified range image for PnP
        rangeImage_tmp = images1[i]
        image0 = images0[i]
        image1 = images1[i]

        # Normalize Images and verify to be same type
        image0 = (((image0 - np.min(image0)) / np.max(image0)) * 255).astype(np.float32)
        image1 = (((image1 - np.min(image1)) / np.max(image1)) * 255).astype(np.float32)

        # Resize range image
        image1 = upsampleRangeImage(image1,upsampleFactor)

        # Apply Gaussian Blur
        if smoothing == True:
            image1 = cv2.GaussianBlur(image1,(5,5),0)

        # # Histogram equalization
        # image0 = cv2.equalizeHist(image0.astype(np.float32))
        # image1 = cv2.equalizeHist(image1.astype(np.float32))

        fileName = str(i).zfill(3) + '.png'
        savePathMatches = output_dir_matches / fileName
        savePathRange = output_dir_range / fileName
        savePathmd2 = output_dir_md2 / fileName
        savePathproj = output_dir_proj / fileName

        # Save Range Images and monodepth2 images
        cv2.imwrite(str(savePathmd2),image0)
        cv2.imwrite(str(savePathRange),image1)

        # Resize Images
        orig_shape0 = np.shape(image0)
        orig_shape1 = np.shape(image1)
        
        resizeImg0 = ( ((orig_shape0[0] // 8) + 1) * 8, ((orig_shape0[1] // 8) + 1) * 8 )
        resizeImg1 = ( ((orig_shape1[0] // 8) + 1) * 8, ((orig_shape1[1] // 8) + 1) * 8 )

        image0bckgrnd = np.zeros(resizeImg0, dtype=np.float32)
        image1bckgrnd = np.zeros(resizeImg1, dtype=np.float32)

        img0_resize = image0bckgrnd
        img1_resize = image1bckgrnd

        img0_resize[:orig_shape0[0], :orig_shape0[1]] = image0
        img1_resize[:orig_shape1[0], :orig_shape1[1]] = image1

        image0 = cv2.resize(image0, resizeImg0)
        image1 = cv2.resize(image1, resizeImg1)

        # Tranform and normalize images
        inp0 = frame2tensor(img0_resize, device)
        inp1 = frame2tensor(img1_resize, device)

        # Perform the matching.
        pred = {'image0': inp0, 'image1': inp1}
        with torch.no_grad():
            matching(pred)
            
        # dict_keys(['image0', 'image1', 'bs', 'hw0_i', 'hw1_i', 'hw0_c', 'hw1_c', 'hw0_f', 'hw1_f', 'conf_matrix', 'b_ids', 'i_ids', 'j_ids', 'gt_mask', 'm_bids', 'mkpts0_c', 'mkpts1_c', 'mconf', 'W', 'expec_f', 'mkpts0_f', 'mkpts1_f'])
        mkpts0 = pred['mkpts0_f'].cpu().detach().numpy()
        mkpts1 = pred['mkpts1_f'].cpu().detach().numpy()
        kpts0 = pred['mkpts0_c'].cpu().detach().numpy()
        kpts1 = pred['mkpts1_c'].cpu().detach().numpy()
        mconf = pred['mconf'].cpu().detach().numpy()

        # Visualize the matches.
        color = cm.jet(mconf)
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0)),
        ]

        make_matching_plot(
            img0_resize, img1_resize, kpts0, kpts1, mkpts0, mkpts1, color,
            text, savePathMatches, show_keypoints=True,
            fast_viz=True, opencv_display=True, opencv_title='Matches')
        
        # Get number of matches
        nmatches, _ = np.shape(mkpts0)

        if checkPC == True:
                keypts = get3dpointFromRangeimage(images1[i],kpts1,v_fov,h_fov,v_res,h_res, upsampleFactor, depth=-1)
                plot3dPoints(velodata[i],keypts)

        # Get 3d Coordinates from matches
        matches_3d = get3dpointFromRangeimage(images1[i],mkpts1,v_fov,h_fov,v_res,h_res, upsampleFactor, depth=True)
        matches_2d = mkpts0

        # Add keypoints to buffer
        if len(buffer_matches2d) < aggrMatches:
            buffer_matches2d.append(matches_2d)
            buffer_matches3d.append(matches_3d)
        else:
            buffer_matches2d.append(matches_2d)
            buffer_matches3d.append(matches_3d)
            buffer_matches2d.pop(0)
            buffer_matches3d.pop(0)

        # Join the keypoints to form numpy array of matches
        jmatches2d = np.vstack(buffer_matches2d)
        jmatches3d = np.vstack(buffer_matches3d)

        nmatchesagg,_ = np.shape(jmatches2d)

        # Append Matches to buffer

        if nmatchesagg >= 6:

            if checkPC == True:
                plot3dPoints(velodata[i],matches_3d)

            # Solve PnP problem
            _,Rvec,tvec,_ = cv2.solvePnPRansac(jmatches3d,jmatches2d,K_gt,np.zeros((1,4)))  #,flags=cv2.SOLVEPNP_ITERATIVE

            # Append solution to list
            T_rel_i = helper_func.rtvec_to_matrix(Rvec,tvec)
            T_rel.append(T_rel_i)

            # Reproject LiDAR to image
            R_pnp = helper_func.rtvec_to_matrix(Rvec, tvec)
            depthimageTF = helper_func.veloToDepthImage(K_gt,velodata[i],original_images[i],R_pnp,mode = 'z', trackPoints=False)
            
            helper_func.plotOverlay(rgb = original_images[i],lidar = depthimageTF, savePath=savePathproj, returnAxis = False)

        else:
            T_rel.append(None)

        numberMatches.append(nmatches)

    transformationError = []
    t_gt, r_gt = helper_func.matrix_to_rtvec(T_gt)
    r_gt = r_gt.ravel()
    t_gt = t_gt.ravel()
    for T in T_rel:
        if T is None:
            transformationError.append(None)
            continue
        else:
            t_loc, r_loc = helper_func.matrix_to_rtvec(T)
            trans_loss, rot_loss = transformationVecLoss(t_gt=t_gt,r_gt=r_gt, t=t_loc, r=r_loc)
            transformationError.append([trans_loss,rot_loss])

    avgTransLoss = 0
    avgRotLoss = 0
    tmpcounter = 0
    for i in range(len(transformationError)):
        if transformationError[i] is None:
            continue
        else:
            avgTransLoss += transformationError[i][0]
            avgRotLoss += transformationError[i][1]
            tmpcounter += 1
    avgTransLoss /= tmpcounter
    avgRotLoss /= tmpcounter

    # Define dict to extract results
    resDict = {'T_rel': T_rel,
               'Number of Matches': numberMatches,
               'Tranformation Error': transformationError}
    avgDict = {'Average Translational Error: ': avgTransLoss,
               'Average Rotational Error: ': avgRotLoss}
    
    # Define location to save results
    fout = output_dir_tf / 'Pose.txt'
    fo = open(fout, "w")

    # Safe Ground Truth
    fo.write('T_GroundTruth >>> \n\n')
    fo.write(str(T_gt) + '\n\n')

    # Get keys
    resKeys = [key for key in resDict.keys()]
    ndict = len(resDict[resKeys[0]])

    # Safe results in dict
    for key in avgDict:
        fo.write(str(key) + str(avgDict[key]))

    for i in range(ndict):
        fo.write(str(i) + ': \n')
        for j in range(len(resKeys)):
            fo.write(str(resKeys[j]) + '>>>' + '\n')
            fo.write(str( resDict[resKeys[j]][i] ) + '\n')
        fo.write('\n\n')

    fo.close()

    return True
