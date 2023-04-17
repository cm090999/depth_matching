from pathlib import Path
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from ST_depth_correspondence import helper_func

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot

from utils import upsampleRangeImage, get3dpointFromRangeimage, plot3dPoints, transformationVecLoss

def matchSuperglue(images0, images1, original_images, velodata, v_fov, h_fov, v_res, h_res, T_gt, K_gt, config, savePath, device = 'cpu', smoothing = False, upsampleFactor = 1, checkPC = False):

    nframes = len(images0)

    matching = Matching(config).eval().to(device)

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

        # Tranform and normalize images
        inp0 = frame2tensor(image0, device)
        inp1 = frame2tensor(image1, device)

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
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            text, savePathMatches, show_keypoints=True,
            fast_viz=True, opencv_display=True, opencv_title='Matches')
        
        # Get number of matches
        nmatches, _ = np.shape(mkpts0)

        if checkPC == True:
                keypts = get3dpointFromRangeimage(images1[i],kpts1,v_fov,h_fov,v_res,h_res, upsampleFactor, depth=-1)
                plot3dPoints(velodata[i],keypts)

        if nmatches >= 6:

        # Get 3d Coordinates from matches
            matches_3d = get3dpointFromRangeimage(images1[i],mkpts1,v_fov,h_fov,v_res,h_res, upsampleFactor, depth=True)
            matches_2d = mkpts0

            if checkPC == True:
                plot3dPoints(velodata[i],matches_3d)

            # Solve PnP problem
            _,Rvec,tvec,_ = cv2.solvePnPRansac(matches_3d,matches_2d,K_gt,np.zeros((1,4)))  #,flags=cv2.SOLVEPNP_ITERATIVE

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
        if type(T) == None:
            transformationError.appen(None)
            continue
        else:
            t_loc, r_loc = helper_func.matrix_to_rtvec(T)
            trans_loss, rot_loss = transformationVecLoss(t_gt=t_gt,r_gt=r_gt, t=t_loc, r=r_loc)
            transformationError.append([trans_loss,rot_loss])

    # Define dict to extract results
    resDict = {'T_rel': T_rel,
               'Number of Matches': numberMatches}
    
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
    for i in range(ndict):
        fo.write(str(i) + ': \n')
        for j in range(len(resKeys)):
            fo.write(str(resKeys[j]) + '>>>' + '\n')
            fo.write(str( resDict[resKeys[j]][i] ) + '\n')
        fo.write('\n\n')

    fo.close()

    return True
