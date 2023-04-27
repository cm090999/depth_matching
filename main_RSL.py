import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from PIL import Image
import open3d as o3d

from KITTI_Tutorial.kitti_tutorial_func import velo_to_range, rangeImagefromImage

from utils import loadmonodepthModel, convertImageToMonodepth2, evaluateMonodepth2, matrix_to_rtvec

from calibrationDataClass import Calilbration, Calibration_Range


############
from match_LoFTR import LoFTR_Matching
from match_SuperGlue import SuperGlue_Matching
from datetime import datetime
############

if __name__ == "__main__":
    # Set if plots should be created (to debug)
    plt.ioff()
    nframes = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    upsampleFactor = 6
    smoothing = False
    checkPC = False
    aggregateMatches = 1

    # Path to data
    data_rsl = 'RSL_Data/rgb_images_and_pc/'

    # Images and LiDAR in grayscale
    images = []
    velodata = []
    for i in range(nframes):
        imageName = data_rsl + 'instance_' + str(i+1) + '.jpg'
        images.append(Image.open(imageName))

        # pcName = data_rsl + 'pc_instance_' + str(i+1) + '.pcd'
        # pcl = o3d.io.read_point_cloud(pcName)
        # pcl = np.asarray(pcl.points)

        pcl = np.asarray(o3d.io.read_point_cloud('RSL_Data/rgb_images_and_pc/pc_instance_orj.pcd').points)

        velodata.append(pcl)
    velo_pc_orj = np.asarray(o3d.io.read_point_cloud('RSL_Data/rgb_images_and_pc/pc_instance_orj.pcd').points)

    # # Calibration Data
    K_gt1 = np.array([[369.448, 0.0, 656.06],
                     [0.0, 369.096, 516.506],
                     [0.0, 0.0, 1.0]])
    K_gt2 = np.array([[377.132, 0.0, 704.905],
                     [0.0, 376.765, 523.327],
                     [0.0, 0.0, 1.0]])
    K_gt3 = np.array([[368.743, 0.0, 794.187],
                     [0.0, 368.65, 553.679],
                     [0.0, 0.0, 1.0]])
    K_gt = K_gt1
    # T_gt = kitti_raw.calib.T_cam3_velo
    # r_gt,t_gt = matrix_to_rtvec(T_gt)

    # Get data from calibration matrix to calculate FOV
    h,w, _= np.shape(images[0])
    fx, fy = K_gt[0,0], K_gt[1,1]

    # Get FOV of camera
    fov_x = 360 # np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    ## Convert the LiDAR point clouds to range images ##
    # Set visibility parameters 
    v_fov, h_fov = (90, -90), (-fov_x/2,fov_x/2) ### START AT TOP LEFT OF IMAGE
    v_res= 1.15# 0.42
    h_res= 0.7 # 0.35

    # Load monodepth2 model
    model_name = "mono_640x192"
    encoder, depth_decoder, loaded_dict_enc = loadmonodepthModel(model_name, device = device)

    # Converted images and LiDAR data
    rangeImages = []
    monodepthImages = []
    for i in range(nframes):
        # Create range image
        rangeImages.append(velo_to_range(velodata[i], v_res=v_res, h_res=h_res, v_fov=v_fov, h_fov=h_fov,recursive=True, scaling = 0.99))

        # Convert image to input to monodepth2
        image_pytorch, original_width, original_height = convertImageToMonodepth2(images[i], loaded_dict_enc)

        # Run monodepth2
        monodepthImages.append(evaluateMonodepth2(image_pytorch,encoder,depth_decoder,original_height,original_width))


    ########### Calibration ###########


    # Paths
    # Get current time
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
    resultPath = 'RESULT_RSL/' + timestamp


    ### SuperGlue ###
    # Initialize Data structure
    CalibrationSuperGlue = Calilbration(monodepthImages,
                                        rangeImages,
                                        v_fov=v_fov,
                                        h_fov=h_fov,
                                        rgbImages=images)
    CalibrationSuperGlue.getModifiedImages(normalize=True,
                                           upsamplefactor=upsampleFactor,
                                           smoothing=smoothing)
    
    # Initialize SuperGlue Matcher
    Superglue_matching = SuperGlue_Matching(savePath=resultPath + '/SuperGlue',
                                            device=device
                                            )
    # Perform Matching
    superGlueMatches = Superglue_matching.match_images(CalibrationSuperGlue,
                                                       showPlot=True)
    
    # Convert mkpts to original image coordinates and 3d coordinates
    CalibrationSuperGlue.get2d3dpts_from_mkpts()
    # Get aggregated matches
    CalibrationSuperGlue.getAggreg_pts(n_agg=aggregateMatches)
    CalibrationSuperGlue.getNumMatches()

    # Solve Pnp
    CalibrationSuperGlue.solve_pnp_agg(K_gt)

    # Reproject Images
    CalibrationSuperGlue.reprojectLidar(K_gt,velodata,Superglue_matching.output_dir_reproj)

    # CalibrationSuperGlue.calculateError(r_gt=r_gt,t_gt=t_gt)

    # CalibrationSuperGlue.writeTo_TXT(Superglue_matching.output_dir_tf, r_gt, t_gt)

    CalibrationSuperGlue.saveImages(Superglue_matching.output_dir_camera, Superglue_matching.output_dir_lidar)
    ### SuperGlue ###


    ### LoFTR ###
    CalibrationLoFTR = Calilbration(monodepthImages,
                                    rangeImages,
                                    v_fov=v_fov,
                                    h_fov=h_fov,
                                    rgbImages=images)
    CalibrationLoFTR.getModifiedImages( normalize=True,
                                        upsamplefactor=upsampleFactor,
                                        smoothing=smoothing)
    loftr_matching = LoFTR_Matching(savePath=resultPath + '/LoFTR',
                                    device=device,
                                    weight='LoFTR/weights/outdoor_ds.ckpt',
                                    resize=-1
                                    )
     
    loftrMatches = loftr_matching.match_images( CalibrationLoFTR,
                                                showPlot=True)
    
    # Convert mkpts to original image coordinates and 3d coordinates
    CalibrationLoFTR.get2d3dpts_from_mkpts()
    # Get aggregated matches
    CalibrationLoFTR.getAggreg_pts(n_agg=aggregateMatches)
    CalibrationLoFTR.getNumMatches()

    # Solve Pnp
    CalibrationLoFTR.solve_pnp_agg(K_gt)

    # Reproject Images
    CalibrationLoFTR.reprojectLidar(K_gt,velodata,loftr_matching.output_dir_reproj)

    # CalibrationLoFTR.calculateError(r_gt=r_gt,t_gt=t_gt)

    # CalibrationLoFTR.writeTo_TXT(loftr_matching.output_dir_tf, r_gt, t_gt)

    CalibrationLoFTR.saveImages(loftr_matching.output_dir_camera, loftr_matching.output_dir_lidar)
    ### LoFTR ###
