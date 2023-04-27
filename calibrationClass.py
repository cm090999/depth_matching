from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from SuperGluePretrainedNetwork.models.utils import frame2tensor, make_matching_plot

from utils import upsampleRangeImage, rtvec_to_matrix, veloToDepthImage, plotOverlay

from calibrationDataClass import Calilbration

class CameraLidarCalibration:
    def __init__(self,
                savePath,
                device = None):
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Define directory to save transformations
        self.output_dir_tf = Path().absolute() / savePath
        self.output_dir_tf.mkdir(exist_ok=True, parents=True)

        # Define directory to save matches
        self.output_dir_matches = self.output_dir_tf / 'Matches'
        self.output_dir_matches.mkdir(exist_ok=True, parents=True)

        # Define directory to save range images
        self.output_dir_lidar = self.output_dir_tf / 'LiDAR'
        self.output_dir_lidar.mkdir(exist_ok=True, parents=True)

        # Define directory to save monodepth2 images
        self.output_dir_camera = self.output_dir_tf / 'camera'
        self.output_dir_camera.mkdir(exist_ok=True, parents=True)

        # Define directory to save reprojected images
        self.output_dir_reproj = self.output_dir_tf / 'reprojection'
        self.output_dir_reproj.mkdir(exist_ok=True, parents=True)

        return
    
    def match_images(self,
                     Dataclass: Calilbration):

        # Store preparedImages as pred = {'image0': inp0, 'image1': inp1} list
        pred_ls = []

        # Loop over image pairs
        for i in range(Dataclass.nframes):

            # Clear plots
            plt.close('all')

            # Tranform and normalize images
            inp0 = frame2tensor(Dataclass.camera_images_mod[i], self.device)
            inp1 = frame2tensor(Dataclass.lidar_images_mod[i], self.device)
            pred = {'image0': inp0, 'image1': inp1}

            pred_ls.append(pred)

        return pred_ls
    

    def calculatePose(self, K, distortion = np.zeros((1,4))):
        for i in range(len(self.buffer_matches2d)):
            nmatchesagg,_ = np.shape(self.buffer_matches2d[i])

            if nmatchesagg >= 6:
                # Solve PnP problem
                _,Rvec,tvec,_ = cv2.solvePnPRansac(self.buffer_matches3d[i],self.buffer_matches2d[i],K,distortion)  #,flags=cv2.SOLVEPNP_ITERATIVE
                self.r_vec.append(Rvec)
                self.t_vec.append(tvec)
            else:
                self.r_vec.append(None)
                self.t_vec.append(None)
        return
    
    def reprojectedImage(self, image, velodata, K):
        for i in range(len(self.r_vec)):
            fileName = str(i).zfill(3) + '.png'
            savepth = self.output_dir_reproj / fileName
            # Reproject LiDAR to image
            R_pnp = rtvec_to_matrix(self.r_vec, self.t_vec)
            depthimageTF = veloToDepthImage(K,velodata[i],image[i],R_pnp,mode = 'z', trackPoints=False)
            
            plotOverlay(rgb = image[i],lidar = depthimageTF, savePath=savepth, returnAxis = False)

        return