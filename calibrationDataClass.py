import numpy as np
import cv2

from utils import upsampleRangeImage, get3dpointFromRangeimage, rtvec_to_matrix, veloToDepthImage, plotOverlay, transformationVecLoss

class Calilbration():
    def __init__(self, 
                 camera_images: list, 
                 lidar_images: list,
                 v_fov: tuple,
                 h_fov: tuple):

        ## assigned here
        self.camera_images = camera_images
        self.lidar_images = lidar_images
        self.nframes = len(self.camera_images)
        self.v_fov = v_fov
        self.h_fov = h_fov

        ## assigned here, modified in getModifiedImages
        self.normalize = True
        self.upsamplefactor = 1
        self.smoothing = False

        ## initialized here, modified in getAggreg_pts
        self.n_agg = 1

        ## assigned in getModifiedImages
        # Modified images
        self.camera_images_mod = [None] * self.nframes
        self.lidar_images_mod = [None] * self.nframes

        ## assigned in match_images
        # Matching keypoints in modified image coordinates
        self.mkpts0 = [None] * self.nframes
        self.mkpts1 = [None] * self.nframes

        # assigned in get2d3dpts_from_mkpts
        # Matching keypoints in 3d, corresponding to image coordinates in original images
        self.matches2d = [None] * self.nframes
        self.matches3d = [None] * self.nframes

        # assigned in getAggreg_pts
        # Aggregated Matching keypoints in 3d, corresponding to image coordinates in original images
        self.matches2d_agg = [None] * self.nframes
        self.matches3d_agg = [None] * self.nframes

        # Initialize Empty lists to store results
        self.numberMatches = [None] * self.nframes
        self.numberAggMatches = [None] * self.nframes
        self.r_vec = [None] * self.nframes
        self.t_vec = [None] * self.nframes

        # deviation from ground truth
        self.r_error = [None] * self.nframes
        self.t_error = [None] * self.nframes
        self.r_error_avg = None
        self.t_error_avg = None

        return
    
    def getModifiedImages(self,
                          normalize = True,
                          upsamplefactor = 1,
                          smoothing = False):
        
        self.normalize = normalize
        self.upsamplefactor = upsamplefactor
        self.smoothing = smoothing
        
        for i in range(self.nframes):
            
            if normalize == True:
                self.camera_images_mod[i] = (((self.camera_images[i] - np.min(self.camera_images[i])) / np.max(self.camera_images[i])) * 255).astype(np.float32)
                self.lidar_images_mod[i] = (((self.lidar_images[i] - np.min(self.lidar_images[i])) / np.max(self.lidar_images[i])) * 255).astype(np.float32)

            # Resize range image
            self.lidar_images_mod[i] = upsampleRangeImage(self.lidar_images_mod[i], upsamplefactor)

            if smoothing == True:
                    self.lidar_images_mod[i] = cv2.GaussianBlur(self.lidar_images_mod[i],(5,5),0)

        return
    
    def get2d3dpts_from_mkpts(self):
        for i in range(self.nframes):
            # Get 3d Coordinates from matches
            self.matches3d[i] = get3dpointFromRangeimage(self.lidar_images_mod[i],self.mkpts1[i],self.v_fov,self.h_fov,0,0, self.upsamplefactor, depth=True)
            self.matches2d[i] = self.mkpts0[i]

        return
    
    def getAggreg_pts(self,n_agg):
        self.n_agg = n_agg

        for i in range(self.n_agg-1,self.nframes):
            d2 = [ ele for ele in self.matches2d[i-(self.n_agg-1):i+1] if ele is not None ]
            d3 = [ ele for ele in self.matches3d[i-(self.n_agg-1):i+1] if ele is not None ]

            if len(d2) == 0:
                continue

            self.matches2d_agg[i] = np.vstack(d2)
            self.matches3d_agg[i] = np.vstack(d3)

            del d2,d3

        return
    
    def getNumMatches(self):

        for i in range(self.nframes):
            if self.matches2d[i] is None:
                self.numberMatches[i] = 0
            else:
                self.numberMatches[i], _ = np.shape(self.matches2d[i])
            if self.matches2d_agg[i] is None:
                self.numberAggMatches[i] = 0
            else:
                self.numberAggMatches[i], _ = np.shape(self.matches2d_agg[i])

        return
    
    def solve_pnp_agg(self, K_int, dist = np.zeros((1,4))):
        for i in range(self.nframes):
            if self.numberAggMatches[i] >= 6:

                # Solve PnP problem
                _,self.r_vec[i],self.t_vec[i],_ = cv2.solvePnPRansac(self.matches3d_agg[i],self.matches2d_agg[i],K_int,dist)  #,flags=cv2.SOLVEPNP_ITERATIVE

        return
    
    def reprojectLidar(self, K_int: np.ndarray, velodata: list, savePathproj):
                    
        for i in range(self.nframes):
            if self.numberAggMatches[i] >= 6:

                fileName = str(i).zfill(3) + '.png'
                savepth = savePathproj / fileName
                    
                # Reproject LiDAR to image
                R_pnp = rtvec_to_matrix(self.r_vec[i], self.t_vec[i])
                depthimageTF = veloToDepthImage(K_int,velodata[i],self.camera_images[i],R_pnp,mode = 'z', trackPoints=False)
                plotOverlay(rgb = self.camera_images[i], lidar = depthimageTF, savePath=savepth, returnAxis = False)

        return
    
    def calculateError(self,r_gt,t_gt, pnorm = 2):
        
        # Get error of each sample
        for i in range(self.nframes):
            if self.t_vec[i] is None:
                continue
            cost_tra, cost_rot = transformationVecLoss(t_gt, r_gt, self.t_vec[i], self.r_vec[i], pnorm = pnorm)
            self.r_error[i] = cost_rot
            self.t_error[i] = cost_tra

        # Get average error
        ntmp = 0
        self.r_error_avg = 0
        self.t_error_avg = 0
        for i in range(self.nframes):
            if self.r_error[i] is None:
                continue

            self.r_error_avg += self.r_error[i]
            self.t_error_avg += self.t_error[i]
            ntmp += 1
        
        self.r_error_avg /= ntmp
        self.t_error_avg /= ntmp

        return
    
    def writeTo_TXT(self,txtpath):

        # Define dict to extract results
        resDict = {'Number of Matches: ': self.numberMatches,
                'Number of aggregated Matches: ': self.numberAggMatches,
                'Tranformation Error Rotation: ': self.r_error,
                'Tranformation Error Translation: ': self.t_error}
        
        avgDict = {'Average Translational Error: ': self.t_error_avg,
                'Average Rotational Error: ': self.r_error_avg}
        
        # Define location to save results
        fout = txtpath / 'Pose.txt'
        fo = open(fout, "w")

        # Get keys
        resKeys = [key for key in resDict.keys()]
        ndict = len(resDict[resKeys[0]])

        # Safe results in dict
        for key in avgDict:
            fo.write(str(key) + str(avgDict[key]))
            fo.write('\n')

        for i in range(ndict):
            fo.write(str(i) + ': \n')
            for j in range(len(resKeys)):
                fo.write(str(resKeys[j]) + '>>>' + '\n')
                fo.write(str( resDict[resKeys[j]][i] ) + '\n')
            fo.write('\n\n')

        fo.close()

        return
    
    def saveImages(self, camPath, lidarPath):

        for i in range(self.nframes):
            fileName = str(str(i).zfill(3) + '.png')
            savepth_cam = camPath / fileName
            savepth_lid = lidarPath / fileName

            cv2.imwrite(savepth_cam, self.camera_images[i])
            cv2.imwrite(savepth_lid, self.lidar_images[i])

        return
