from pathlib import Path
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import make_matching_plot

from calibrationDataClass import Calilbration
from calibrationClass import CameraLidarCalibration

class SuperGlue_Matching(CameraLidarCalibration):
    def __init__(self,
                 savePath,
                 device,
                 nms_radius = 4,
                 sinkhorn_iterations = 50,
                 match_threshold = 0.4,
                 keypoint_threshold = 0.005,
                 max_keypoints = 1024,
                 superglue = 'outdoor'):
        
        super().__init__(savePath=savePath, device=device)

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

        # Initialize SuperGlue matcher
        self.matching = Matching(config).eval().to(self.device)

        
        return
    
    # Overload to adjust image size
    def match_images(self,
                     Dataclass: Calilbration, 
                     showPlot=True):
        
        pred_inp = super().match_images(Dataclass)

        for i in range(Dataclass.nframes):
            
            print('Work on Frame #' + str(i))
            p = pred_inp[i]

            with torch.no_grad():
                pred = self.matching(p)
            
            pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']

            # Keep the matching keypoints.
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf = conf[valid]

            # Save matching keypoints
            Dataclass.mkpts0[i] = mkpts0
            Dataclass.mkpts1[i] = mkpts1

            # Visualize the matches if enabled.
            if showPlot == True:
                color = cm.jet(mconf)
                text = [
                    'SuperGlue',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]

                # Make Plot
                # Display extra parameter info.
                k_thresh = self.matching.superpoint.config['keypoint_threshold']
                m_thresh = self.matching.superglue.config['match_threshold']

                savePathMatches = self.output_dir_matches / str(str(i).zfill(3) + '.png')

                make_matching_plot(
                    Dataclass.camera_images_mod[i], Dataclass.lidar_images_mod[i], kpts0, kpts1, mkpts0, mkpts1, color,
                    text, savePathMatches, show_keypoints=True,
                    fast_viz=True, opencv_display=True, opencv_title='Matches')
        return




