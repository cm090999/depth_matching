from pathlib import Path
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
from copy import deepcopy

from calibrationDataClass import Calilbration

from LoFTR.src.loftr import LoFTR, default_cfg

from SuperGluePretrainedNetwork.models.utils import make_matching_plot

from calibrationClass import CameraLidarCalibration

class LoFTR_Matching(CameraLidarCalibration):
    def __init__(self,
                 savePath,
                 device,
                 weight,
                 resize = -1
                 ):
        
        super().__init__(savePath=savePath, device=device)

        opt = {'resize': resize,
                'weight': weight}

        # Initialize LoFTR
        _default_cfg = deepcopy(default_cfg)
        # _default_cfg['coarse']['temp_bug_fix'] = True # set to False when using the old ckpt
        matcher = LoFTR(config=_default_cfg)
        matcher.load_state_dict(torch.load(opt['weight'])['state_dict'])
        self.matching = matcher.eval().to(device=device)
                
        return
    
    # Overload to adjust image size
    def match_images(self, 
                     Dataclass: Calilbration,
                     showPlot = True):
        
        resized_Dataclass = Dataclass

        for i in range(resized_Dataclass.nframes):
            resized_Dataclass.camera_images_mod[i] = self.resizeForResNet(resized_Dataclass.camera_images_mod[i])
            resized_Dataclass.lidar_images_mod[i] = self.resizeForResNet(resized_Dataclass.lidar_images_mod[i])
        
        pred_inp = super().match_images(resized_Dataclass)

        del resized_Dataclass

        for i in range(Dataclass.nframes):

            print('Work on Frame #' + str(i))
            p = pred_inp[i]

            with torch.no_grad():
                self.matching(p)
            
            # dict_keys(['image0', 'image1', 'bs', 'hw0_i', 'hw1_i', 'hw0_c', 'hw1_c', 'hw0_f', 'hw1_f', 'conf_matrix', 'b_ids', 'i_ids', 'j_ids', 'gt_mask', 'm_bids', 'mkpts0_c', 'mkpts1_c', 'mconf', 'W', 'expec_f', 'mkpts0_f', 'mkpts1_f'])
            mkpts0 = p['mkpts0_f'].cpu().detach().numpy()
            mkpts1 = p['mkpts1_f'].cpu().detach().numpy()
            kpts0 = p['mkpts0_c'].cpu().detach().numpy()
            kpts1 = p['mkpts1_c'].cpu().detach().numpy()
            mconf = p['mconf'].cpu().detach().numpy()

            # Save matching keypoints
            Dataclass.mkpts0[i] = mkpts0
            Dataclass.mkpts1[i] = mkpts1

            # Visualize the matches if enabled.
            if showPlot == True:
                # Make Plot
                color = cm.jet(mconf)
                text = [
                    'LoFTR',
                    'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                    'Matches: {}'.format(len(mkpts0)),
                ]

                savePathMatches = self.output_dir_matches / str(str(i).zfill(3) + '.png')

                make_matching_plot(
                    Dataclass.camera_images_mod[i], Dataclass.lidar_images_mod[i], kpts0, kpts1, mkpts0, mkpts1, color,
                    text, savePathMatches, show_keypoints=True,
                    fast_viz=True, opencv_display=True, opencv_title='Matches')
                
                plt.close()
                plt.clf()

        return
    
    def resizeForResNet(self,image):
        orig_shape = np.shape(image)
        resizeImg = ( ((orig_shape[0] // 8) + 1) * 8, ((orig_shape[1] // 8) + 1) * 8 )
        imagebckgrnd = np.zeros(resizeImg, dtype=np.float32)
        img_resize = imagebckgrnd
        img_resize[:orig_shape[0], :orig_shape[1]] = image
        return img_resize
