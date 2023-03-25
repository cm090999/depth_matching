import os
import pykitti as pk
import matplotlib.pyplot as plt
import torch
import PIL.Image as pil
from torchvision import transforms
import numpy as np

from monodepth2.utils import download_model_if_doesnt_exist
import monodepth2.networks as networks

from ST_depth_correspondence import helper_func
from KITTI_Tutorial.kitti_tutorial_func import velo_points_2_pano

if __name__ == "__main__":
    # Set if plots should be created (to debug)
    debug = True
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
    v_fov, h_fov = (-25, 0), (-180,180)
    v_res=0.42
    h_res=0.35

    range0 = velo_points_2_pano(velo0, v_res, h_res, v_fov, h_fov, depth=False)
    range1 = velo_points_2_pano(velo1, v_res, h_res, v_fov, h_fov, depth=False)

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
    original_width, original_height = image0.size

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    image0_resized = image0.resize((feed_width, feed_height), pil.LANCZOS)

    image0_pytorch = transforms.ToTensor()(image0_resized).unsqueeze(0)

    # Monodepth2 inference
    with torch.no_grad():
        features = encoder(image0_pytorch)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]

    # Resize image to original format
    disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)

    if debug == True:
        # Plot colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
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

    print('FINISHED')


