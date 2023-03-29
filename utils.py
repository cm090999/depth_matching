import os
import torch
from torchvision import transforms
import PIL.Image as pil
import cv2
import numpy as np

from monodepth2.utils import download_model_if_doesnt_exist
import monodepth2.networks as networks

def loadmonodepthModel(modelName, device = 'cpu'):

    download_model_if_doesnt_exist(modelName)
    encoder_path = os.path.join("models", modelName, "encoder.pth")
    depth_decoder_path = os.path.join("models", modelName, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()

    return encoder, depth_decoder, loaded_dict_enc

def convertImageToMonodepth2(image, loaded_dict_enc):

    # Get original width and height
    original_width, original_height = image.size

    # Get width and height required by monodepth2
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    # Resize image
    image0_resized = image.resize((feed_width, feed_height), pil.LANCZOS)

    # Transform to tensor
    torchImage = transforms.ToTensor()(image0_resized).unsqueeze(0)

    return torchImage, original_width, original_height

def evaluateMonodepth2(imagePT, encoder, depth_decoder, original_height, original_width):

    # Monodepth2 inference
    with torch.no_grad():
        features = encoder(imagePT)
        outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]

    # Resize image to original format
    disp_resized = torch.nn.functional.interpolate(disp,
        (original_height, original_width), mode="bilinear", align_corners=False)
    
    # Save images
    depth_image = disp_resized.squeeze().cpu().numpy()

    return depth_image

def upsampleRangeImage(rangeImage,factor):
    heigth, width = np.shape(rangeImage)
    resized = cv2.resize(rangeImage,(factor*width,factor*heigth), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized,(5,5),0)
    return blurred


def get3dpointFromRangeimage(rangeImage,kpts, v_fov, h_fov, v_res, h_res, upsamplefactor, depth = False):
    # Scale coordinates if image is upsampled
    kpts /= upsamplefactor
    kptsx = kpts[:,1]
    kptsy = kpts[:,0]

    # Initialize coordinate frame
    velopts = np.zeros((np.shape(kpts)[0],3))

    # Values from kitti_tutorial_func
    vmax = 120
    vmin = 0

    # Transform depth values to real distances
    if depth == True:
        depim = vmax - 1/255*rangeImage * (vmax - vmin)
    if depth == False:
        depim = vmin + 1/255*rangeImage * (vmax - vmin)

    # # 0 index horizontal angle
    # h_angle_0 = h_fov[0]
    # h_angle_end = h_fov[1]

    # x_offset = h_fov[0] / h_res
    # y_offset = v_fov[1] / v_res
    
    ## Get angles of all the kpts
    # add the offset back
    # x_uncent = kpts[:,0] + x_offset
    # y_uncent = kpts[:,1] - y_offset - 1
    # consider filtered points by FOV setting

    # 0 degrees at horizontal line, positive upwards
    # verticalAngle = - np.arctan(y_uncent[:] * v_res * np.pi / 180)
    # # 0 degrees facing forward on the kitti car, positive to left (positive z rotation), for reference see KITTI setup at https://www.cvlibs.net/datasets/kitti/setup.php
    # horizontalAngle = - np.arctan(x_uncent[:] * h_res * np.pi / 180.0)
    verticalAngle = v_fov[0] + (v_fov[1] - v_fov[1]) * kptsy[:]
    horizontalAngle = h_fov[0] + (h_fov[1] - h_fov[1]) * kptsx[:]

    # Calculate z
    velopts[:,2] = depim[kptsx[:].astype(int),kptsy[:].astype(int)] * np.sin(verticalAngle)

    # Calculate x
    velopts[:,0] = depim[kptsx[:].astype(int),kptsy[:].astype(int)] * np.sin(horizontalAngle)

    # Calculate y
    velopts[:,1] = depim[kptsx[:].astype(int),kptsy[:].astype(int)] * np.cos(horizontalAngle)

    return velopts