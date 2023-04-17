import os
import torch
from torchvision import transforms
import PIL.Image as pil
import cv2
import numpy as np
import open3d as o3d

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

    # Invert and normalize
    depth_image = ((depth_image - np.min(depth_image)) / np.max(depth_image) * (-1) + 1) * 255

    depth_image = cv2.equalizeHist(depth_image.astype(np.uint8))

    depth_image = (depth_image.astype(np.float32) / 255)

    return depth_image

def upsampleRangeImage(rangeImage,factor):
    heigth, width = np.shape(rangeImage)
    resized = cv2.resize(rangeImage,(factor*width,factor*heigth), interpolation=cv2.INTER_NEAREST) # interpolation=cv2.INTER_LINEAR
    return resized


def get3dpointFromRangeimage(rangeImage,kpts, v_fov, h_fov, v_res, h_res, upsamplefactor, depth = False):
    # # Scale coordinates if image is upsampled
    kpts /= upsamplefactor

    # Get shape of range image
    px_y, px_x = np.shape(rangeImage)

    # Get x, y image coordinates
    kptsx = kpts[:,0]
    kptsy = kpts[:,1]
    # kptsdepth = rangeImage[kptsy.astype(int),kptsx.astype(int)]

    # Get angles from range image
    v_fov_range = np.absolute(v_fov[1] - v_fov[0])
    h_fov_range = np.absolute(h_fov[1] - h_fov[0])
    v_fov_topleft = v_fov[0]
    h_fov_topleft = h_fov[0]
    v_fov_ascending = np.sign(v_fov[1] - v_fov[0])
    h_fov_ascending = np.sign(h_fov[1] - h_fov[0])

    # Get angles from keypoints
    vert_ang_kpts = v_fov_topleft + v_fov_ascending * v_fov_range / px_y * kptsy
    horz_ang_kpts = h_fov_topleft + h_fov_ascending * h_fov_range / px_x * kptsx


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
    if depth == -1:
        depim = rangeImage

    depim = rangeImage[kptsy.astype(int),kptsx.astype(int)]

    # Calculate z
    velopts[:,2] = depim * np.tan(vert_ang_kpts * np.pi / 180)

    ydivx = depim * np.tan(horz_ang_kpts * np.pi / 180)
    #    y = alpha * x

    # velopts[:,0] = np.sqrt((depim**2 - velopts[:,2]**2) / (1 + ydivx**2))
    # #   dist**2 = x**2 + y**2 + z**2 = x**2 + (alpha * x)**2 + z**2
    # #   dist**2 - z**2 = (1 + alpha**2) * x**2
    # #   np.sqrt((dist**2 - z**2) / (1 + alpha**2)) = x

    # velopts[:,1] = ydivx * velopts[:,0]

    # # Calculate x
    velopts[:,0] = depim * np.cos(horz_ang_kpts * np.pi / 180)

    # # Calculate y
    velopts[:,1] = -depim * np.sin(horz_ang_kpts * np.pi / 180)

    return velopts

def plot3dPoints(lidar,reprojection):

    _,dim = np.shape(lidar)
    if dim == 4:
        lidar = lidar[:,0:3]

    lidarpc = o3d.geometry.PointCloud()
    lidarpc.points = o3d.utility.Vector3dVector(lidar)
    lidarpc.colors = o3d.utility.Vector3dVector(np.c_[ np.ones((len(lidar),1)) , np.zeros((len(lidar),1)) , np.zeros((len(lidar),1)) ]) 


    reprojpc = o3d.geometry.PointCloud()
    reprojpc.points = o3d.utility.Vector3dVector(reprojection)
    reprojpc.colors = o3d.utility.Vector3dVector(np.c_[ np.zeros((len(reprojection),1)) , np.ones((len(reprojection),1)) , np.zeros((len(reprojection),1)) ])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.zeros((3))
    opt.show_coordinate_frame = True

    vis.add_geometry(lidarpc)
    vis.add_geometry(reprojpc)

    vis.run()

    vis.destroy_window()

    return

# Cost functions
##############

def transformationVecLoss(t_gt, r_gt, t, r, pnorm = 2):
    cost_tra = 0
    cost_rot = 0
    
    t = t.ravel()
    r = r.ravel()

    for i in range(3):
        cost_tra += (np.abs(t_gt[i]**pnorm - t[i]**pnorm))**(1/pnorm)
    
    for i in range(3):
        cost_rot += (np.abs(r_gt[i]**pnorm - r[i]**pnorm))**(1/pnorm)

    return cost_tra, cost_rot


def pointCloudLoss(pc, T_gt, T):
    return

###############