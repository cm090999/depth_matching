import os
import torch
from torchvision import transforms
import PIL.Image as pil
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt

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


def get3dpointFromRangeimage(rangeImage,kpts, v_fov, h_fov, upsamplefactor):
    # # Scale coordinates if image is upsampled
    # kpts /= upsamplefactor

    # Get shape of range image
    px_y, px_x = np.shape(rangeImage)

    # Get x, y image coordinates
    kptsx = kpts[:,0]
    kptsy = kpts[:,1]

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

    depim = rangeImage
    depim = rangeImage[kptsy.astype(int),kptsx.astype(int)]

    # Calculate z
    velopts[:,2] = depim * np.tan(vert_ang_kpts * np.pi / 180)

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

def transformPC(velo_pts: np.array, T: np.array):
    """
    Performs the coordinate transformation given a transformation matrix
    """
    pts, crd = np.shape(velo_pts)

    # Get LiDAR points in homogeneous shape
    velo_pts_hom = np.transpose(np.c_[velo_pts[:,0:3], np.ones(pts)])
    
    # Perform coordinate transformation and transpose the array to remove homogeneous coordinate
    velo_pts_hom_tf = np.matmul(T,velo_pts_hom)
    velo_pts_tf = np.transpose(velo_pts_hom_tf)[:,0:3]

    # Add back column vector of intensities if they were included in the input
    if crd >= 4:
        intens = velo_pts[:,-1]
        velo_pts_tf = np.c_[velo_pts_tf,intens]

    return velo_pts_tf

def veloToDepthImage(K: np.array, velo_pts: np.array, image: np.array, T = np.identity(4), mode='z', trackPoints = True):
    """
    A function that maps the points of the LiDAR point cloud to an image plane, given a calibration matrix and a transformation matrix. The image has 2 channels, [0] with depth information, [1] with intensity information.
    If no intensity chanel is given, the image has 1 channel (depth)

    mode: 'distance' or 'z'
    """
    # Perform the coordinate transformation on the point cloud
    velo_pts_tf = transformPC(velo_pts,T)

    # Add column to track the points original position
    sv = np.transpose(np.arange(0,np.shape(velo_pts)[0]))
    if trackPoints == True:
        velo_pts_tf = np.c_[velo_pts_tf,sv]

    # Remove all points with z <= 0
    velo_pts_tf = velo_pts_tf[velo_pts_tf[:, 2] > 1e-10]

    # Initialize empty depth image
    pts, crd = np.shape(velo_pts_tf)
    if np.ndim(image) == 3:
        img_hpx, img_wpx, img_c = np.shape(image)
    if np.ndim(image) == 2:
        img_hpx, img_wpx = np.shape(image)

    chann_dep = crd - 2
    # if crd >= 4:
    #     chann_dep = 2

    # Initialize Depth image with 0
    depth_image = np.zeros((img_hpx,img_wpx,chann_dep))
    depth_image[:,:,-1] = -1

    # get [u,v] from all point cloud points ([u,v] = K*Eye*PC)
    eyetmp = np.eye(3)
    eye = np.c_[eyetmp,np.zeros((3,1))]
    pc_img_coord_tmp = np.matmul(K,eye)
    pc_img_coord = np.matmul(pc_img_coord_tmp, np.transpose(np.c_[velo_pts_tf[:,0:3],np.zeros((pts,1))]) )
    pc_img_coord = np.transpose(pc_img_coord)
    pc_img_coord[:,0] /= pc_img_coord[:,2]
    pc_img_coord[:,1] /= pc_img_coord[:,2]
    pc_img_coord = pc_img_coord[:,0:2]

    # Get depth value of all points and attach to image coordinates array
    # attach intensity value if given
    depth_vec = np.zeros((pts,1))
    if mode == 'z':
        depth_vec[:,0] = velo_pts_tf[:,2]
    if mode == 'distance':
        depth_vec[:,0] = np.sqrt( velo_pts_tf[:,0]**2 + velo_pts_tf[:,1]**2 + velo_pts_tf[:,2]**2 )
    pc_img_coord = np.c_[pc_img_coord,depth_vec]
    if crd >= 4:
        pc_img_coord = np.c_[pc_img_coord,velo_pts_tf[:,3:]]

    # Remove points outside of image
    pc_img_coord = pc_img_coord[pc_img_coord[:,0] >= 0]
    pc_img_coord = pc_img_coord[pc_img_coord[:,1] >= 0]
    pc_img_coord = pc_img_coord[pc_img_coord[:,0] <= img_wpx]
    pc_img_coord = pc_img_coord[pc_img_coord[:,1] <= img_hpx]

    # Get pixel coordinates of all points, i.e. round coordinates
    pc_px_coord = (pc_img_coord[:,0:2]).astype(int)

    # Get indices of all duplicate coordinates
    pc_px_coord_flattened = pc_px_coord[:,0] + pc_px_coord[:,1] * img_wpx
    _, idcs = np.unique(pc_px_coord_flattened, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(pc_px_coord_flattened)), idcs)

    # Fill depth image matrix
    depth_image[pc_px_coord[:, 1], pc_px_coord[:, 0],:] = pc_img_coord[:, 2:]
    for i in range(len(duplicate_indices)):
        coord = pc_px_coord[duplicate_indices[i],:]
        if depth_image[coord[1],coord[0],0] > pc_img_coord[duplicate_indices[i],2]:
            depth_image[coord[1],coord[0],:] = pc_img_coord[duplicate_indices[i],2:]

    return depth_image

def rtvec_to_matrix(rvec=(0,0,0), tvec=(0,0,0)):
    "Convert rotation vector and translation vector to 4x4 matrix"
    rvec = np.asarray(rvec)
    tvec = np.asarray(tvec)

    T = np.eye(4)
    (R, jac) = cv2.Rodrigues(rvec)
    T[:3, :3] = R
    T[:3, 3] = tvec.squeeze()
    return T

def matrix_to_rtvec(matrix):
    "Convert 4x4 matrix to rotation vector and translation vector"
    (rvec, jac) = cv2.Rodrigues(matrix[:3, :3])
    tvec = matrix[:3, 3]
    return rvec, tvec

def depthTo3Dand2D(depthImage, K):

    depthImage = depthImage.astype(float)

    if np.ndim(depthImage) >= 3:
        depthImage = depthImage[:,:,0]

    depth_2d = np.transpose(np.nonzero(depthImage))
    # Switch x y in 2d coordinates
    depth_2d[:, 0], depth_2d[:, 1] = depth_2d[:, 1], depth_2d[:, 0].copy()
    
    # Initialize 3D depth points array and fill z coordinates
    depth_3d = np.zeros_like(depth_2d)
    depth_3d = np.c_[depth_3d, depthImage[depth_2d[:,1], depth_2d[:,0]]]

    ## Get x and y coordinates of 3d points by backprojecting pixel coordinates with depth value
    # x
    depth_3d[:,0] = (depth_2d[:,0] - K[0,2]) / K[0,0] * depth_3d[:,2]
    # y
    depth_3d[:,1] = (depth_2d[:,1] - K[1,2]) / K[1,1] * depth_3d[:,2]

    return depth_2d.astype(float), depth_3d

def depthmapToPts(depthImage,veloPts):
    """
    depthImage: (H,W,2), channel 0 = depth value, channel 1 = idx from lidar data
    """

    depthImage = depthImage.astype(float)

    depth_2d = np.transpose(np.nonzero(depthImage[:,:,1]+1))
    veloidcs = depthImage[depth_2d[:,0],depth_2d[:,1]][:,1].astype(int)
    depth_3d = veloPts[veloidcs,0:3]
    
    # Switch x y in 2d coordinates
    depth_2d[:, 0], depth_2d[:, 1] = depth_2d[:, 1], depth_2d[:, 0].copy()

    return depth_2d.astype(float), depth_3d
    

def plotOverlay(rgb, lidar, ax = None, color_map = 'jet', size_scale = 800, savePath = -1, returnAxis = True, **plt_kwargs):
    plt.close()

    if ax is None:
        fig, ax = plt.subplots()
    
        # Clear the figure
        fig.clf()

    if returnAxis == False:
        del ax
        ax = plt.gca()

    ax.clear()

    # Display Gray / RGB images
    if np.ndim(rgb) == 3:
        _,_,ctmp = np.shape(rgb)
        if ctmp == 3:
            # Display RGB
            ax.imshow(rgb)
        elif ctmp == 1:
            ax.imshow(rgb, cmap = 'gray')
    if np.ndim(rgb) == 2:
        ax.imshow(rgb, cmap = 'gray')

    # Remove intensity channel
    if np.ndim(lidar) == 3:
        lidar = lidar[:,:,0]

    # Normalize Depth Images
    depth1_scaled = (lidar[:,:]-np.min(lidar[:,:,]))/np.max(lidar[:,:])*255

    # Get the indices of non-zero elements in the depth map
    points1 = np.nonzero(lidar[:,:])

    # Plot depth points with scatter
    ax.scatter(points1[1], points1[0], s=depth1_scaled[points1]/size_scale, c=depth1_scaled[points1], cmap=color_map, alpha=0.99)

    if savePath != -1:
        print('Saving Image')
        plt.savefig(savePath, bbox_inches='tight', pad_inches=0, transparent = True, dpi = 250)
   
    return ax

def visualizeCalibration(rgb, lidar, K_int, rvec, tvec, color_map = 'jet', size_scale = 1600, savePath = -1, **plt_kwargs):
    # Start plotting
    ax = plt.gca()
    ax.clear()

    # Get image shape
    h, w, c = np.shape(rgb)

    # Get transformed lidar
    T = rtvec_to_matrix(rvec,tvec)
    lidar_tf = transformPC(lidar, T)

    # Mask points behind camera
    mask_behind = lidar_tf[:,2] > 0
    lidar = lidar[mask_behind,:]

    # Get corresponding depth value
    depth = lidar[:,2]

    # Check if npoints > 0
    pts, _ = np.shape(lidar)
    if pts == 0:
        ax.imshow(rgb)
        if savePath != -1:
            print('Saving Image')
            plt.savefig(savePath, bbox_inches='tight', pad_inches=0, transparent = True, dpi = 256)
        return

    # Project LiDAR points onto image plane
    lidar = lidar.astype(np.float32)
    projected_lidar,_ = cv2.projectPoints(lidar, rvec, tvec, K_int, distCoeffs=np.zeros((1,4)))

    mask = (projected_lidar[:, 0, 0] >= 0) & (projected_lidar[:, 0, 0] < w) & (projected_lidar[:, 0, 1] >= 0) & (projected_lidar[:, 0, 1] < h)
    projected_lidar_filtered = projected_lidar[mask, :]
    depth = depth[mask,]
    projected_lidar = projected_lidar_filtered[:,0,:]

    # Normalize depth
    npoints, _ = np.shape(projected_lidar)

    if npoints == 0:
        ax.imshow(rgb)
        if savePath != -1:
            print('Saving Image')
            plt.savefig(savePath, bbox_inches='tight', pad_inches=0, transparent = True, dpi = 256)
        return
        
    depth_norm = (depth - np.min(depth)) / np.max(depth) * 255

    # Lookup array
    lookupVec = np.arange(0,npoints)

    # Show image
    ax.imshow(rgb)

    # Plot points
    ax.scatter(projected_lidar[lookupVec,0], projected_lidar[lookupVec,1], s=depth_norm[lookupVec]/size_scale, c=depth_norm[lookupVec], cmap=color_map, alpha=0.99)

    if savePath != -1:
        print('Saving Image')
        plt.savefig(savePath, bbox_inches='tight', pad_inches=0, transparent = True, dpi = 256)

    return

def delta_rot_matrix(t_gt, r_gt, t, r):

    T_gt = rtvec_to_matrix(rvec=r_gt, tvec=t_gt)
    T = rtvec_to_matrix(rvec=r, tvec=t)

    deltaT = np.matmul(T_gt, np.linalg.inv(T))
    return deltaT

def rotationError(deltaT):

    r_angleAxis, _ = matrix_to_rtvec(deltaT)
    err = np.linalg.norm(r_angleAxis)

    return err

# Cost functions
##############

def transformationVecLoss(t_gt, r_gt, t, r):
    cost_tra = 0
    cost_rot = 0
    
    # t = t.ravel()
    # r = r.ravel()

    # t_gt_tmp = t_gt.ravel()
    # r_gt_tmp = r_gt.ravel()

    cost_tra = np.linalg.norm(t)

    rotmtrix = delta_rot_matrix(t_gt=t_gt,r_gt=r_gt,r=r,t=t)
    cost_rot = rotationError(rotmtrix)

    return cost_tra, cost_rot

###############