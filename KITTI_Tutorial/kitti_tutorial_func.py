import numpy as np

def normalize_depth(val, min_v, max_v):
    """ 
    print 'nomalized depth value' 
    nomalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
    """
    return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)

def normalize_val(val, min_v, max_v):
    """ 
    print 'nomalized depth value' 
    nomalize values to 0-255 & close distance value has low value.
    """
    return (((val - min_v) / (max_v - min_v)) * 255).astype(np.uint8)

def in_h_range_points(m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), \
                          np.arctan2(n,m) < (-fov[0] * np.pi / 180))

def in_v_range_points(m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), \
                          np.arctan2(n,m) > (fov[0] * np.pi / 180))

def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """
    
    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points
    
    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
        return points[in_h_range_points(x, y, h_fov)]
    else:
        h_points = in_h_range_points(x, y, h_fov)
        v_points = in_v_range_points(dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]

def velo_points_2_pano(points, v_res, h_res, v_fov, h_fov, depth=False):

    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # project point cloud to 2D point map
    x_img = np.arctan2(-y, x) / (h_res * (np.pi / 180))
    y_img = -(np.arctan2(z, dist) / (v_res * (np.pi / 180)))

    """ filter points based on h,v FOV  """
    x_img = fov_setting(x_img, x, y, z, dist, h_fov, v_fov)
    y_img = fov_setting(y_img, x, y, z, dist, h_fov, v_fov)
    dist = fov_setting(dist, x, y, z, dist, h_fov, v_fov)

    x_size = int(np.ceil((h_fov[1] - h_fov[0]) / h_res))
    y_size = int(np.ceil((v_fov[1] - v_fov[0]) / v_res))
    
    # shift negative points to positive points (shift minimum value to 0)
    x_offset = h_fov[0] / h_res
    x_img = np.trunc(x_img - x_offset).astype(np.int32)
    y_offset = v_fov[1] / v_res
    y_fine_tune = 1
    y_img = np.trunc(y_img + y_offset + y_fine_tune).astype(np.int32)

    if depth == True:
        # nomalize distance value & convert to depth map
        dist = normalize_depth(dist, min_v=0, max_v=120)
    else:
        dist = normalize_val(dist, min_v=0, max_v=120)

    # array to img
    img = np.zeros([y_size + 1, x_size + 1], dtype=np.uint8)
    img[y_img, x_img] = dist
    
    return img

def velo_to_range(points, v_res: float, h_res: float, v_fov, h_fov, recursive = True, scaling = 0.95):
    """
    v_res,h_res in degrees
    """

    # Get coordinates and distances
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    dist = np.sqrt(x**2 + y**2 + z**2)

    # Get all vertical angles
    verticalAngles = np.arctan2(z, dist) / np.pi * 180 # Degrees

    # Get all horizontal angles
    horizontalAngles = np.arctan2(-y, x) / np.pi * 180 # Degrees

    # Filter based on FOV setting
    combined_condition = (verticalAngles < v_fov[0]) & (verticalAngles > v_fov[1]) & (horizontalAngles > h_fov[0]) & (horizontalAngles < h_fov[1])

    verticalAngles = verticalAngles[combined_condition]
    horizontalAngles = horizontalAngles[combined_condition]
    dist = dist[combined_condition]

    # Shift angles to all be positive
    verticalAnglesShifted = (verticalAngles - v_fov[0]) * -1
    horizontalAnglesShifted = horizontalAngles - h_fov[0]

    # Get maximum shifted angles
    vertMax = np.max(verticalAnglesShifted)
    horizMax = np.max(horizontalAnglesShifted)

    # Get Number of pixels in range image
    vertPix = int((np.absolute(v_fov[1] - v_fov[0]) / v_res))
    horiPix = int((np.absolute(h_fov[1] - h_fov[0]) / h_res))

    # Initialize Range image
    rangeImage = np.zeros((vertPix,horiPix),dtype=np.float32)

    # Get image coordinates of all points
    x_img_fl = np.round(horizontalAnglesShifted / np.absolute(h_fov[1] - h_fov[0]) * (horiPix - 1))
    y_img_fl = np.round(verticalAnglesShifted / np.absolute(v_fov[1] - v_fov[0]) * (vertPix - 1))
    x_img = x_img_fl.astype(int)
    y_img = y_img_fl.astype(int)

    # Fill values in range image
    rangeImage[y_img, x_img] = dist

    if recursive == True:
        scale = 1
        for i in range(6):
            # Define new scaling 
            scale *= scaling
            # Get empty values
            empty_pixels_mask = rangeImage == 0

            red_rangeImage = velo_to_range(points, v_res / scale, h_res / scale, v_fov, h_fov, recursive = False)
            redheight, redwidth = np.shape(red_rangeImage)
            red_x, red_y = np.meshgrid(np.arange(redwidth), np.arange(redheight))
            red_y = red_y.flatten().astype(float)
            red_x = red_x.flatten().astype(float)
            red_depth = red_rangeImage.flatten()

            # Rescale coordinates to match original image
            red_y = red_y / redheight * (vertPix)
            red_x = red_x / redwidth * (horiPix)

            # Round and convert to int
            red_y = np.round(red_y).astype(int)
            red_x = np.round(red_x).astype(int)

            # Consider only coordinates that are empty in original depth image
            condition = empty_pixels_mask[red_y,red_x]
            red_x = red_x[condition]
            red_y = red_y[condition]
            red_depth = red_depth[condition]

            rangeImage[red_y,red_x] = red_depth

    return rangeImage

def rangeImagefromImage(image, K_int, v_res: float, h_res: float):

    if np.ndim(image) == 2:
        image = np.expand_dims(image, axis = -1)

    # Get fov of monodepth image
    h, w, _ = np.shape(image)
    cx = K_int[0,2]
    cy = K_int[1,2]

    # fov_x_delta = 2 * np.arctan(w / (2 * K_int[0,0]))
    # fov_y_delta = 2 * np.arctan(h / (2 * K_int[1,1]))

    # # Consider center offset
    # delta_c_x = cx - w.astype(float) / 2
    # delta_c_y = cy - h.astype(float) / 2
    
    fov_x = ( -np.arctan((cx) / (K_int[0,0])), np.arctan((w - cx) / (K_int[0,0])))
    fov_y = ( -np.arctan((cy) / (K_int[1,1])), np.arctan((h - cy) / (K_int[1,1])))

    # Get range of angles 
    hor_angle_range = fov_x[1] - fov_x[0]
    ver_angle_range = fov_y[1] - fov_y[0]

    # Get Number of pixels in range image
    vertPix = int((np.absolute(ver_angle_range) / v_res) * 180 / np.pi)
    horiPix = int((np.absolute(hor_angle_range) / h_res) * 180 / np.pi)

    # Initialize Range image
    rangeImage = np.zeros((vertPix,horiPix),dtype=np.float32)

    # Create two lookup vectors for the angles of the pixels in the range image
    hor_angle_lookup = np.linspace(fov_x[0],fov_x[1],horiPix)
    ver_angle_lookup = np.linspace(fov_y[0],fov_y[1],vertPix)

    # Get angles of all pixel locations in image
    hor_angle_img = np.zeros((w,1))
    ver_angle_img = np.zeros((h,1))

    hor_px = np.linspace(0,w-1,w) - cx
    ver_px = np.linspace(0,h-1,h) - cy

    hor_angle_img[:,0] = np.arctan(hor_px / (K_int[0,0]))
    ver_angle_img[:,0] = np.arctan(ver_px / (K_int[1,1]))

    horLocs = np.abs(np.subtract.outer(hor_angle_lookup, hor_angle_img)).argmin(0)
    verLocs = np.abs(np.subtract.outer(ver_angle_lookup, ver_angle_img)).argmin(0)


    ###
    # for i in range(vertPix):
    #     for j in range(horiPix):
    #         rangeImage[i,j] = np.linalg.norm(image[verLocs[i],horLocs[j]])
    rangeImage = np.linalg.norm(image[verLocs[:, None], horLocs][:,:,0,:], axis=2) / 3

    ###

    return rangeImage

def rangeImagefromImage2(image: np.ndarray, K: np.ndarray, h_res: float, v_res: float):

    if np.ndim(image) == 2:
        image = np.expand_dims(image, axis = -1)

    # Get fov of image
    h, w, _ = np.shape(image)
    cx = K[0,2]
    cy = K[1,2]
    
    fov_x = ( -np.arctan((cx) / (K[0,0])), np.arctan((w - cx) / (K[0,0])) )
    fov_y = ( -np.arctan((cy) / (K[1,1])), np.arctan((h - cy) / (K[1,1])) )

    # Get range of angles 
    hor_angle_range = fov_x[1] - fov_x[0]
    ver_angle_range = fov_y[1] - fov_y[0]

    # Get Number of pixels in range image
    vertPix = int((np.absolute(ver_angle_range) / v_res) * 180 / np.pi)
    horiPix = int((np.absolute(hor_angle_range) / h_res) * 180 / np.pi)

    # Initialize Range image
    rangeImage = np.zeros((vertPix,horiPix),dtype=np.float32)

    # Range image to store corresponding original pixel locations
    rangeImage_corr = np.zeros((vertPix,horiPix,2), dtype=np.float32 )

    # Find correspondences of each pixel in rangeImage and the original image "image"
    ###
    # Create Lookup array for pixel <--> angle in rangeImage
    hor_lookup_range = np.linspace(fov_x[0], fov_x[1], horiPix)
    ver_lookup_range = np.linspace(fov_y[0], fov_y[1], vertPix)

    # Create 3d points with dummy distance one
    x_coords = np.tan(hor_lookup_range)
    y_coords = np.tan(ver_lookup_range)
    n_mesh, m_mesh = np.meshgrid(x_coords, y_coords)
    all_coords = np.column_stack([n_mesh.ravel(), m_mesh.ravel()])
    coords_3d = np.c_[all_coords, np.ones_like(all_coords)[:,0]]
    
    # Projected Coordinates (== image coordinates)
    coords_proj = np.matmul(K  ,np.transpose(coords_3d))[0:2,:]
    x_rangeCorr = np.reshape(coords_proj[0,:], np.shape(rangeImage))
    y_rangeCorr = np.reshape(coords_proj[1,:], np.shape(rangeImage))
    
    rangeImage_corr[:,:,0] = x_rangeCorr
    rangeImage_corr[:,:,1] = y_rangeCorr
    ###

    # Fill the values from image into rangeImage
    ###
    rangeImage = image[(np.floor(rangeImage_corr[:,:,1])).astype(int), (np.floor(rangeImage_corr[:,:,0])).astype(int)]
    ###

    return rangeImage

import cv2
testimg = cv2.imread('/home/colin/semesterThesis/conda_env/depth_matching/RES_SuperGlue/monodepth2/000.png')
testimg[50:100,120:150] = 0
K = np.array(  [[721.5377,   0.    , 609.5593],
                [  0.    , 721.5377, 172.854 ],
                [  0.    ,   0.    ,   1.    ]] )
v_res= 0.42
h_res= 0.35

test = rangeImagefromImage2(testimg, K, v_res, h_res)
cv2.imwrite('test.png',test)
cv2.imwrite('orig.png', testimg)
print("end")

