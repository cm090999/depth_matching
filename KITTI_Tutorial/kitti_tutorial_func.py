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
    # fov_x_delta = 2 * np.arctan(w / (2 * K_int[0,0]))
    # fov_y_delta = 2 * np.arctan(h / (2 * K_int[1,1]))

    # # Consider center offset
    # delta_c_x = K_int[0,2] - w.astype(float) / 2
    # delta_c_y = K_int[1,2] - h.astype(float) / 2
    
    fov_x = ( -np.arctan((w - K_int[0,2]) / (2 * K_int[0,0])), np.arctan((w + K_int[0,2]) / (2 * K_int[0,0])))
    fov_y = ( -np.arctan((h - K_int[1,2]) / (2 * K_int[1,1])), np.arctan((h + K_int[1,2]) / (2 * K_int[1,1])))

    # Get Number of pixels in range image
    vertPix = int((np.absolute(fov_y[1] - fov_y[0]) / v_res) * 180 / np.pi)
    horiPix = int((np.absolute(fov_x[1] - fov_x[0]) / h_res) * 180 / np.pi)

    # Initialize Range image
    rangeImage = np.zeros((vertPix,horiPix),dtype=np.float32)



    return rangeImage

# import cv2
# testimg = cv2.imread('/home/colin/semesterThesis/conda_env/depth_matching/RES_SuperGlue/monodepth2/000.png')
# K = np.array(  [[721.5377,   0.    , 609.5593],
#                 [  0.    , 721.5377, 172.854 ],
#                 [  0.    ,   0.    ,   1.    ]] )
# v_res= 0.42
# h_res= 0.35

# test = rangeImagefromImage(testimg, K, v_res, h_res)
# cv2.imsave('test.png',test)
# print("end")

