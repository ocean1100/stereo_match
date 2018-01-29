# @Author: Hao G <hao>
# @Date:   2018-01-02T13:38:10+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-29T14:03:50+00:00


from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
import cv2


def write_ply(fn, verts, colors):
    """Ply write."""
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    asd = bytes(ply_header % dict(vert_num=len(verts)), 'utf-8')
    with open(fn, 'wb') as f:
        f.write(asd)
        np.savetxt(f, verts, '%f %f %f %d %d %d')
    # with open(fn, 'r') as original:
    #         data = original.read()
    # with open('out_put_name.ply', 'w') as modified:
    #         modified.write(ply_header % dict(vert_num=len(verts)) + data)
    # modified.close()


def map2Dto3D(pt2D, ptd, cx, cy, fx, fy, threshold):
    """
    3d pts and depth point calculation.

    Arguments:
    pt2D N by N 2D points;
    ptd is the corresponding depth map;
    cx, cy, fx, fy are the intrinsic parameters.
    threshold is the value for the boundary
    """
    pt2D_row, pt2D_column = np.where(pt2D > threshold)
    ptd = ptd[pt2D > threshold]
    u3d = (pt2D_column - cx) * ptd / fx
    v3d = (pt2D_row - cy) * ptd / fy
    u3d = -np.reshape(u3d, pt2D.shape)
    v3d = np.reshape(v3d, pt2D.shape)
    ptd = np.reshape(ptd, pt2D.shape)
    # ptd = ptd[:, ::-1]
    pt3 = np.stack([u3d, v3d, ptd], axis=2)
    return pt3


def axis_equal_3d(ax):
    """Function that makes 3D axis equal

    Parameters
    ----------
    ax : matplotlib 3D axes

    """
    extents = np.array([
        getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']
    )
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_camera(camera_pose, label, ax, s=1):
    """Function that plots camera poses

    Parameters
    ----------
    camera_pose : pose of the camera with respect to world coordinates
    label : text to display over the center of the camera
    ax : matplotlib 3D axes
    s : length of the axis arrows

    """
    ax.scatter(*camera_pose[:3, 3])
    ax.quiver(*camera_pose[:3, 3], *camera_pose[:3, 0], color='red', length=s)
    ax.quiver(*camera_pose[:3, 3], *camera_pose[:3, 1], color='green', length=s)
    ax.quiver(*camera_pose[:3, 3], *camera_pose[:3, 2], color='blue', length=s)
    ax.text(*(camera_pose[:3, 3] + np.array([0.1, 0.1, 0.1])), label)


def show_image_pair(img_rect1, img_rect2, window_name, scale=1):
    """Function that shows a pair of images in the same window with horizontal lines to show that the y of the
    two images is supposed to be the same

    Parameters
    ----------
    img_rect1 : rectified left image
    img_rect2 : rectified right image
    window_name : name of the window
    scale : if set to a number greater than 1, it resizes the image to (width/scale, height/scale)
    """

    if scale < 1:
        raise ValueError('scale must be greather or equal than 1')

    # here we are resizing the images down for visualization purposes
    img_rect1_res = cv2.resize(img_rect1, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
    img_rect2_res = cv2.resize(img_rect2, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)

    # computes the total size of the two images
    total_size = (max(img_rect1_res.shape[0], img_rect2_res.shape[0]),
                  img_rect1_res.shape[1] + img_rect2_res.shape[1], 3)
    img = np.zeros(total_size, dtype=np.uint8)
    # fill a new bigger image with the two small images
    img[:img_rect1_res.shape[0], :img_rect1_res.shape[1]] = img_rect1_res
    img[:img_rect2_res.shape[0], img_rect1_res.shape[1]:] = img_rect2_res

    # draw horizontal lines every 25 px accross the side by side image
    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

    # and we show
    cv2.imshow(window_name, img)


def show_disparity(disparity, filtered=None, scale=1):
    """Function that shows disparity images

    Parameters
    ----------
    disparity : int16 raw disparity image as returned by stereoBM (or stereoSGBM).compute
    filtered : int16 filtered disparity image returned by wls_filter
    scale : if set to a number greater than 1, it resizes the image to (width/scale, height/scale)
    """

    if scale < 1:
        raise ValueError('scale must be greather or equal than 1')

    disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)
    displ_res = cv2.resize(disparity, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("raw disparity", displ_res)

    if filtered is not None:
        filtered = cv2.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filtered = np.uint8(filtered)
        filtered_img_res = cv2.resize(filtered, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("filtered disparity", filtered_img_res)


def plot_transforms(transforms):
    """Function that plots a set of camera poses

    Parameters
    ----------
    transforms : set of camera poses to plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pose in transforms:
        plot_camera(pose, '', ax, s=0.01)
    axis_equal_3d(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()



def path_join(*args):
    """Do the os path join thing."""
    return os.path.join(*args)


def dict_wrap(
        img_dir, timestamps, intrinsics, extrinsics,
        npz_dir, highresolution_suffix='-1.000',
        split_symbol='-', file_format='.jpeg'):
    """Wrap the data into a npz file.

    Parameters:
    img_dir: the image directory
    timestamps: the time stamp array
    intrinsics: the intrinsic matrix for each time stamp
    extrinsics: the extrinsic matrix for each time stamp

    Returns:
    Nothing but a npz file.
    """

    image_name = str(timestamp) + highresolution_suffix + file_format
    image_full_path = path_join(img_dir, image_name)
    assert os.path.isfile(image_full_path), (
        'File {} does not exist!'.format(image_full_path)
    )

    frame_id = (np.abs(timestamps - timestamp)).argmin()
    return [
        {
            'timestamp': timestamp,
            'image': image,
            'frame_id': frame_id,
            'extrinsic': extrinsics,
            'intrinsic': intrinsics
        }]


def plot_camera(camera_pose, label, ax, s=1):
    """Function that plots camera poses

    Parameters
    ----------
    camera_pose : pose of the camera with respect to world coordinates
    label : text to display over the center of the camera
    ax : matplotlib 3D axes
    s : length of the axis arrows

    """
    ax.scatter(*camera_pose[:3, 3])
    ax.quiver(*camera_pose[:3, 3], *camera_pose[:3, 0], color='red', length=s)
    ax.quiver(*camera_pose[:3, 3], *camera_pose[:3, 1], color='green', length=s)
    ax.quiver(*camera_pose[:3, 3], *camera_pose[:3, 2], color='blue', length=s)
    ax.text(*(camera_pose[:3, 3] + np.array([0.1, 0.1, 0.1])), label)


def extrinsic_matrix_get(transforms, mode):
    """Calculate the extrinsic matrix.

    Parameters:
    transforms: the "ARKit" transform matrix.

    Return:
    the extrinsic matrix in a list.
    """
    # Transformation from arkit camera to opencv camera. Opencv camera has the z pointing forward. This transformation
    # depends on the modality the scan has been made. If the phone is in portrait mode, then the x points down, the
    # y points right and the z points backwards. If the phone is in landscape mode, then the coordinate frame is
    # the same as ARKit world (x pointing right, y pointing up, z pointing backwards).
    # Opencv camera, however, has x pointing right, y pointing down and z pointing forward.
    if mode == 'P':
        arkitc_matrix_opencv = np.array([[0, 1, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 0, 0, 1]])

    elif mode == 'LR':
        arkitc_matrix_opencv = np.array([[-1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 0, 0, 1]])

    else:
        arkitc_matrix_opencv = np.array([[1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, -1, 0],
                                         [0, 0, 0, 1]])
    # Transformation from world to arkit world. ARKit world has the y axis pointing up, the x axis pointing towards
    # right and the z axis pointing backwards. The world has the x axis pointing right, the y axis pointing forward
    # and the z axis pointing up.
    world_matrix_arkitw = np.array([[1, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 1]])
    # here we get transforms from the world to opencv. This way we are going to have the opencv-style orientation
    # of the camera with respect to the world. To do so, the opencv transformation has to go on the right.
    tmp = world_matrix_arkitw.dot(transforms)
    # extrinsic_matrix.append(tmp.dot(arkitc_matrix_opencv))

    return tmp.dot(arkitc_matrix_opencv)


def json_read(json_file):
    """Reads a json file and returns a json structure"""
    with open(json_file, 'r') as fp:
        data_json = json.load(fp)
    return data_json



def check_epipoles(intrinsics_l, intrinsics_r, camera_l_pose, camera_r_pose, vis_l, vis_r):
    """Function that checks whether the epipoles are inside the images.

    Parameters
    ----------
    intrinsics_l : intrinsics first camera
    intrinsics_r : intrinsics second camera
    camera_l_pose : 4x4, left camera pose
    camera_r_pose : 4x4, right camera pose
    vis_l : first image
    vis_r : second image

    Returns
    -------
    valid : True if the epipoles are not in the image, false otherwise

    """
    # projective matrices left and right. We need them to compute the epipoles
    proj_left = intrinsics_l.dot(np.linalg.inv(camera_l_pose)[:3, :])
    proj_right = intrinsics_r.dot(np.linalg.inv(camera_r_pose)[:3, :])

    # epipole left and right: camera centers projected in the opposite image plane
    e_left = proj_left.dot(camera_r_pose[:, 3])
    e_right = proj_right.dot(camera_l_pose[:, 3])

    # epipoles are in homogeneous coordinates, so we divide by the third coordinate to obtain coordinates on the
    # image plane
    e_left = e_left / e_left[2]
    e_right = e_right / e_right[2]

    valid = True
    # if epipoles are in the image, rectification cannot be performed (at least using fusiello or opencv methods)
    if ((0 < e_left[0] < vis_l.shape[1]) and (0 < e_left[1] < vis_l.shape[0])) or (
            (0 < e_right[0] < vis_r.shape[1]) and (0 < e_right[1] < vis_r.shape[0])):
        valid = False
    return valid


def rectify_opencv(camera_l_pose, camera_r_pose, intrinsics_l, intrinsics_r, vis_l, vis_r, dist_coefficients_l=None,
                   dist_coefficients_r=None):
    """Function that rectifies two images using opencv stereoRectify method.

    Parameters
    ----------
    camera_l_pose : 4x4, left camera pose
    camera_r_pose : 4x4, right camera pose
    intrinsics_l : intrinsic parameters for left camera
    intrinsics_r : intrinsic parameters for right camera
    vis_l : left image
    vis_r : right image
    dist_coefficients_l : a 5-dimensional numpy array of distortion coefficients for the first camera
    dist_coefficients_r : a 5-dimensional numpy array of distortion coefficients for the second camera
    scale : multiplier that determines the size of the rectified image

    Returns
    -------
    img_rect1 : left image rectified
    img_rect2 : right image rectified
    Q : projection matrix to re-project points in 3D
    """
    # if not check_epipoles(intrinsics_l, intrinsics_r, camera_l_pose, camera_r_pose, vis_l, vis_r):
    #     raise RuntimeError('One of the epipoles is in the image, rectification cannot be performed')

    if dist_coefficients_l is None:
        dist_coefficients_l = np.array([0.0, 0, 0, 0, 0])
    if dist_coefficients_r is None:
        dist_coefficients_r = np.array([0.0, 0, 0, 0, 0])

    rotation = camera_r_pose[:3, :3].transpose().dot(camera_l_pose[:3, :3])
    translation = camera_r_pose[:3, :3].transpose().dot(camera_l_pose[:3, 3] - camera_r_pose[:3, 3])
    # or, we could use the inverse of the camera r pose multiplied by the left camera pose directly and then
    # take the rotation and translation from it
    # r_transform_l = np.linalg.inv(camera_r_pose).dot(camera_l_pose)

    h, w = vis_l.shape[:2]
    # it looks like initUndistortRectifyMap wants (width, height)
    size = (w, h)
    # size of the rectified image
    # new_size = (round(size[0] * scale), round(size[1] * scale))
    new_size = size

    # stereoRectify returns two rotation matrices R1 and R2, two new camera projective matrices with new intrinsics
    # P1 and P2, Q to reproject points in 3D and valid rois on the images.
    # N.B.  "alpha=0 means that the rectified images are zoomed and shifted so that
# only valid pixels are visible (no black areas after rectification). alpha=1 means
# that the rectified image is decimated and shifted so that all the pixels from the original images
# from the cameras are retained in the rectified images (no source image pixels are lost)."
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        intrinsics_l, dist_coefficients_l,
        intrinsics_r, dist_coefficients_r,
        size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    # TODO: alpha=0

    # R1 = np.mat(np.empty((3, 3)))
    # R2 = np.float32(np.empty((3, 3)))
    # P1 = np.float32(np.empty((3, 4)))
    # P2 = np.float32(np.empty((3, 4)))
    # Q = np.float32(np.empty((4, 4)))
    # cv2.stereoRectify(
    #     intrinsics_l, dist_coefficients_l, intrinsics_r, dist_coefficients_r, (w, h),
    #     rotation, translation, R1, R2, P1, P2, Q, cv2.CALIB_ZERO_DISPARITY, -1, (0, 0))
    # ipdb.set_trace()
    # we create the maps to rectify the images.
    mapx1, mapy1 = cv2.initUndistortRectifyMap(intrinsics_l, dist_coefficients_l, R1, P1,
                                               new_size,
                                               cv2.CV_32FC1)
    # TODO: cv2.CV_32F
    mapx2, mapy2 = cv2.initUndistortRectifyMap(intrinsics_r, dist_coefficients_r, R2, P2,
                                               new_size,
                                               cv2.CV_32FC1)
    # TODO: undistort and rectify based on the mappings
    # (could improve interpolation and image border settings here)
    # remapping the original images into the rectified images
    img_rect1 = cv2.remap(vis_l, mapx1, mapy1, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(vis_r, mapx2, mapy2, cv2.INTER_LINEAR)

    return img_rect1, img_rect2, Q


def compute_disparity(gray_l, gray_r, disparity_settings, method="SGBM"):
    """Function that computes disparity between two rectified grayscale images.

    Parameters
    ----------
    gray_l : left image
    gray_r : right image
    method : it can be SGBM or BM
    disparity_settings : dictionary of parameters for StereoSGBM or StereoBM

    Returns
    -------
    displ : raw disparity map, int16
    filtered_img : filtered disparity map, int16

    """
    p1 = 8 * 3 * disparity_settings['window_size'] ** 2
    p2 = 32 * 3 * disparity_settings['window_size'] ** 2

    if method == "SGBM":
        # careful here: numDisparities is the value that regulates how much we cut the disparity image from the left.
        left_matcher = cv2.StereoSGBM_create(minDisparity=disparity_settings['min_disparity'],
                                             numDisparities=disparity_settings['num_disparities'],
                                             blockSize=disparity_settings['block_size'],
                                             P1=p1,
                                             P2=p2,
                                             disp12MaxDiff=disparity_settings['disp12_max_diff'],
                                             uniquenessRatio=disparity_settings['uniqueness_ratio'],
                                             speckleWindowSize=disparity_settings['speckle_window_size'],
                                             speckleRange=disparity_settings['speckle_range'],
                                             preFilterCap=disparity_settings['pre_filter_cap']
                                             )
    elif method == "BM":
        left_matcher = cv2.StereoBM_create(numDisparities=disparity_settings['num_disparities'],
                                           blockSize=disparity_settings['block_size'])
    else:
        raise RuntimeError('Method not supported')

    # here we create a filter to improve the final depth map
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)

    wls_filter.setLambda(disparity_settings['lmbda'])
    wls_filter.setSigmaColor(disparity_settings['sigma'])

    # disparity map computation
    displ = left_matcher.compute(gray_l, gray_r)
    dispr = right_matcher.compute(gray_r, gray_l)

    # we filter the depth map here
    filtered_img = wls_filter.filter(displ, gray_l, None, dispr)

    return displ, filtered_img


def project_points_3D(disparity, Q):
    """Project 2D points back to 3D.

    description:
    This function reprojects points in 3D using the disparity image and the projective matrix Q returned by
    opencv stereo_rectify function.

    Parameters
    ----------
    disparity : int16 disparity map as returned by stereoBM (or stereoSGBM).compute
    Q : projection matrix returned by stereo_rectify

    Returns
    -------
    points_3d : set of 3D points
    """
    if disparity.dtype is not np.float32:
        disparity = disparity.astype(np.float32) / 16.0

    if Q.dtype is not np.float32:
        Q = Q.astype(np.float32)

    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    return points_3d


def is_file(the_path):
    """Check whether the file exists.

    inputs: the full path
    output: bool value whether it exists.

    """
    return os.path.isfile(the_path)


def rectify_opencv(camera_l_pose, camera_r_pose, intrinsics_l, intrinsics_r, vis_l, vis_r, dist_coefficients_l=None,
                   dist_coefficients_r=None):
    """Function that rectifies two images using opencv stereoRectify method.

    Parameters
    ----------
    camera_l_pose : 4x4, left camera pose
    camera_r_pose : 4x4, right camera pose
    intrinsics_l : intrinsic parameters for left camera
    intrinsics_r : intrinsic parameters for right camera
    vis_l : left image
    vis_r : right image
    dist_coefficients_l : a 5-dimensional numpy array of distortion coefficients for the first camera
    dist_coefficients_r : a 5-dimensional numpy array of distortion coefficients for the second camera
    scale : multiplier that determines the size of the rectified image

    Returns
    -------
    img_rect1 : left image rectified
    img_rect2 : right image rectified
    Q : projection matrix to re-project points in 3D
    """
    # if not check_epipoles(intrinsics_l, intrinsics_r, camera_l_pose, camera_r_pose, vis_l, vis_r):
    #     raise RuntimeError('One of the epipoles is in the image, rectification cannot be performed')

    if dist_coefficients_l is None:
        dist_coefficients_l = np.array([0.0, 0, 0, 0, 0])
    if dist_coefficients_r is None:
        dist_coefficients_r = np.array([0.0, 0, 0, 0, 0])

    rotation = camera_r_pose[:3, :3].transpose().dot(camera_l_pose[:3, :3])
    translation = camera_r_pose[:3, :3].transpose().dot(camera_l_pose[:3, 3] - camera_r_pose[:3, 3])
    # or, we could use the inverse of the camera r pose multiplied by the left camera pose directly and then
    # take the rotation and translation from it
    # r_transform_l = np.linalg.inv(camera_r_pose).dot(camera_l_pose)

    h, w = vis_l.shape[:2]
    # it looks like initUndistortRectifyMap wants (width, height)
    size = (w, h)
    # size of the rectified image
    # new_size = (round(size[0] * scale), round(size[1] * scale))
    new_size = size

    # stereoRectify returns two rotation matrices R1 and R2, two new camera projective matrices with new intrinsics
    # P1 and P2, Q to reproject points in 3D and valid rois on the images.
    # N.B.  "alpha=0 means that the rectified images are zoomed and shifted so that
# only valid pixels are visible (no black areas after rectification). alpha=1 means
# that the rectified image is decimated and shifted so that all the pixels from the original images
# from the cameras are retained in the rectified images (no source image pixels are lost)."
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        intrinsics_l, dist_coefficients_l,
        intrinsics_r, dist_coefficients_r,
        size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    # TODO: alpha=0

    # R1 = np.mat(np.empty((3, 3)))
    # R2 = np.float32(np.empty((3, 3)))
    # P1 = np.float32(np.empty((3, 4)))
    # P2 = np.float32(np.empty((3, 4)))
    # Q = np.float32(np.empty((4, 4)))
    # cv2.stereoRectify(
    #     intrinsics_l, dist_coefficients_l, intrinsics_r, dist_coefficients_r, (w, h),
    #     rotation, translation, R1, R2, P1, P2, Q, cv2.CALIB_ZERO_DISPARITY, -1, (0, 0))
    # ipdb.set_trace()
    # we create the maps to rectify the images.
    mapx1, mapy1 = cv2.initUndistortRectifyMap(intrinsics_l, dist_coefficients_l, R1, P1,
                                               new_size,
                                               cv2.CV_32FC1)
    # TODO: cv2.CV_32F
    mapx2, mapy2 = cv2.initUndistortRectifyMap(intrinsics_r, dist_coefficients_r, R2, P2,
                                               new_size,
                                               cv2.CV_32FC1)
    # TODO: undistort and rectify based on the mappings
    # (could improve interpolation and image border settings here)
    # remapping the original images into the rectified images
    img_rect1 = cv2.remap(vis_l, mapx1, mapy1, cv2.INTER_LINEAR)
    img_rect2 = cv2.remap(vis_r, mapx2, mapy2, cv2.INTER_LINEAR)

    return img_rect1, img_rect2, Q


def image_measure(img, test_mode=True):
    """Image smoothing and denoising."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    img = ndimage.gaussian_filter(img, sigma=5)
    filter_blurred_f = ndimage.gaussian_filter(img, 3)
    alpha = 30
    sharpened = img + alpha * (img - filter_blurred_f)
    if(test_mode):
        plt.imshow(sharpened, cmap='gray')
        plt.show()
    img = sharpened
    return img


def image_save(the_path, the_img_data):
    """Save the image.

    Arguments:
    the_path: the full path you want to save.
    the_img_data: the image data you want to save.

    No return.
    """
    from scipy.misc import imsave
    imsave(the_path, the_img_data)
