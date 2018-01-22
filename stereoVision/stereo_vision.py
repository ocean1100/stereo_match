# @Author: Hao G <hao>
# @Date:   2018-01-02T13:38:10+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-15T17:01:18+00:00



import numpy as np
import cv2


def check_epipoles(intrinsics_l, intrinsics_r, camera_l_pose, camera_r_pose, vis_l, vis_r):
    """Function that checks whether the epipoles are inside the images

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
    ipdb.set_trace()
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
    """Function that computes disparity between two rectified grayscale images

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
    """This function reprojects points in 3D using the disparity image and the projective matrix Q returned by
    opencv stereo_rectify function

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
