# @Author: Hao G <hao>
# @Date:   2018-01-02T11:29:25+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-29T13:11:56+00:00


import argparse
import configparser
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image


def build_parser():
    """Input arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-s', '--settings_file',
        help='settings file to load',
        dest='settings_file',
        type=str,
        default='/media/hao/DATA/Arkit/05122017-105828/src/settings.ini'
    )

    parser.add_argument(
        '-il', '--image_left',
        help='left_image id',
        dest='imgl',
        type=int,
        default='8'
    )

    parser.add_argument(
        '-ir', '--image_right',
        help='right_image id',
        dest='imgr',
        type=int,
        default='11'
    )

    parser.add_argument(
        '-write_ply', '--write_ply',
        action='store_true', dest='write_ply',
        default=False,
        help='write 3D point cloud into a file called pointcloud.ply'
    )

    parser.add_argument(
        '-show_images', '--show_images',
        action='store_true', dest='show_images',
        default=False,
        help='show rectified and disparity images'
    )

    parser.add_argument(
        '-test_mode',
        action='store_true', dest='test_mode',
        default=False,
        help='test_mode'
    )

    return parser


def parse_config_file(settings_file):
    """Parses settings file and return a set of configuration parameters for disparity.

    Parameters
    ----------
    settings_file : settings file

    Return
    ------
    settings : set of configuration parameters
    """
    config = configparser.ConfigParser()
    settings = {
        'npz_file': '/media/hao/DATA/Arkit/05122017-105828/src/tmp.npz',
        'mode': 'P',
        'window_size': 3, 'min_disparity': 0, 'num_disparities': 160, 'block_size': 5, 'disp12_max_diff': 1,
        'uniqueness_ratio': 15, 'speckle_window_size': 0, 'speckle_range': 2, 'pre_filter_cap': 63,
        'lmbda': 80000, 'sigma': 1.2}
    if settings_file is None:
        return settings

    if not os.path.isfile(settings_file):
        return settings

    config.read(settings_file)
    disparity_config = config['disparity']
    if 'npz_file' in disparity_config:
        settings['npz_file'] = str(disparity_config['npz_file'])
    if 'mode' in disparity_config:
        settings['mode'] = str(disparity_config['mode'])
    if 'window_size' in disparity_config:
        settings['window_size'] = int(disparity_config['window_size'])
    if 'min_disparity' in disparity_config:
        settings['min_disparity'] = int(disparity_config['min_disparity'])
    if 'num_disparities' in disparity_config:
        settings['num_disparities'] = int(disparity_config['num_disparities'])
    if 'block_size' in disparity_config:
        settings['block_size'] = int(disparity_config['block_size'])
    if 'disp12_max_diff' in disparity_config:
        settings['disp12_max_diff'] = int(disparity_config['disp12_max_diff'])
    if 'uniqueness_ratio' in disparity_config:
        settings['uniqueness_ratio'] = int(disparity_config['uniqueness_ratio'])
    if 'speckle_window_size' in disparity_config:
        settings['speckle_window_size'] = int(disparity_config['speckle_window_size'])
    if 'speckle_range' in disparity_config:
        settings['speckle_range'] = int(disparity_config['speckle_range'])
    if 'pre_filter_cap' in disparity_config:
        settings['pre_filter_cap'] = int(disparity_config['pre_filter_cap'])
    if 'lmbda' in disparity_config:
        settings['lmbda'] = int(disparity_config['lmbda'])
    if 'sigma' in disparity_config:
        settings['sigma'] = float(disparity_config['sigma'])

    return settings


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


def show_image_pair(img_rect1_res, img_rect2_res, scale=1):
    """Function that shows a pair of images in the same window with horizontal lines to show that the y of the
    two images is supposed to be the same

    Parameters
    ----------
    img_rect1_res : rectified let image
    img_rect2_res : rectified right image

    """

    # computes the total size of the two images
    total_size = (max(img_rect1_res.shape[0], img_rect2_res.shape[0]),
                  img_rect1_res.shape[1] + img_rect2_res.shape[1], 3)
    img = np.zeros(total_size, dtype=np.uint8)
    # fill a new bigger image with the two small images
    # ipdb.set_trace()
    img[:img_rect1_res.shape[0], :img_rect1_res.shape[1]] = img_rect1_res[:, :, :3]
    img[:img_rect2_res.shape[0], img_rect1_res.shape[1]:] = img_rect2_res[:, :, :3]

    # draw horizontal lines every 25 px accross the side by side image
    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

    # resize the image if it's too big
    res = cv2.resize(img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)

    # and we show
    plt.imshow(res)
    plt.show()


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
    # ipdb.set_trace()
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
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
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


def image_save(the_path, the_img_data):
    """Save the image.

    Arguments:
    the_path: the full path you want to save.
    the_img_data: the image data you want to save.

    No return.
    """
    from scipy.misc import imsave
    imsave(the_path, the_img_data)


def npz_load(the_file_path, the_value_name):
    """Load the npz file.

    Arguments:
    the_file_path: the full path.
    the_value_name: the value name in the npz file.

    return:
    the npz value.
    """
    npzfile = np.load(the_file_path)
    return npzfile[the_value_name]


def image_show(image_array):
    """Show the image.

    Arguments:
    image_array: the image array.

    No return.
    """
    # change from the range 0-1 to 0-255 for display.
    if(image_array.max() != 255):
        tmp = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        image_array = Image.fromarray(tmp.astype('uint8'))
    else:
        image_array = Image.fromarray(image_array.astype('uint8'))
    image_array.show()


def directory_current_get():
    """Get the current directory."""
    return os.getcwd()


def path_join(*args):
    """Do the os path join thing."""
    return os.path.join(*args)


def main():
    """Main."""
    args = build_parser().parse_args()
    # pair of frames that we are taking, for example it could be 0 and 10
    id1 = args.imgl
    id2 = args.imgr

    settings = parse_config_file(args.settings_file)

    if id2 <= id1:
        raise ValueError('id2 must be greater than id1')
    if id1 < 0 or id2 < 0:
        raise ValueError('id1 and id2 must be greater than 0')

    mode = settings['mode']
    npz_file_full_path = settings['npz_file']
    image_data = npz_load(npz_file_full_path, 'image_data')
    # pair of images
    vis_l = image_data[id1]['image_mat']
    vis_r = image_data[id2]['image_mat']
    # denoise

    # vis_l = image_measure(vis_l)
    # vis_r = image_measure(vis_r)
    # ipdb.set_trace()
    # vis_l = cv2.fastNlMeansDenoisingColored(vis_l, None, 10, 10, 7, 21)
    # vis_r = cv2.fastNlMeansDenoisingColored(vis_r, None, 10, 10, 7, 21)
    image_show(vis_l)
    image_show(vis_r)
    #
    # ipdb.set_trace()
    # poses relative to those frames
    camera_l_pose = image_data[id1]['extrinsic']
    camera_r_pose = image_data[id2]['extrinsic']

    # intrinsics relative to those frames
    intrinsics_l = image_data[id1]['intrinsic']
    intrinsics_r = image_data[id2]['intrinsic']
    # ipdb.set_trace()
    # It looks like the intrinsics are kept the same even when the image is in portrait mode. I'm quite convinced
    # that we need to switch the principal point coordinates if the camera is in portrait mode.
    if mode == 'P':
        intrinsics_l[:2, 2] = intrinsics_l[:2, 2][::-1]
        intrinsics_r[:2, 2] = intrinsics_r[:2, 2][::-1]

    # opencv rectification
    img_rect1, img_rect2, Q = rectify_opencv(
        camera_l_pose, camera_r_pose, intrinsics_l, intrinsics_r, vis_l,
        vis_r)
    img_rect1 = cv2.cvtColor(img_rect1, cv2.COLOR_BGR2RGB)
    img_rect2 = cv2.cvtColor(img_rect2, cv2.COLOR_BGR2RGB)
    scale = 1
    if args.show_images:
        # we show the image pairs
        show_image_pair(img_rect1, img_rect2, scale)
    # ipdb.set_trace()
    current_dir = directory_current_get()
    imgL_save = path_join(current_dir, 'rectified_l11.png')
    imgR_save = path_join(current_dir, 'rectified_r11.png')
    image_save(imgL_save, img_rect1)
    image_save(imgR_save, img_rect2)
    print('Image L and R saved!')

    # handy_function()


if __name__ == '__main__':
    main()
