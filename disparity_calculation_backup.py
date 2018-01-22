# @Author: Hao G <hao>
# @Date:   2017-12-22T11:23:46+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2017-12-22T12:35:35+00:00


import argparse
import configparser
import os

import cv2
import ipdb
import plot_functions as pf
import handy_function as hf
import stereo_vision.stereo_vision as sv
from matplotlib import pyplot as plt
from scipy import misc


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
        default='0'
    )

    parser.add_argument(
        '-ir', '--image_right',
        help='right_image id',
        dest='imgr',
        type=int,
        default='10'
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
    image_data = hf.npz_load(npz_file_full_path, 'image_data')
    # pair of images
    vis_l = image_data[id1]['image_mat']
    vis_r = image_data[id2]['image_mat']
    # denoise
    rgb_l = cv2.cvtColor(vis_l, cv2.COLOR_BGR2GRAY)
    rgb_r = cv2.cvtColor(vis_r, cv2.COLOR_BGR2GRAY)
    vis_l = cv2.fastNlMeansDenoisingColored(rgb_l, None, 10, 10, 7, 21)
    vis_r = cv2.fastNlMeansDenoisingColored(rgb_r, None, 10, 10, 7, 21)

    plt.imshow(vis_l)
    plt.show()
    ipdb.set_trace()
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
    img_rect1, img_rect2, Q = sv.rectify_opencv(camera_l_pose, camera_r_pose, intrinsics_l, intrinsics_r, vis_l,
                                                vis_r)
    scale = 2
    if args.show_images:
        # we show the image pairs
        pf.show_image_pair(img_rect1, img_rect2, "rectified", scale)

    # we transform the images in grayscale
    gray_l = cv2.cvtColor(img_rect1, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_rect2, cv2.COLOR_BGR2GRAY)

    displ, filtered_img16 = sv.compute_disparity(gray_l, gray_r, settings)

    points_3d = sv.project_points_3D(displ, Q)

    if args.show_images:
        pf.show_disparity(displ, filtered_img16, scale)
        cv2.waitKey()

    if args.write_ply:
        colors = cv2.cvtColor(img_rect1, cv2.COLOR_BGR2RGB)
        mask = displ > displ.min()
        points_3d = points_3d[mask]
        out_colors = colors[mask]
        io.write_ply('pointcloud.ply', points_3d, out_colors)


if __name__ == '__main__':
    main()
