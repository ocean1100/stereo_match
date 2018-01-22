# @Author: Hao G <hao>
# @Date:   2017-12-21T12:17:06+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2017-12-22T11:36:56+00:00


import numpy as np
import matplotlib.pyplot as plt
import handy_function
import argparse
import ipdb
import json
import cv2
import os


def build_parser():
    """Input arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '-f', '--filename',
        help='Name of json file to load',
        dest='filename',
        type=str,
        default='/media/hao/DATA/Arkit/05122017-105828/05122017-105828.json'
    )

    parser.add_argument(
        '-image_dir',
        help='image directory',
        dest='folder',
        type=str,
        default='/media/hao/DATA/Arkit/05122017-105828/'
    )

    parser.add_argument(
        '-npz_dir',
        help='npz saved directory',
        dest='npz_dir',
        type=str,
        default='/media/hao/DATA/Arkit/05122017-105828/src/'
    )

    parser.add_argument(
        '-highresolution_suffix',
        help='highest_resolution suffix of the image name',
        dest='highresolution_suffix',
        type=str,
        default='-1.000'
    )

    parser.add_argument(
        '-img_format',
        help='image format',
        dest='img_format',
        type=str,
        default='.jpeg'
    )

    parser.add_argument(
        '-m', '--mode',
        help='camera mode',
        dest='mode',
        type=str,
        default='P'
    )

    parser.add_argument(
        '-test_mode',
        help='test mode is True of False',
        dest='test_mode',
        type=bool,
        default=False
    )

    return parser


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
    image_full_path = handy_function.path_join(img_dir, image_name)
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


def main():
    """Main function."""
    args = build_parser().parse_args()
    assert os.path.isfile(args.filename), (
        'File {} does not exist!'.format(args.filename)
    )
    json_data = handy_function.json_read(
        handy_function.path_join(args.filename))
    folder = handy_function.path_join(args.folder)
    mode = args.mode
    # write_transformations = False
    # if plot_transformations is set to true, all the camera positions are plotted using matplotlib
    plot_transformations = False
    if(args.test_mode is True):
        # write_transformations = True
        plot_transformations = True

    frames = json_data['frames']
    image_data = []
    dump_num = 0
    # ipdb.set_trace()
    image_name_tmp = []
    for ind, frame in enumerate(frames):
        # ipdb.set_trace()
        timestamp = frame['timestamp']
        image_name = str(timestamp) + args.highresolution_suffix + args.img_format
        image_full_path = handy_function.path_join(args.folder, image_name)
        if (handy_function.is_file(image_full_path) is False) or (image_name in image_name_tmp):
            if(args.test_mode is True):
                print('the image {} in the json file does not exist in the folder or repeated, so skip'.format([image_name]))
            dump_num = dump_num + 1
            continue

        t = frame['camera']['transform']
        i = frame['camera']['intrinsics']
        # in the .json file transforms and intrinsic are given in column
        # format [x, y, z, w], hence we need to transpose it to get it right
        intrinsic_matrix = np.array(
            [[i[0]['x'], i[0]['y'], i[0]['z']],
             [i[1]['x'], i[1]['y'], i[1]['z']],
             [i[2]['x'], i[2]['y'], i[2]['z']]]).T
        transforms_transpose = np.array(
            [[t[0]['x'], t[0]['y'], t[0]['z'], t[0]['w']],
             [t[1]['x'], t[1]['y'], t[1]['z'], t[1]['w']],
             [t[2]['x'], t[2]['y'], t[2]['z'], t[2]['w']],
             [t[3]['x'], t[3]['y'], t[3]['z'], t[3]['w']]]).T

        extrinsic_matrix = extrinsic_matrix_get(transforms_transpose, args.mode)

        frame_id = ind - dump_num
        # TODO: looks darker
        image_mat = cv2.imread(image_full_path, 1)
        image_data.append(
            {
                'timestamp': timestamp,
                'image_mat': image_mat,
                'frame_id': frame_id,
                'extrinsic': extrinsic_matrix,
                'intrinsic': intrinsic_matrix,
                'image_name': image_name
            })
        image_name_tmp.append(image_name)
    print('Now saving...')
    npz_file_full_path = handy_function.path_join(
        args.npz_dir, args.folder.split('/')[-1], 'tmp.npz')
    np.savez(npz_file_full_path, image_data=image_data)
    print('done!')
    if(args.test_mode is True):
        print('test_mode start...')
        i = 48
        tmp = image_data[i]['image_mat']
        cv2.imwrite('asd.png', tmp)
        from scipy.misc import imsave
        tmp1 = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        tmp2 = tmp1
        imsave('tmp1.png', tmp2)
        plt.imshow(tmp2)
        plt.show()
        for i in range(len(image_data)):
            print('{}'.format(image_data[i]['image_name']))
    if plot_transformations:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for pose in extrinsic_matrix:
            plot_camera(pose, '', ax, s=0.01)
        axis_equal_3d(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


if __name__ == '__main__':
    main()
