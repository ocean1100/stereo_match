# @Author: Hao G <hao>
# @Date:   2017-12-21T12:17:06+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2017-12-21T15:11:24+00:00


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
        default='/media/hao/DATA/Arkit/05122017-105828/npzfile/'
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


def npz_wrap(
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
    image_data = []
    dump_num = 0
    for ind, file_name in enumerate(sorted(os.listdir(img_dir))):
        ipdb.set_trace()
        if os.path.splitext(file_name)[1] != file_format:
            dump_num = dump_num + 1
            continue

        timestamp = float(os.path.splitext(file_name)[0].split(split_symbol)[0])
        image_name = str(timestamp) + highresolution_suffix + file_format
        image_full_path = handy_function.path_join(img_dir, image_name)
        try:
            image = cv2.imread(image_full_path)
        except IOError:
            continue
        frame_id = (np.abs(timestamps - timestamp)).argmin()
        image_data.append(
            {
                'timestamp': timestamp,
                'image': image,
                'frame_id': frame_id,
                'extrinsic': extrinsics[ind - dump_num],
                'intrinsic': intrinsics[ind - dump_num]
            }
        )
        print('{}'.format(ind))
    ipdb.set_trace()
    npz_file_full_path = handy_function.path_join(
        npz_dir, img_dir.split('/')[-1])
    np.savez(npz_file_full_path, image_data=image_data)


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
    extrinsic_matrix = []
    for t in transforms:
        tmp = world_matrix_arkitw.dot(t)
        extrinsic_matrix.append(tmp.dot(arkitc_matrix_opencv))

    return extrinsic_matrix


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

    timestamps = np.array([frame['timestamp'] for frame in frames])
    transforms = [frame['camera']['transform'] for frame in frames]
    # image_data = read_image_data(folder, timestamps)

    # in the .json file transforms are given in column format [x, y, z, w], hence we need to transpose it to get it
    # right
    transforms_transpose = [
        np.array([[t[0]['x'], t[0]['y'], t[0]['z'], t[0]['w']],
                  [t[1]['x'], t[1]['y'], t[1]['z'], t[1]['w']],
                  [t[2]['x'], t[2]['y'], t[2]['z'], t[2]['w']],
                  [t[3]['x'], t[3]['y'], t[3]['z'], t[3]['w']]]).T
        for t in transforms
    ]
    intrinsics = [frame['camera']['intrinsics'] for frame in frames]
    # again, here we transpose because the json file gives us columns
    intrinsics = [
        np.array([[i[0]['x'], i[0]['y'], i[0]['z']],
                  [i[1]['x'], i[1]['y'], i[1]['z']],
                  [i[2]['x'], i[2]['y'], i[2]['z']]]).T
        for i in intrinsics
    ]
    extrinsic_matrix = extrinsic_matrix_get(transforms_transpose, args.mode)
    npz_wrap(folder, timestamps, intrinsics, extrinsic_matrix, args.npz_dir)
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
