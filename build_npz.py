# @Author: Hao G <hao>
# @Date:   2017-12-21T12:17:06+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-29T13:30:17+00:00


import numpy as np
import matplotlib.pyplot as plt
import common.common as cm
import argparse
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


def main():
    """Main function."""
    args = build_parser().parse_args()
    assert os.path.isfile(args.filename), (
        'File {} does not exist!'.format(args.filename)
    )
    json_data = cm.json_read(
        cm.path_join(args.filename))
    # write_transformations = False
    # if plot_transformations is set to true, all the camera positions are plotted using matplotlib
    plot_transformations = False
    if(args.test_mode is True):
        # write_transformations = True
        plot_transformations = True

    frames = json_data['frames']
    image_data = []
    dump_num = 0
    image_name_tmp = []
    for ind, frame in enumerate(frames):
        timestamp = frame['timestamp']
        image_name = str(timestamp) + args.highresolution_suffix + args.img_format
        image_full_path = cm.path_join(args.folder, image_name)
        if (cm.is_file(image_full_path) is False) or (image_name in image_name_tmp):
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

        extrinsic_matrix = cm.extrinsic_matrix_get(transforms_transpose, args.mode)

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
    npz_file_full_path = cm.path_join(
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
            cm.plot_camera(pose, '', ax, s=0.01)
        cm.axis_equal_3d(ax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


if __name__ == '__main__':
    main()
