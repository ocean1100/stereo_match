# @Author: Hao G <hao>
# @Date:   2018-01-10T17:19:04+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-10T17:25:19+00:00



""" Machine Learning Recipe """
# pylint: disable=C0103
# pylint: disable=E0401
# pylint: disable=maybe-no-member

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

# local modules
# from common import splitfn

from matplotlib import pyplot as plt

# built-in modules
import os
import glob

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    print('loading images...')

    imgL = cv2.pyrDown( cv2.imread('/home/hao/MyCode/disparity_estimation/imgs/aloeL.jpg') )  # downscale images for faster processing
    imgR = cv2.pyrDown( cv2.imread('/home/hao/MyCode/disparity_estimation/imgs/aloeR.jpg') )

    #imgL = cv2.pyrDown( cv2.imread('D:/dev/AI/python/tsukuba-l.png') )  # downscale images for faster processing
    #imgR = cv2.pyrDown( cv2.imread('D:/dev/AI/python/tsukuba-r.png') )

    #imgL = cv2.pyrDown( cv2.imread('D:/dev/AI/python/output/left00_chess.png_undistorted.png') )  # downscale images for faster processing
    #imgR = cv2.pyrDown( cv2.imread('D:/dev/AI/python/output/right00_chess.png_undistorted.png') )

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                   [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                   [0, 0, 0,     -f], # so that y-axis looks up
                   [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp-min_disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #imgL = cv2.imread('D:/dev/AI/python/output/left00_chess.png_undistorted.png',0)
    #imgR = cv2.imread('D:/dev/AI/python/output/right00_chess.png_undistorted.png',0)
    #imgL = cv2.imread('D:/dev/AI/python/tsukuba_l.png',0)
    #imgR = cv2.imread('D:/dev/AI/python/tsukuba_r.png',0)

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size='])

    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)

    if not img_mask:
        img_mask = '../data/left*.jpg'  # default
    else:
        img_mask = img_mask[0]
    img_mask = 'D:/dev/AI/python/data/*'
    img_names = glob(img_mask)
    print(img_names)
    debug_dir = args.get('--debug')
    if not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))
    square_size = 1.0
    pattern_size = (9, 6)
    pattern_size = (7, 6)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0
    img_names_undistort = []
    for fn in img_names:
        print('processing %s... ' % fn, end='')
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            continue

        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            path, name, ext = splitfn(fn)
            outfile = debug_dir + name + '_chess.png'
            cv2.imwrite(outfile, vis)
            if found:
                img_names_undistort.append(outfile)

        if not found:
            print('chessboard not found')
            continue

        img_points.append(corners.reshape(-1, 2))
        obj_points.append(pattern_points)

        print('ok')

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('')
    for img_found in img_names_undistort:
        img = cv2.imread(img_found)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

        dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

        # crop and save the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        outfile = img_found + '_undistorted.png'
        print('Undistorted image written to: %s' % outfile)
        cv2.imwrite(outfile, dst)

cv2.destroyAllWindows()
