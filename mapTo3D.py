# @Author: Hao G <hao>
# @Date:   2018-01-02T11:22:32+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-29T14:06:11+00:00
# !/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2
import common.common as cm


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


if __name__ == '__main__':
    print('loading images...')
    img_depth = cv2.imread('./imgs/sgbm_disparity.png', 0)  # downscale images for faster processing
    imgOri = cv2.imread('./imgs/rectified_l_tmp.png', 1)
    # imgOri = cv2.imread('./imgs/aloeL.jpg', 1)
    # imgOri_gray = cv2.imread('rectified_l_tmp.png', 0)
    # ipdb.set_trace()
    cm.image_show(img_depth)

    print('computing disparity...')
    # disp = imgL.astype(np.float32) / 16.0
    f = 1164
    # baseline = 0.001
    # disp = f * baseline / img_depth
    disp = img_depth  # depth = focal*baseline/disparity
    # disp = stereo1.astype(np.float32) / 16.0
    # disp = stereo1 #filtered_img
    print('generating 3d point cloud...')
    h, w = img_depth.shape[:2]
    # f = 0.8 * w                          # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5 * w],
    #                 [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0, -f],  # so that y-axis looks up
    #                 [0, 0, 1, 0]])
    # ipdb.set_trace()
    focal_x = 1164
    focal_y = 1164
    c_x = 360
    c_y = 640
    Q = np.float32([[1, 0, 0, -c_y],
                    [0, -1, 0, c_x],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])
    # Q = np.float32([[1, 0, 0, -c_x],
    #                 [0, -1, 0, c_y],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0, -f],  # so that y-axis looks up
    #                 [0, 0, -1/Tx, (c_x-c_x')/Tx]])
    # points = cv2.reprojectImageTo3D(disp, Q, ddepth=cv2.CV_32F)
    # depth_array = img_depth
    imgOri_gray = cv2.cvtColor(imgOri, cv2.COLOR_BGR2GRAY)
    points = cm.map2Dto3D(imgOri_gray, img_depth, c_x, c_y, focal_x, focal_y, -1)
    tmp = np.unique(img_depth)
    # mask = np.logical_and(disp < 25, disp > 0)
    # img_depth[~mask] = 0
    # ipdb.set_trace()

    # ipdb.set_trace()
    points_normalize = np.zeros(points.shape)
    points_normalize = cv2.normalize(points, points_normalize, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    colors1 = cv2.cvtColor(imgOri, cv2.COLOR_BGR2RGB)
    colors = colors1[:, ::-1]
    # ipdb.set_trace()
    # yaxis = colors[:,]
    cm.npz_load('transform.npz', 'transform')
    points = (transform[:3, :3].dot(points.T) + transform[:3, 3, None]).T
    points = np.fliplr(points)
    # ipdb.set_trace()

    tmp = np.unique(disp.flatten())
    # ipdb.set_trace()
    # mask = np.logical_and(disp < 2, disp > 0)
    # ipdb.set_trace()
    # ipdb.set_trace()
    # mask = disp == 1
    mask = disp > disp.min() - 1

    # mask = disp <10
    out_points = points[mask]
    out_colors = colors[mask]
    # ipdb.set_trace()
    out_points[np.isnan(out_points)] = 0
    out_points[np.isinf(out_points)] = 0
    # ipdb.set_trace()
    out_fn = 'try.ply'
    cm.write_ply(out_fn, out_points, out_colors)
    print('saved {} ply'.format(out_fn))

    K = cm.intrinsic_cal(focal_x, focal_y, c_x, c_y)
    points1, _ = cm.depthTo3D(img_depth, K)
    out_fn = 'try1.ply'
    # ipdb.set_trace()

    colors1 = np.reshape(colors1, (colors1.shape[0] * colors1.shape[1], colors1.shape[2]))
    cm.mesh_to_ply(out_fn, points1, colors=colors1)
