# @Author: Hao G <hao>
# @Date:   2018-01-02T11:22:32+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-05T13:42:41+00:00
# !/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import handy_function as hf
from scipy import misc, ndimage
# from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
#                                  denoise_wavelet, estimate_sigma)
import ipdb

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
    """Ply write."""
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    asd = bytes(ply_header % dict(vert_num=len(verts)), 'utf-8')
    with open(fn, 'wb') as f:
        f.write(asd)
        np.savetxt(f, verts, '%f %f %f %d %d %d')
    # ipdb.set_trace()
    # with open(fn, 'r') as original:
    #         data = original.read()
    # with open('out_put_name.ply', 'w') as modified:
    #         modified.write(ply_header % dict(vert_num=len(verts)) + data)
    # modified.close()


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


if __name__ == '__main__':
    print('loading images...')
    left = cv2.imread('/home/hao/mc-cnn/left.png', 0)
    asd = np.memmap('/home/hao/mc-cnn/left.bin', dtype=np.float32, shape=(1, 228, 1280, 720))
    ipdb.set_trace()
    righ = cv2.imread('/home/hao/mc-cnn/right.png', 0)
    imgL = cv2.imread('rectified_l.png', 1)
    gray_l = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    displ16 = np.int16(left)
    dispr16 = np.int16(righ)
    window_size = 3
    min_disp = 0
    num_disp = 16
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63
        #mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    lmbda = 80000
    sigma = 1.2
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    ipdb.set_trace()
    filtered_img = wls_filter.filter(displ16, gray_l, None, dispr16)
    # filtered_img = cv2.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

    print('computing disparity...')
    # ipdb.set_trace()
    disp = filtered_img.astype(np.float32) / 16.0
    # disp = stereo1.astype(np.float32) / 16.0
    # disp = stereo1 #filtered_img
    print('generating 3d point cloud...')
    h, w = gray_l.shape[:2]
    # f = 0.8 * w                          # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5 * w],
    #                 [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0, -f],  # so that y-axis looks up
    #                 [0, 0, 1, 0]])
    f = 1164
    focal_x = 1164
    focal_y = 1164
    c_x = 360
    c_y = 640
    Q = np.float32([[1, 0, 0, -c_x],
                    [0, -1, 0, c_y],  # turn points 180 deg around x-axis,
                    [0, 0, 0, -f],  # so that y-axis looks up
                    [0, 0, 1, 0]])
    points = cv2.reprojectImageTo3D(disp, Q, ddepth=cv2.CV_32F)
    points_normalize = np.zeros(points.shape)
    points_normalize = cv2.normalize(points, points_normalize, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    # ipdb.set_trace()
    # points = points[:,:,2]*100
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    # ipdb.set_trace()

    # colors = colors[~np.isnan(points).any(axis=1),:,:]
    # disp[disp == disp.min()] = 0
    # disp[disp == disp.max()] = 0
    # ipdb.set_trace()
    [h, w] = disp.shape
    # disp = disp[np.int(h/4):np.int(h/2+h/4), np.int(w/4):np.int(w/2+w/4)]
    # points = points[np.int(h/4):np.int(h/2+h/4), np.int(w/4):np.int(w/2+w/4),:]
    # colors = colors[np.int(h/4):np.int(h/2+h/4), np.int(w/4):np.int(w/2+w/4),:]
    # mask = disp > disp.min()
    tmp = np.unique(disp.flatten())
    # ipdb.set_trace()
    # mask = np.logical_and(disp < tmp[-1], disp > 2)
    # disp = disp/disp.max()
    # from sklearn.preprocessing import normalize
    # disp = normalize(disp, norm='l2')
    # ipdb.set_trace()
    asd = np.unique(disp)
    # mask = disp > asd[6]
    mask = disp > disp.min()
    # mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    # ipdb.set_trace()
    out_points[np.isnan(out_points)] = 0
    out_points[np.isinf(out_points)] = 0
    # ipdb.set_trace()
    out_fn = 'out4.ply'
    write_ply(out_fn, out_points, out_colors)
    print('saved {} ply'.format(out_fn))

    hf.image_show(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    # cv2.imshow('left', imgL)
    hf.image_show((disp - min_disp) / num_disp)
    hf.image_save('dis.png', (disp - min_disp) / num_disp)
    # ipdb.set_trace()
    # cv2.imshow('disparity', (disp - min_disp) / num_disp)
    # test
    # norm_coeff = 255 / stereo1.max()
    # cv2.imshow("disparity", stereo1 * norm_coeff / 255)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
