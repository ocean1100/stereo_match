# @Author: Hao G <hao>
# @Date:   2018-01-02T11:22:32+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-04T14:32:42+00:00
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
    # imgL = cv2.pyrDown( cv2.imread('../gpu/1.jpeg') )  # downscale images for faster processing
    # imgR = cv2.pyrDown( cv2.imread('../gpu/2.jpg') )
    # imgL_tmp = hf.image_read('rectified_l.png')
    # imgR_tmp = hf.image_read('rectified_r.png')
    # imgL_tmp = image_measure(imgL_tmp)
    # imgR_tmp = image_measure(imgR_tmp)
    # hf.image_save(hf.path_join(hf.directory_current_get, 'rectified_l_tmp.png'), imgL_tmp)
    # hf.image_save(hf.path_join(hf.directory_current_get, 'rectified_r_tmp.png'), imgR_tmp)
    imgL = cv2.imread('rectified_l.png', 1)  # downscale images for faster processing
    imgR = cv2.imread('rectified_r.png', 1)

    vis_l = image_measure(imgL)
    vis_r = image_measure(imgR)
    # ipdb.set_trace()
    imgL = cv2.fastNlMeansDenoisingColored(imgL, None, 10, 10, 7, 21)
    imgR = cv2.fastNlMeansDenoisingColored(imgR, None, 10, 10, 7, 21)
    hf.image_save('rectified_l_tmp.png', cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    hf.image_save('rectified_r_tmp.png', cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
    hf.image_show(imgL)
    gray_l = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # hf.image_show(vis_l)
    # ipdb.set_trace()

    # imgL = cv2.imread('/home/hao/MyCode/openCV_examples/samples/gpu/tsucuba_left.png', 1)  # downscale images for faster processing
    # imgR = cv2.imread('/home/hao/MyCode/openCV_examples/samples/gpu/tsucuba_right.png', 1)

    # imgL = cv2.imread('/home/hao/MyCode/openCV_examples/samples/gpu/123.jpeg', 1)  # downscale images for faster processing
    # imgR = cv2.imread('/home/hao/MyCode/openCV_examples/samples/gpu/1231.jpeg', 1)

    # disparity range is tuned for 'aloe' image pair
    # window_size = 3
    # min_disp = 16
    # num_disp = 112-min_disp
    # stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    #     numDisparities = num_disp,
    #     # SADWindowSize = window_size,
    #     uniquenessRatio = 10,
    #     speckleWindowSize = 100,
    #     speckleRange = 32,
    #     disp12MaxDiff = 1,
    #     P1 = 8*3*window_size**2,
    #     P2 = 32*3*window_size**2,
    #     # fullDP = False
    # )
    #     # depth computation using SGBM (we can use StereoBM alternatively)
    window_size = 3
    min_disp = 0
    num_disp = 16
    # careful here: numDisparities is the value that regulates how much we cut the disparity image from the left.
    stereo = cv2.StereoSGBM_create(
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
    left_matcher = stereo
    # here we create a filter to improve the final depth map
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    lmbda = 80000
    sigma = 1.2
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # disparity map computation
    # ipdb.set_trace()

    displ = left_matcher.compute(gray_l, gray_r)
    dispr = right_matcher.compute(gray_r, gray_l)
    # displ = left_matcher.compute(gray_l, gray_r).astype(np.float32)
    # dispr = right_matcher.compute(gray_r, gray_l).astype(np.float32)
    displ16 = np.int16(displ)
    dispr16 = np.int16(dispr)
    # we filter the depth map here
    filtered_img = wls_filter.filter(displ16, gray_l, None, dispr16)
    # filtered_img = cv2.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)

    print('computing disparity...')
    # stereo1 = stereo.compute(imgL, imgR)
    stereo1 = stereo.compute(gray_l, gray_r)
    disp = filtered_img.astype(np.float32)/16.0

    # disp = stereo1.astype(np.float32) / 16.0
    # disp = stereo1 #filtered_img
    print('generating 3d point cloud...')
    h, w = gray_l.shape[:2]
    f = 0.8 * w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                    [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
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
    mask = disp > 2
    out_points = points[mask]
    out_colors = colors[mask]
    # ipdb.set_trace()
    out_points[np.isnan(out_points)] = 0
    out_points[np.isinf(out_points)] = 0
    # ipdb.set_trace()
    out_fn = 'out1.ply'
    write_ply(out_fn, out_points, out_colors)
    print('saved {} ply'.format(out_fn))

    hf.image_show(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
    # cv2.imshow('left', imgL)
    hf.image_show((disp - min_disp) / num_disp)
    # ipdb.set_trace()
    # cv2.imshow('disparity', (disp - min_disp) / num_disp)
    # test
    # norm_coeff = 255 / stereo1.max()
    # cv2.imshow("disparity", stereo1 * norm_coeff / 255)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
