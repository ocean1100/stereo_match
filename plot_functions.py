# @Author: Hao G <hao>
# @Date:   2017-12-22T12:08:21+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2017-12-22T12:08:24+00:00



import cv2
from matplotlib import pyplot as plt
import numpy as np


def axis_equal_3d(ax):
    """Function that makes 3D axis equal

    Parameters
    ----------
    ax : matplotlib 3D axes

    """
    extents = np.array([
        getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz']
    )
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


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


def show_image_pair(img_rect1, img_rect2, window_name, scale=1):
    """Function that shows a pair of images in the same window with horizontal lines to show that the y of the
    two images is supposed to be the same

    Parameters
    ----------
    img_rect1 : rectified left image
    img_rect2 : rectified right image
    window_name : name of the window
    scale : if set to a number greater than 1, it resizes the image to (width/scale, height/scale)
    """

    if scale < 1:
        raise ValueError('scale must be greather or equal than 1')

    # here we are resizing the images down for visualization purposes
    img_rect1_res = cv2.resize(img_rect1, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
    img_rect2_res = cv2.resize(img_rect2, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)

    # computes the total size of the two images
    total_size = (max(img_rect1_res.shape[0], img_rect2_res.shape[0]),
                  img_rect1_res.shape[1] + img_rect2_res.shape[1], 3)
    img = np.zeros(total_size, dtype=np.uint8)
    # fill a new bigger image with the two small images
    img[:img_rect1_res.shape[0], :img_rect1_res.shape[1]] = img_rect1_res
    img[:img_rect2_res.shape[0], img_rect1_res.shape[1]:] = img_rect2_res

    # draw horizontal lines every 25 px accross the side by side image
    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

    # and we show
    cv2.imshow(window_name, img)


def show_disparity(disparity, filtered=None, scale=1):
    """Function that shows disparity images

    Parameters
    ----------
    disparity : int16 raw disparity image as returned by stereoBM (or stereoSGBM).compute
    filtered : int16 filtered disparity image returned by wls_filter
    scale : if set to a number greater than 1, it resizes the image to (width/scale, height/scale)
    """

    if scale < 1:
        raise ValueError('scale must be greather or equal than 1')

    disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)
    displ_res = cv2.resize(disparity, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("raw disparity", displ_res)

    if filtered is not None:
        filtered = cv2.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filtered = np.uint8(filtered)
        filtered_img_res = cv2.resize(filtered, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("filtered disparity", filtered_img_res)


def plot_transforms(transforms):
    """Function that plots a set of camera poses

    Parameters
    ----------
    transforms : set of camera poses to plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pose in transforms:
        plot_camera(pose, '', ax, s=0.01)
    axis_equal_3d(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
