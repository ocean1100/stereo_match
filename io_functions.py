# @Author: Hao G <hao>
# @Date:   2017-12-22T13:58:16+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2017-12-22T13:58:18+00:00



import json
import numpy as np
import os
import cv2


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
    """Function that writes 3D points and associated colors on a .ply file.

    Parameters
    ----------
    fn : name of the file on which we want to save the point cloud
    verts : set of 3D points
    colors : set of colors

    """

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def read_json(folder):
    """Reads a json file and returns a json structure

    Parameters
    ----------
    folder : absolute path to the folder containing .json file

    Returns
    -------
    data_json : a json structure

    """
    json_file = [f for f in os.listdir(folder) if f.endswith('.json')][0]
    json_file = os.path.join(folder, json_file)
    with open(json_file, 'r') as fp:
        data_json = json.load(fp)
    return data_json


def read_image_data(folder, timestamps):
    """Function to read images from a folder, compute the closest timestamp from a set of timestamps
    taken from a .json file, and return a list of images, frames and frame_ids

    Parameters
    ----------
    folder : folder from which we want to read image data
    timestamps : set of timestamps returned from ARKit and saved in the .json file

    Returns
    -------
    image_data : list of dictionaries with the following fields:
        timestamp : timestamp of the image
        image : image read from the folder
        frame_id : index of the closest timestamp in the list of timestamps passed in input
    """
    image_data = []

    for fn in sorted(os.listdir(folder)):

        # we want only images
        if fn[len(fn) - 4:] != "jpeg":
            continue

        image_fn = os.path.join(folder, fn)
        try:
            image = cv2.imread(image_fn)
        except IOError:
            continue

        timestamp = float(fn[:len(fn) - 11])

        frame_id = (np.abs(timestamps - timestamp)).argmin()

        image_data.append(
            {
                'timestamp': timestamp,
                'image': image,
                'frame_id': frame_id
            }
        )

    return image_data


def write_transform(file, t):
    """Function to save a transformation matrix on file

    Parameters
    ----------
    file : opened stream
    t : transformation matrix
    """
    mat = np.matrix(t)
    np.savetxt(file, mat, fmt='%.8f')
