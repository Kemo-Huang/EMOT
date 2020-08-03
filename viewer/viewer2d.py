import cv2
import sys
import json
import os
import numpy as np
from numba import njit


data_root = '/home/kemo/Dataset/sustechscapes-mini-dataset'


@njit
def euler_angle_to_rotate_matrix(eu, t):
    theta = eu
    # Calculate rotation about x axis
    R_x = np.array([
        [1,       0.,              0.],
        [0.,      np.cos(theta[0]),   -np.sin(theta[0])],
        [0.,      np.sin(theta[0]),   np.cos(theta[0])]
    ])

    # Calculate rotation about y axis
    R_y = np.array([
        [np.cos(theta[1]),      0.,      np.sin(theta[1])],
        [0.,                    1.,      0.],
        [-np.sin(theta[1]),     0.,      np.cos(theta[1])]
    ])

    # Calculate rotation about z axis
    R_z = np.array([
        [np.cos(theta[2]), -np.sin(theta[2]),   0.],
        [np.sin(theta[2]), np.cos(theta[2]),    0.],
        [0.,               0.,                  1.]])

    R = np.dot(R_x, np.dot(R_y, R_z))
    t = t.reshape((-1, 1))
    R = np.concatenate((R, t), axis=-1)
    R = np.concatenate((R, np.array([0, 0, 0, 1]).reshape((1, -1))), axis=0)
    return R

@njit
def psr_to_xyz(p, s, r):
    trans_matrix = euler_angle_to_rotate_matrix(r, p)
    
    x = s[0]/2
    y = s[1]/2
    z = s[2]/2

    local_coord = np.array([
        x, y, -z, 1,   x, -y, -z, 1,  # front-left-bottom, front-right-bottom
        x, -y, z, 1,   x, y, z, 1,  # front-right-top,   front-left-top

        -x, y, -z, 1,   -x, -y, -z, 1,  # rear-left-bottom, rear-right-bottom
        -x, -y, z, 1,   -x, y, z, 1,  # rear-right-top,   rear-left-top
    ]).reshape((-1, 4))

    world_coord = np.dot(trans_matrix, np.transpose(local_coord))
    return world_coord

@njit
def proj_pts3d_to_img(pts, extrinsic, intrinsic):
    imgpos3 = np.dot(extrinsic, pts)

    # rect matrix shall be applied here, for kitti

    if np.any(imgpos3[2] < 0):
        return None

    imgpos2 = np.dot(intrinsic, imgpos3)
    
    return imgpos2[0:2, :] / imgpos2[2:, :]


def box_to_2d_points(psr, extrinsic, intrinsic):
    box = np.array([
        [psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]],
        [psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]],
        [psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]],
    ])
    box3d = psr_to_xyz(box[0], box[1], box[2])
    return proj_pts3d_to_img(box3d, extrinsic, intrinsic)


def draw_projected_box3d(image, box, color=(0, 255, 0), thickness=1):
    box = box.astype(np.int32)
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (box[0, i], box[1, i]),
                 (box[0, j], box[1, j]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (box[0, i], box[1, i]), (box[0, j], box[1, j]), color,
                 thickness)

        i, j = k, k + 4
        cv2.line(image, (box[0, i], box[1, i]), (box[0, j], box[1, j]), color,
                 thickness)
    cv2.fillPoly(image, [np.transpose(box[:, :4])], color)
    return image


def main():
    frame = sys.argv[1]
    image_filename = os.path.join(data_root, "camera", "front", frame + ".jpg")
    label_filename = os.path.join(data_root, "label", frame + ".json")
    calib_filename = os.path.join(data_root, "calib", "camera", "front.json")

    with open(image_filename, 'r') as f:
        image = cv2.imread(image_filename)

    with open(label_filename, 'r') as f:
        labels = json.load(f)

    with open(calib_filename, 'r') as f:
        calib = json.load(f)

    extrinsic = np.array(calib['extrinsic']).reshape(4, 4)[:3, :]
    intrinsic = np.array(calib['intrinsic']).reshape(3, 3)

    # image_shape = [0, 0, image.shape[0], image.shape[1]]

    for label in labels:
        if label['obj_type'] != 'Car':
            continue
        psr = label['psr']
        corners_2d = box_to_2d_points(psr, extrinsic, intrinsic)
        if corners_2d is not None:
            draw_projected_box3d(image, corners_2d)

    cv2.imshow('image', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
