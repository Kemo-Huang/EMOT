import numpy as np
import json
import os
from numba import njit
from typing import List, Dict
from .box_util import psr_to_corners3d, euler_angle_to_rotation_matrix


@njit
def proj_pts3d_to_img(pts, extrinsic, intrinsic):
    pts = np.vstack((pts, np.ones((1, pts.shape[1]))))
    imgpos3 = np.dot(extrinsic, pts)

    # rect matrix shall be applied here, for kitti

    if np.any(imgpos3[2] < 0):
        return None

    imgpos2 = np.dot(intrinsic, imgpos3)

    return imgpos2[0:2, :] / imgpos2[2:, :]


def corners3d_to_box2d(corners3d, extrinsic, intrinsic):
    """
    world coord to image coord
    :param corners3d: 3 * 8
    :param extrinsic: 3 * 4
    :param intrinsic: 3 * 3
    :return: x1, y1, x2, y2
    """
    pts2d = proj_pts3d_to_img(corners3d, extrinsic, intrinsic)
    if pts2d is None:
        return None
    x1, y1 = np.min(pts2d, axis=1)
    x2, y2 = np.max(pts2d, axis=1)
    if np.all([x1, x2, y1, y2]) < 0 or (x2 - x1) * (y2 - y1) > 3145728:
        return None
    else:
        return int(x1), int(y1), int(x2), int(y2)


def read_calib_file(path):
    with open(path) as fp:
        data = json.load(fp)
    extrinsic = np.array(data["extrinsic"]).reshape((4, 4))[:3, :]
    intrinsic = np.array(data["intrinsic"]).reshape((3, 3))
    return extrinsic, intrinsic


def read_gt_seq(root_dir, modality, allow_empty=False) -> List[Dict]:
    extrinsic, intrinsic = read_calib_file(f"{root_dir}/calib/camera/front.json")
    seq_gt = []
    # check input
    with open(os.path.join(f"{root_dir}", "front_cars.txt"), 'w+') as front_cars_file:
        # sustech mini: 000000 - 000595
        for i in range(0, 595, 5):
            frame_id = str(i).zfill(6)
            with open(os.path.join(root_dir, "label", frame_id + ".json")) as fp:
                labels = json.load(fp)
            boxes_3d = []
            boxes_2d = []
            for label in labels:
                if label['obj_type'] != modality:
                    continue
                psr = label['psr']
                p = np.array((psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]))
                s = np.array((psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]))
                r = np.array((psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]))
                R = euler_angle_to_rotation_matrix(r)
                corners3d = psr_to_corners3d(p, s, R)
                box2d = corners3d_to_box2d(corners3d, extrinsic, intrinsic)
                if box2d is None:
                    continue
                boxes_3d.append(np.concatenate([p, s, r]))
                boxes_2d.append(box2d)
                px, py, pz = p
                sx, sy, sz = s
                rx, ry, rz = r
                front_cars_file.write(f"{frame_id} {label['obj_id']} "
                                 f"{px}, {py}, {pz}, {sx}, {sy}, {sz}, {rx}, {ry}, {rz}\n")
            frame = {
                'frame_id': frame_id,
                'image_path': os.path.join(root_dir, "camera", "front", frame_id + ".jpg"),
                'point_path': os.path.join(root_dir, "lidar", frame_id + ".pcd"),
                'extrinsic': extrinsic,
                'intrinsic': intrinsic,
                'boxes_3d': np.array(boxes_3d),
                'boxes_2d': np.array(boxes_2d)
            }
            if len(boxes_3d) > 0 or allow_empty:
                seq_gt.append(frame)

    return seq_gt


def write_results(results, detections, out_file):
    for (tid, box, info, score) in results:
        frame_id = detections['frame_id']
        px, py, pz, sx, sy, sz, rx, ry, rz = box.flatten()
        x1, y1, x2, y2 = info['bbox']
        out_file.write(f"{frame_id} {tid} "
                       # f"{x1} {y1} {x2} {y2} "
                       f"{px}, {py}, {pz}, {sx}, {sy}, {sz}, {rx}, {ry}, {rz}"
                       f"{score}\n")
