import cv2
import numpy as np
import os
import colorsys
from utils.kitti_util import draw_projected_box3d, Calibration
from utils.box_util import compute_box_3d


def convert_data_to_object(data):
    # [type(str), truncation(float), occlusion(int),
    #  observation angle(rad), xmin, ymin, xmax, ymax,
    #  h, w, l, x, y, z, yaw angle]
    target = {}
    data[1:] = [float(x) for x in data[1:]]
    target['type'] = data[0]  # 'Car', 'Pedestrian', ...
    target['truncation'] = data[1]  # truncated pixel ratio [0..1]
    # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
    target['occlusion'] = int(data[2])
    target['alpha'] = data[3]  # object observation angle [-pi..pi]

    # extract 2d bounding box in 0-based coordinates
    target['xmin'] = int(data[4])  # left
    target['ymin'] = int(data[5])  # top
    target['xmax'] = int(data[6])  # right
    target['ymax'] = int(data[7])  # bottom
    target['box2d'] = np.array(
        [target['xmin'], target['ymin'], target['xmax'], target['ymax']])
    # extract 3d bounding box information
    target['h'] = data[8]  # box height
    target['w'] = data[9]  # box width
    target['l'] = data[10]  # box length (in meters)
    # location (x,y,z) in camera coord.
    target['center3d'] = (data[11], data[12], data[13])
    # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
    target['ry'] = data[14]
    
    return target


def project_3d_box_to_image(obj, calib):
    center = obj['center3d']
    bbox_3d = np.array([center[0], center[1], center[2],
                        obj['ry'], obj['l'], obj['w'], obj['h']])
    corners_3d = compute_box_3d(bbox_3d)

    # project the 3d bounding box (camera coordinate) into the image plane
    corners_2d = calib.project_rect_to_image(corners_3d).astype(int)
    return corners_2d


def save_image_with_2d_boxes(img_path, objects, save_path, display_id=True):
    img = cv2.imread(img_path)
    for obj in objects:
        text = 'ID: %d' % obj['id']
        box_2d = obj['box2d'].astype(int)
        if display_id:
            color = tuple([int(tmp * 255)
                           for tmp in colors[obj['id'] % max_color]])
            cv2.rectangle(img, (box_2d[0], box_2d[1]),
                          (box_2d[2], box_2d[3]), color, 2)
            img = cv2.putText(img, text, (box_2d[0], box_2d[1]),
                              cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color)
        else:
            cv2.rectangle(img, (box_2d[0], box_2d[1]),
                          (box_2d[2], box_2d[3]), (0, 255, 0), 2)
    img = cv2.resize(img, (1242, 374))
    cv2.imwrite(save_path, img)


def save_image_with_3d_boxes(img_path, objects, calib, save_path):
    img = cv2.imread(img_path)
    print(img_path)
    cv2.imshow("", img)
    cv2.waitKey(0)
    for obj in objects:
        box3d_pts_2d = project_3d_box_to_image(obj, calib)
        img = draw_projected_box3d(img, box3d_pts_2d)
    cv2.imwrite(save_path, img)


def visualization(frame):
    img_path = os.path.join(data_root, "image_2/%s.png" % frame)
    calib_file = os.path.join(data_root, 'calib/%s.txt' % frame)
    with open (os.path.join(result_dir, '%s.txt' % frame)) as f:
        results = f.readlines()
    save_image_with_3d_boxes(img_path,
                             [convert_data_to_object(line.split())
                              for line in results],
                             Calibration(calib_file),
                             os.path.join(vis_seq_dir, '%06s.png' % frame))


if __name__ == "__main__":
    # Generate random colors.
    # To get visually distinct colors, generate them in HSV space then convert to RGB.
    max_color = 30
    hsv = [(i / float(max_color), 1, 1) for i in range(max_color)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    # result_root = "results"
    result_dir = "/home/kemo/Github/EMOT/algorithm/detection/second/pretrained_models_v1.5/car_onestage/eval_results/step_27855"
    vis_seq_dir = result_dir
    data_root = '/home/kemo/Dataset/sustechscapes-mini-dataset/kitti/training'
    visualization("000000")
