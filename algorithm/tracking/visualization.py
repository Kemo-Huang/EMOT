import cv2
import numpy as np
import os
import colorsys
from utils.kitti_util import Calibration
from utils.box_util import box_to_corners_kitti


def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color,
                 thickness)
    return image


def convert_data_to_object(data):
    # [frame, id, type(str), truncation(float), occlusion(int),
    #  observation angle(rad), xmin, ymin, xmax, ymax,
    #  h, w, l, x, y, z, yaw angle, score]
    target = {}
    data[3:] = [float(x) for x in data[3:]]
    target['id'] = int(data[1])
    target['type'] = data[2]  # 'Car', 'Pedestrian', ...
    target['truncation'] = data[3]  # truncated pixel ratio [0..1]
    target['occlusion'] = int(data[4])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
    target['alpha'] = data[5]  # object observation angle [-pi..pi]

    # extract 2d bounding box in 0-based coordinates
    target['xmin'] = data[6]  # left
    target['ymin'] = data[7]  # top
    target['xmax'] = data[8]  # right
    target['ymax'] = data[9]  # bottom
    target['box2d'] = np.array([target['xmin'], target['ymin'], target['xmax'], target['ymax']])
    # extract 3d bounding box information
    target['h'] = data[10]  # box height
    target['w'] = data[11]  # box width
    target['l'] = data[12]  # box length (in meters)
    target['center3d'] = (data[13], data[14], data[15])  # location (x,y,z) in camera coord.
    target['ry'] = data[16]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
    if len(data) > 17:
        target['score'] = float(data[17])
    return target


def project_3d_box_to_image(obj, calib):
    center = obj['center3d']
    bbox_3d = np.array([center[0], center[1], center[2], obj['ry'], obj['l'], obj['w'], obj['h']])
    corners_3d = box_to_corners_kitti(bbox_3d)

    # project the 3d bounding box (camera coordinate) into the image plane
    corners_2d = calib.project_rect_to_image(corners_3d).astype(int)
    return corners_2d


def save_image_with_2d_boxes(img_path, objects, save_path, display_id=True):
    img = cv2.imread(img_path)
    for obj in objects:
        text = 'ID: %d' % obj['id']
        box_2d = obj['box2d'].astype(int)
        if display_id:
            color = tuple([int(tmp * 255) for tmp in colors[obj['id'] % max_color]])
            cv2.rectangle(img, (box_2d[0], box_2d[1]), (box_2d[2], box_2d[3]), color, 2)
            img = cv2.putText(img, text, (box_2d[0], box_2d[1]),
                              cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color)
        else:
            cv2.rectangle(img, (box_2d[0], box_2d[1]), (box_2d[2], box_2d[3]), (0, 255, 0), 2)
    img = cv2.resize(img, (1242, 374))
    cv2.imwrite(save_path, img)


def save_image_with_3d_boxes(img_path, objects, calib, save_path, display_id=True):
    img = cv2.imread(img_path)
    for obj in objects:
        text = 'ID: %d' % obj['id']
        box3d_pts_2d = project_3d_box_to_image(obj, calib)
        if display_id:
            color = tuple([int(tmp * 255) for tmp in colors[obj['id'] % max_color]])
            img = draw_projected_box3d(img, box3d_pts_2d, color=color)
            img = cv2.putText(img, text, (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]) - 8),
                              cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color)
        else:
            img = draw_projected_box3d(img, box3d_pts_2d)
    img = cv2.resize(img, (1242, 374))
    cv2.imwrite(save_path, img)


def visualization(display_id=True):
    for seq in VALID_SEQ_ID:
        image_dir = os.path.join(data_root, "image_02/%s" % seq)
        calib_file = os.path.join(data_root, 'calib/%s.txt' % seq)
        result_dir = os.path.join(result_root, result_sha, part)
        vis_seq_dir = os.path.join(result_root, 'visualization', part, seq)
        if not os.path.exists(vis_seq_dir):
            os.makedirs(vis_seq_dir)
        image_file_list = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
        num_frames = len(image_file_list)
        print('number of images to visualize is %d' % num_frames)
        results = [[] for _ in range(num_frames)]
        with open(os.path.join(result_dir, '%s.txt' % seq)) as f:
            lines = f.readlines()
            for line in lines:
                split_line = line.split()
                results[int(split_line[0])].append(split_line)
        print(f"processing sequence: {seq}")
        for frame, img_path in enumerate(image_file_list):
            # save_image_with_2d_boxes(img_path,
            #                          [convert_data_to_object(split_line) for split_line in results[frame]],
            #                          os.path.join(vis_seq_dir, '%06d.jpg' % frame))
            save_image_with_3d_boxes(img_path,
                                     [convert_data_to_object(split_line) for split_line in results[frame]],
                                     Calibration(calib_file),
                                     os.path.join(vis_seq_dir, '%06d.jpg' % frame),
                                     display_id)


if __name__ == "__main__":
    VALID_SEQ_ID = ['0005', '0007', '0017', '0011', '0002',
                    '0014', '0000', '0010', '0016', '0019', '0018']
    # Generate random colors.
    # To get visually distinct colors, generate them in HSV space then convert to RGB.
    max_color = 30
    hsv = [(i / float(max_color), 1, 1) for i in range(max_color)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

    # result_root = "results"
    result_root = os.path.dirname(os.path.realpath(__file__))
    result_sha = "data"
    part = 'val'
    data_root = '/home/kemo/Kitti/tracking/training'
    visualization(False)
