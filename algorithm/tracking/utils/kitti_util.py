import csv
import os
import pickle
import numpy as np
import pyproj
import torch
import motmetrics
from collections import OrderedDict


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def read_calib_file(filepath, extend_matrix=True):
    """ Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    calibration = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(info)
                   for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
    P1 = np.array([float(info)
                   for info in lines[1].split(' ')[1:13]]).reshape([3, 4])
    P2 = np.array([float(info)
                   for info in lines[2].split(' ')[1:13]]).reshape([3, 4])
    P3 = np.array([float(info)
                   for info in lines[3].split(' ')[1:13]]).reshape([3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    calibration['P0'] = P0
    calibration['P1'] = P1
    calibration['P2'] = P2
    calibration['P3'] = P3
    R0_rect = np.array([float(info)
                        for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect
    calibration['R0_rect'] = rect_4x4
    Tr_velo_to_cam = np.array(
        [float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
    Tr_imu_to_velo = np.array(
        [float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    calibration['Tr_velo_to_cam'] = Tr_velo_to_cam
    calibration['Tr_imu_to_velo'] = Tr_imu_to_velo
    return calibration


class Calibration(object):
    """ Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    """

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = read_calib_file(calib_filepath, extend_matrix=False)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        if calibs.__contains__(('Tr_velo_to_cam')):
            self.V2C = calibs['Tr_velo_to_cam']
        else:
            self.V2C = calibs['Tr_velo_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        if calibs.__contains__('R0_rect'):
            self.R0 = calibs['R0_rect']
        else:
            self.R0 = calibs['R_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_from_video(self, calib_root_dir):
        """ Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        """
        # TODO: Need modified since read_calib_file function has been changed
        data = {}
        cam2cam = read_calib_file(
            os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(
            os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(
            np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)
        # x1 = min(x1, proj.image_width)
        y0 = max(0, y0)
        # y1 = min(y1, proj.image_height)
        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: 4 points in image2 coord.
        '''
        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = (
                    (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = (
                    (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


LABEL = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': -1
}

LABEL_VERSE = {v: k for k, v in LABEL.items()}


def kitti_result_line(result_dict, precision=4):
    prec_float = "{" + ":.{}f".format(precision) + "}"
    res_line = []
    all_field_default = OrderedDict([
        ('frame', None),
        ('id', None),
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError("you must specify a value for {}".format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'frame':
            res_line.append(str(val))
        elif key == 'id':
            res_line.append(str(val))
        elif key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError("unknown key. supported key:{}".format(
                res_dict.keys()))
    return ' '.join(res_line)


def write_kitti_result(root,
                       seq_name,
                       step,
                       frames_id,
                       frames_det,
                       part='train'):
    result_lines = []
    # print(frames_id)
    # print(frames_det)
    assert len(frames_id) == len(frames_det)
    for i in range(len(frames_id)):
        if frames_det[i]['id'].size(0) == 0:
            continue
        frames_det[i]['dimensions'] = frames_det[i]['dimensions'][:, [1, 2, 0]]
        # lhw->hwl(change to label file format)
        for j in range(frames_det[i]['id'].size(0)):
            # assert frames_det[i]['id'][j] == frames_id[i][j]
            try:
                if frames_det[i]['id'][j] != frames_id[i][j]:
                    print(frames_det[i]['id'])
                    print(frames_id[i])
            except:  # noqa
                print(frames_det[i]['id'])
                print(frames_id[i])
            result_dict = {
                'frame': int(frames_det[i]['frame_idx'][0]),
                'id': frames_id[i][j],
                'name': LABEL_VERSE[frames_det[i]['name'][j].item()],
                'truncated': frames_det[i]['truncated'][j].item(),
                'occluded': frames_det[i]['occluded'][j].item(),
                'alpha': frames_det[i]['alpha'][j].item(),
                'bbox': frames_det[i]['bbox'][j].numpy(),
                'location': frames_det[i]['location'][j].numpy(),
                'dimensions': frames_det[i]['dimensions'][j].numpy(),
                'rotation_y': frames_det[i]['rotation_y'][j].item(),
                'score': 0.9,
            }
            result_line = kitti_result_line(result_dict)
            result_lines.append(result_line)

    path = f"{root}/{step}/{part}"
    if not os.path.exists(path):
        print("Make directory: " + path)
        os.makedirs(path)
    filename = f"{path}/{seq_name}.txt"
    result_str = '\n'.join(result_lines)
    with open(filename, 'w') as f:
        f.write(result_str)


# The following code defines the basic data structures to be use
def get_start_gt_anno():
    return {'id': [],
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': [],
            }


def get_frame_det_points():
    return {
        'rot': [],
        'loc': [],
        'dim': [],
        'points': [],
        'det_lens': [],
        'info_id': []
    }


def get_empty_det(img_frame_id):
    dets = {
        'frame_idx': img_frame_id,
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'image_idx': []
    }
    return dets


def get_frame(img_seq_id, img_frame_id, dets, frame_info):
    id_path = f'{img_seq_id}-{img_frame_id}'
    return {
        'seq_id': img_seq_id,
        'frame_id': img_frame_id,
        'image_id': id_path,
        'point_path': f'{id_path}.bin',
        'image_path': f'{img_seq_id}/{img_frame_id}.png',
        'frame_info': frame_info,
        'detection': dets,
    }


def get_frame_info(seq_id, frame_id, seq_calib, pos, rad):
    return {
        'info_id': f'{seq_id}-{frame_id}',
        'calib/R0_rect': seq_calib['R0_rect'],
        'calib/Tr_velo_to_cam': seq_calib['Tr_velo_to_cam'],
        'calib/Tr_imu_to_velo': seq_calib['Tr_imu_to_velo'],
        'calib/P2': seq_calib['P2'],
        'pos': pos,
        'rad': rad,
    }


def generate_seq_dets(root_dir,
                      link_file,
                      det_file,
                      iou_threshold=0.2,
                      fix_threshold=2,
                      allow_empty=False):
    assert os.path.exists(det_file)
    print("Building dataset using dets file {}".format(det_file))

    with open(link_file) as f:
        lines = f.readlines()
    with open(det_file, 'rb') as f:
        detections = pickle.load(f)
    has_det = False
    count = 0
    total = 0
    obj_count = 0
    sequence_det = {}
    oxts_seq = {}
    calib = {}
    prev_dets = None
    prev_seq_id = -1
    add_count = 0
    add_frame = 0
    for i in range(21):
        seq_id = f'{i:04d}'
        with open(f"{root_dir}/oxts/{seq_id}.txt") as f_oxts:
            oxts_seq[seq_id] = f_oxts.readlines()
        calib[seq_id] = read_calib_file(f"{root_dir}/calib/{seq_id}.txt")

    for line in lines:
        id_path = line.strip()
        img_seq_id = id_path.split('-')[0]
        img_frame_id = id_path.split('-')[1]

        curr_seq_id = int(img_seq_id)
        if curr_seq_id != prev_seq_id:
            prev_dets = None
            prev_seq_id = curr_seq_id

        for x in detections:
            if len(x['image_idx']) == 0:
                continue
            elif x['image_idx'][0] == id_path:
                dets = x
                has_det = True
                break
        pos, rad = get_pos(oxts_seq[img_seq_id], int(img_frame_id))
        frame_info = get_frame_info(img_seq_id, img_frame_id,
                                    calib[img_seq_id], pos, rad)
        if has_det:
            # import pdb; pdb.set_trace()
            dets['frame_idx'] = img_frame_id
            dets['name'] = np.array(
                [LABEL[dets['name'][i]] for i in range(len(dets['name']))])
            dets['fix_count'] = np.zeros((len(dets['name']),))

            curr_dets, add_num = add_miss_dets(
                prev_dets,
                dets,
                iou_threshold=iou_threshold,
                fix_threshold=fix_threshold)  # add missed dets
            add_count += add_num
            add_frame += int(add_num > 0)

            frame = get_frame(img_seq_id, img_frame_id, curr_dets, frame_info)
            if img_seq_id in sequence_det:
                # sequence_det[img_seq_id][img_frame_id] = frame
                sequence_det[img_seq_id].update({img_frame_id: frame})
            else:
                # sequence_det[img_seq_id] = {img_frame_id: frame}
                sequence_det[img_seq_id] = OrderedDict([(img_frame_id, frame)])
            count = count + 1
            obj_count += len(x['name'])

            prev_dets = curr_dets

        elif allow_empty:
            dets = get_empty_det(img_frame_id)
            frame = get_frame(img_seq_id, img_frame_id, dets, frame_info)
            if img_seq_id in sequence_det:
                # sequence_det[img_seq_id][img_frame_id] = frame
                sequence_det[img_seq_id].update({img_frame_id: frame})
            else:
                # sequence_det[img_seq_id] = {img_frame_id: frame}
                sequence_det[img_seq_id] = OrderedDict([(img_frame_id, frame)])

        total = total + 1
        has_det = False

    print(f"Detect [{obj_count:6d}] cars in [{count}/{total}] images")
    print(f"Add [{add_count}] cars in [{add_frame}/{total}] images")
    return sequence_det


def generate_seq_gts(root_dir, seq_ids, sequence_det):
    sequence_gt = {}
    total = 0
    oxts_seq = {}
    calib = {}

    for seq_id in seq_ids:
        sequence_gt[seq_id] = []
        with open(f"{root_dir}/oxts/{seq_id}.txt") as f_oxts:
            oxts_seq[seq_id] = f_oxts.readlines()
        calib[seq_id] = read_calib_file(f"{root_dir}/calib/{seq_id}.txt")
        with open(f"{root_dir}/label_02/{seq_id}.txt") as f:
            f_csv = csv.reader(f, delimiter=' ')

            gt_det = None
            for row in f_csv:
                total += 1
                frame_id = int(row[0])

                if gt_det is None:
                    prev_id = frame_id
                    gt_det = get_start_gt_anno()
                obj_id = int(row[1])
                label = row[2]
                # if label == 'DontCare':
                # if label != modality:
                #     continue
                truncated = float(row[3])
                occluded = int(row[4])
                alpha = float(row[5])
                bbox = [x for x in map(float, row[6:10])]
                dimensions = [x for x in map(float, row[10:13])]
                location = [x for x in map(float, row[13:16])]
                rotation_y = float(row[16])

                if prev_id != frame_id and len(
                        gt_det['id']) > 0:  # frame switch during the sequence
                    if sequence_det[seq_id].__contains__(f"{prev_id:06d}"):
                        for k, v in gt_det.items():
                            gt_det[k] = np.array(v)
                        gt_det['frame_idx'] = f"{prev_id:06d}"
                        gt_det['dimensions'] = gt_det['dimensions'][:, [2, 0, 1]]
                        # From original hwl-> lhw
                        pos, rad = get_pos(oxts_seq[seq_id], int(prev_id))
                        frame_info = get_frame_info(seq_id, prev_id,
                                                    calib[seq_id], pos, rad)
                        frame = get_frame(seq_id, f"{prev_id:06d}", gt_det,
                                          frame_info)
                        sequence_gt[seq_id].append(frame)

                    gt_det = get_start_gt_anno()

                gt_det['id'].append(obj_id)
                gt_det['name'].append(LABEL[label])
                gt_det['truncated'].append(truncated)
                gt_det['occluded'].append(occluded)
                gt_det['alpha'].append(alpha)
                gt_det['bbox'].append(bbox)
                gt_det['dimensions'].append(dimensions)
                gt_det['location'].append(location)
                gt_det['rotation_y'].append(rotation_y)

                prev_id = frame_id

            # Load the last frame at the end of the sequence
            if sequence_det[seq_id].__contains__(f"{prev_id:06d}") and len(
                    gt_det['id']) > 0:
                for k, v in gt_det.items():
                    gt_det[k] = np.array(v)
                gt_det['frame_idx'] = f"{prev_id:06d}"
                pos, rad = get_pos(oxts_seq[seq_id], int(prev_id))
                frame_info = get_frame_info(seq_id, prev_id, calib[seq_id],
                                            pos, rad)
                frame = get_frame(seq_id, f"{prev_id:06d}", gt_det, frame_info)
                sequence_gt[seq_id].append(frame)

        assert len(sequence_gt[seq_id]) == len(sequence_det[seq_id])

    return sequence_gt


def get_pos(oxts_seq, id):
    oxt = oxts_seq[id].strip().split(' ')
    lat = float(oxt[0])
    lon = float(oxt[1])
    alt = float(oxt[2])
    pos_x, pos_y = proj_trans(lon, lat)
    pos = np.array([pos_x, pos_y, alt])
    rad = np.array([x for x in map(float, oxt[3:6])])
    return pos, rad


# wgs84->utm，
def proj_trans(lon, lat):
    # p3 = pyproj.Proj("epsg:4326")
    p1 = pyproj.Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=False)
    p2 = pyproj.Proj("epsg:3857")
    x1, y1 = p1(lon, lat)
    x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
    return x2, y2


def add_miss_dets(prev_dets, dets, iou_threshold=0.2, fix_threshold=2):
    if prev_dets is None:
        return dets, 0
    distance = calculate_distance(
        dets['bbox'], prev_dets['bbox'], max_iou=iou_threshold)  # NxM
    mat = distance.copy(
    )  # the smaller the value, the close between det and gt
    mask_nan = np.isnan(mat)
    mask_val = np.isnan(mat) == False  # noqa
    mat[mask_val] = 1  # just set it to 1 if it has value not nan
    mat[mask_nan] = 0
    fix_count = torch.Tensor(prev_dets['fix_count'])
    mask = torch.Tensor(mat).sum(dim=-1).eq(0)
    fix_count += mask.float()
    mask ^= fix_count.gt(fix_threshold)
    index = mask.nonzero().squeeze(0).numpy()

    if len(index) == 0:
        return dets, 0
    for k, v in prev_dets.items():
        if k == 'frame_idx':
            continue
        # select_v = np.take(prev_dets[k], indices=index, axis=0)
        select_v = prev_dets[k][index]
        if k == 'fix_count':
            select_v += 1
        if len(select_v.shape) >= 2 and select_v.shape[1] == 1:
            select_v = select_v.squeeze(1)
        dets[k] = np.concatenate([dets[k], select_v], axis=0)

    return dets, len(index)


def calculate_distance(dets, gt_dets, max_iou):
    # dets format: X1, Y1, X2, Y2
    # distance input format: X1, Y1, W, H
    # for i in range(len(dets)):
    det = dets.copy()
    det[:, 2:] = det[:, 2:] - det[:, :2]
    gt_det = gt_dets.copy()
    gt_det[:, 2:] = gt_det[:, 2:] - gt_det[:, :2]
    return motmetrics.distances.iou_matrix(gt_det, det, max_iou=max_iou)


def write_kitti_format(results, frame_id, out_file):
    frame_id = frame_id.lstrip('0')
    if len(frame_id) == 0:
        frame_id = '0'
    for (tid, box_3d, box_2d, score) in results:
        x, y, z, yaw, l, w, h = box_3d.flatten()
        x1, y1, x2, y2 = box_2d
        out_file.write(f"{frame_id} {tid} Car 0 0 0 "
                       f"{x1} {y1} {x2} {y2} "
                       f"{h} {w} {l} {x} {y} {z} "
                       f"{yaw} "
                       f"{score}\n")
