import numpy as np
from numba import njit
from scipy.spatial import ConvexHull


@njit
def box_to_corners_kitti(bbox3d):
    """
    Computes the rotation and translation of the bounding box.
    :param bbox3d: [x,y,z,theta,l,w,h]
    :return: a numpy array (8,3) - Eight corners of the bounding box

        y  x
       | /
       o  --- z

        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7

    """
    # Rotation about the y-axis
    theta = bbox3d[3]
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, 0., s],
                  [0., 1., 0.],
                  [-s, 0., c]])

    # 3d bounding box dimensions
    l = bbox3d[4]
    w = bbox3d[5]
    h = bbox3d[6]

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.array((x_corners, y_corners, z_corners)))
    corners_3d[0, :] += bbox3d[0]
    corners_3d[1, :] += bbox3d[1]
    corners_3d[2, :] += bbox3d[2]

    return np.transpose(corners_3d)


@njit
def euler_angle_to_rotation_matrix(r):
    # Calculate rotation about x axis
    R_x = np.array([
        [1., 0., 0.],
        [0., np.cos(r[0]), -np.sin(r[0])],
        [0., np.sin(r[0]), np.cos(r[0])]
    ])

    # Calculate rotation about y axis
    R_y = np.array([
        [np.cos(r[1]), 0., np.sin(r[1])],
        [0., 1., 0.],
        [-np.sin(r[1]), 0., np.cos(r[1])]
    ])

    # Calculate rotation about z axis
    R_z = np.array([
        [np.cos(r[2]), -np.sin(r[2]), 0.],
        [np.sin(r[2]), np.cos(r[2]), 0.],
        [0., 0., 1.]
    ])

    R = np.dot(R_x, np.dot(R_y, R_z))
    return R


@njit
def psr_to_corners3d(p, s, R) -> np.ndarray:
    """
             z  x
             | /
        y -- o

        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7

    """
    x = s[0] / 2
    y = s[1] / 2
    z = s[2] / 2

    local_coord = np.array([
        x, -y, z,  # front-right-top
        x, y, z,  # front-left-top
        -x, y, z,  # rear-left-top
        -x, -y, z,  # rear-right-top
        x, -y, -z,  # front-right-bottom
        x, y, -z,  # front-left-bottom
        -x, y, -z,  # rear-left-bottom
        -x, -y, -z,  # rear-right-bottom
    ]).reshape((-1, 3))

    world_coord = np.dot(R, np.transpose(local_coord)) + p.reshape((3, 1))
    return world_coord


@njit
def box3d_vol(corners):
    a = np.linalg.norm(corners[0, :] - corners[1, :])
    b = np.linalg.norm(corners[0, :] - corners[3, :])
    c = np.linalg.norm(corners[0, :] - corners[4, :])
    return a * b * c


def polygon_clip(subject_polygon, clip_polygon):
    """ Clip a polygon with another polygon.
    Args:
      subject_polygon: a list of (x,y) 2d points, any polygon.
      clip_polygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """
    output_list = subject_polygon
    cp1 = clip_polygon[-1]

    for clip_vertex in clip_polygon:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []
        s = input_list[-1]

        for subject_vertex in input_list:
            e = subject_vertex
            inside_e = (cp2[0] - cp1[0]) * (e[1] - cp1[1]) > (cp2[1] - cp1[1]) * (e[0] - cp1[0])
            inside_s = (cp2[0] - cp1[0]) * (s[1] - cp1[1]) > (cp2[1] - cp1[1]) * (s[0] - cp1[0])

            if (inside_e and not inside_s) or (inside_s and not inside_e):
                dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
                dp = [s[0] - e[0], s[1] - e[1]]
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = s[0] * e[1] - s[1] * e[0]
                n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
                output_list.append([(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3])
            if inside_e:
                output_list.append(e)
            s = e
        cp1 = cp2
        if len(output_list) == 0:
            return None
    return output_list


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        return ConvexHull(inter_p).volume
    else:
        return 0.0


def iou3d(box_3d_1, box_3d_2, criterion='union'):
    """ Compute 3D bounding box IoU.
    """
    corners1 = box_to_corners_kitti(box_3d_1)
    corners2 = box_to_corners_kitti(box_3d_2)
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    inter_area = convex_hull_intersection(rect1, rect2)
    ymax = min(corners1[0, 1], corners2[0, 1])
    ymin = max(corners1[4, 1], corners2[4, 1])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    if criterion.lower() == 'union':
        iou = inter_vol / (vol1 + vol2 - inter_vol)
    elif criterion.lower() == 'a':
        iou = inter_vol / vol1
    else:
        raise TypeError("Unknown type for criterion")
    return iou


@njit
def iou2d(a, b, criterion="union"):
    """
        Computes intersection over union for bbox a and b in KITTI format.
        If the criterion is 'union', overlap = (a inter b) / a union b).
        If the criterion is 'a', overlap = (a inter b) / a, where b should be a don't-care area.
    """

    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    w = x2 - x1
    h = y2 - y1

    if w <= 0. or h <= 0.:
        return 0.
    inter = w * h
    a_area = (a.x2 - a.x1) * (a.y2 - a.y1)
    b_area = (b.x2 - b.x1) * (b.y2 - b.y1)
    # intersection over union overlap
    if criterion.lower() == "union":
        o = inter / float(a_area + b_area - inter)
    elif criterion.lower() == "a":
        o = float(inter) / float(a_area)
    else:
        raise TypeError("Unknown type for criterion")
    return o


def distance_iou_kitti(box1, box2):
    corners1 = box_to_corners_kitti(box1)
    corners2 = box_to_corners_kitti(box2)
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 1]) for i in range(3, -1, -1)]
    inter_area = convex_hull_intersection(rect1, rect2)
    x_max1, y_max1, z_max1 = np.max(corners1, axis=0)
    x_max2, y_max2, z_max2 = np.max(corners2, axis=0)
    x_min1, y_min1, z_min1 = np.min(corners1, axis=0)
    x_min2, y_min2, z_min2 = np.min(corners2, axis=0)
    # iou
    inter_vol = inter_area * max(0.0, min(y_max1, y_max2) - max(y_min1, y_min2))
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    # distance
    p_max = np.array([max(x_max1, x_max2), max(y_max1, y_max2), max(z_max1, z_max2)], dtype=float)
    p_min = np.array([min(x_min1, x_min2), min(y_min1, y_min2), min(z_min1, z_min2)], dtype=float)
    distance = np.linalg.norm(box1[:3] - box2[:3]) / np.linalg.norm(p_max - p_min)
    return iou, 1 - distance


from shapely.geometry import Polygon


# @njit
def distance_iou_sustech(psr1, psr2):
    p1 = psr1[:3]
    p2 = psr2[:3]
    s1 = psr1[3:6]
    s2 = psr2[3:6]
    R1 = euler_angle_to_rotation_matrix(psr1[6:])
    R2 = euler_angle_to_rotation_matrix(psr2[6:])
    corners1 = np.transpose(psr_to_corners3d(p1, s1, R1))  # 8 * 3
    corners2 = np.transpose(psr_to_corners3d(p2, s2, R2))
    # # corner points are in counter clockwise order
    # rect1 = [(corners1[i, 0], corners1[i, 2]) for i in range(3, -1, -1)]
    # rect2 = [(corners2[i, 0], corners2[i, 2]) for i in range(3, -1, -1)]
    # inter_area = convex_hull_intersection(rect1, rect2)

    rect1 = Polygon(corners1[:4, :2])
    rect2 = Polygon(corners2[:4, :2])
    inter_area = rect1.intersection(rect2).area
    print(inter_area)

    x_max1, y_max1, z_max1 = np.max(corners1, axis=0)
    x_max2, y_max2, z_max2 = np.max(corners2, axis=0)
    x_min1, y_min1, z_min1 = np.min(corners1, axis=0)
    x_min2, y_min2, z_min2 = np.min(corners2, axis=0)
    # # iou
    inter_vol = inter_area * max(0.0, min(y_max1, y_max2) - max(y_min1, y_min2))
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    # distance
    p_max = np.array([max(x_max1, x_max2), max(y_max1, y_max2), max(z_max1, z_max2)], dtype=float)
    p_min = np.array([min(x_min1, x_min2), min(y_min1, y_min2), min(z_min1, z_min2)], dtype=float)
    distance = np.linalg.norm(p1 - p2) / np.linalg.norm(p_max - p_min)
    return iou, 1 - distance
