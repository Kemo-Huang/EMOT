from utils.box_util import distance_iou_kitti, distance_iou_sustech
import numpy as np
from ortools.linear_solver import pywraplp


def ortools_solve(det_boxes,
                  pred_boxes,
                  det_score,
                  link_score,
                  new_score,
                  end_score,
                  w_app,
                  w_iou,
                  w_motion,
                  kitti=True):
    num_det = len(det_boxes)
    num_pred = len(pred_boxes)
    # IoU cost
    iou_matrix = np.zeros((num_pred, num_det))
    motion_matrix = np.zeros((num_pred, num_det))
    for t, trk in enumerate(pred_boxes):
        for d, det in enumerate(det_boxes):
            iou, motion = distance_iou_kitti(trk, det) if kitti else distance_iou_sustech(trk, det)
            iou_matrix[t, d] = iou
            motion_matrix[t, d] = motion
    solver = pywraplp.Solver('SolveAssignmentProblemMIP',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    y_det = {}
    y_new = {}
    y_end = {}
    y_link = {}
    for i in range(det_score.size(0)):
        y_det[i] = solver.BoolVar('y_det[%i]' % i)
        y_new[i] = solver.BoolVar('y_new[%i]' % i)
        y_end[i] = solver.BoolVar('y_end[%i]' % i)
    w_link_y = []
    for j in range(num_pred):
        y_link[j] = {}
        for k in range(num_det):
            y_link[j][k] = solver.BoolVar(f'y_link[{j}, {k}]')
            w_link_y.append(y_link[j][k] * (
                    link_score[j][k].item() * w_app +
                    iou_matrix[j, k] * w_iou +
                    motion_matrix[j, k] * w_motion
            ))
    w_det_y = [y_det[i] * det_score[i].item() for i in range(det_score.size(0))]
    w_new_y = [y_new[i] * new_score[i].item() for i in range(det_score.size(0))]
    w_end_y = [y_end[i] * end_score[i].item() for i in range(det_score.size(0))]

    # Objective
    solver.Maximize(solver.Sum(w_det_y + w_new_y + w_end_y + w_link_y))

    # Constraints
    for j in range(num_pred):
        det_idx = j
        # pred = link + end
        solver.Add(
            solver.Sum([y_end[det_idx], (-1) * y_det[det_idx]] +
                       [y_link[j][k] for k in range(num_det)]) == 0)
        solver.Add(
            solver.Sum([y_new[det_idx], (-1) * y_det[det_idx]]) == 0)

    for k in range(num_det):
        det_idx = num_pred + k
        # det = link + start
        solver.Add(
            solver.Sum([y_new[det_idx], (-1) * y_det[det_idx]] +
                       [y_link[j][k] for j in range(num_pred)]) == 0)
        solver.Add(
            solver.Sum([y_end[det_idx], (-1) * y_det[det_idx]]) == 0)

    solver.Solve()

    assign_det = det_score.new_zeros(det_score.size())
    assign_new = det_score.new_zeros(det_score.size())
    assign_end = det_score.new_zeros(det_score.size())
    assign_link = det_score.new_zeros(link_score.size())

    for j in range(num_pred):
        for k in range(num_det):
            assign_link[j][k] = y_link[j][k].solution_value()

    for i in range(len(det_score)):
        assign_det[i] = y_det[i].solution_value()
        assign_new[i] = y_new[i].solution_value()
        assign_end[i] = y_end[i].solution_value()

    matched = assign_link.nonzero().tolist()
    unmatched_detections = assign_new[num_pred:].nonzero().flatten().tolist()
    tentative_detections = (assign_det[num_pred:] == 0).nonzero().flatten().tolist()

    return matched, unmatched_detections, tentative_detections
