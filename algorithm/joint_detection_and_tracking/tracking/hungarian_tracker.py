import numpy as np
import torch
from tracking.data_association import ortools_solve, hungarian_match
from tracking.kitti_track import Track

other_info_keys = ['name', 'truncated', 'occluded', 'alpha', 'bbox', 'location', 'dimensions', 'rotation_y']


class Tracker:
    def __init__(self, model, t_miss=4, t_hit=1, w_app=0.3, w_iou=0.35, w_loc=0.35):
        self.tracks = []
        self.t_miss = t_miss
        self.t_hit = t_hit
        self.w_app = w_app
        self.w_iou = w_iou
        self.w_motion = w_loc
        self.frame_count = 0
        self.model = model
        self.last_frame_idx = 0

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        self.last_frame_idx = 0
        self.model.eval()

    def track_management(self):
        idx = len(self.tracks)
        results = []
        for trk in reversed(self.tracks):
            if trk.hits >= self.t_hit or self.frame_count <= self.t_hit:
                if trk.misses == 0:
                    results.append(trk.get_data())
            idx -= 1
            # remove dead tracks
            if trk.misses >= self.t_miss:
                self.tracks.pop(idx)
        return results

    def update(self, frame_detections_dict):
        boxes_3d = frame_detections_dict['boxes_lidar']

        num_det = len(boxes_3d)
        num_pred = len(self.tracks)

        if num_det == 0:
            return []
        cur_frame_idx = int(frame_detections_dict['frame_id'])
        passed_frames = cur_frame_idx - self.last_frame_idx
        self.last_frame_idx = cur_frame_idx
        self.frame_count += passed_frames

        det_features = frame_detections_dict['feature'].cuda()
        det_scores = self.model.get_cls_scores(det_features)

        # for the first frame
        if num_pred == 0:
            # add in tracks
            for d in range(num_det):
                info = {k: frame_detections_dict[k][d] for k in other_info_keys}
                self.tracks.append(Track(boxes_3d[d], det_scores[d].item(), feature=det_features[d],
                                         info=info))
            return self.track_management()

        # get predictions of the current frame.
        pred_boxes = np.empty((num_pred, 7), dtype=np.float32)
        for i, trk in enumerate(self.tracks):
            box, score, feature = trk.predict(passed_frames)
            pred_boxes[i] = box

        matched, unmatched_dets, tentative_dets = hungarian_match(
            torch.from_numpy(boxes_3d).cuda(),
            torch.from_numpy(pred_boxes).cuda(),
            det_scores
        )
        # update matched tracks
        for t, d in matched:
            info = {k: frame_detections_dict[k][d] for k in other_info_keys}
            self.tracks[t].update_with_feature(boxes_3d[d],
                                               det_features[d],
                                               det_scores[d].item(),
                                               info)
        # init new tracks for unmatched detections
        for i in unmatched_dets:
            info = {k: frame_detections_dict[k][i] for k in other_info_keys}
            trk = Track(bbox=boxes_3d[i], feature=det_features[i],
                        score=det_scores[i].item(), info=info)
            self.tracks.append(trk)

        for i in tentative_dets:
            info = {k: frame_detections_dict[k][i] for k in other_info_keys}
            trk = Track(bbox=boxes_3d[i], feature=det_features[i],
                        score=det_scores[i].item(), info=info)
            trk.misses += 1
            self.tracks.append(trk)
        return self.track_management()
