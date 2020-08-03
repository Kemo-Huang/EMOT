import numpy as np
import torch
from tracking.data_association import ortools_solve
from tracking.kitti_track import Track
from appearance_model.tracking_net import TrackingNet


class Tracker:
    def __init__(self, ckpt, t_miss=4, t_hit=1, w_app=0.3, w_iou=0.35, w_loc=0.35):
        self.tracks = []
        self.t_miss = t_miss
        self.t_hit = t_hit
        self.w_app = w_app
        self.w_iou = w_iou
        self.w_motion = w_loc
        self.frame_count = 0
        model = TrackingNet().cuda()
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
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

    def update(self, det_boxes, images, points, points_split, detections: dict):
        cur_frame_idx = int(detections['frame_idx'])
        passed_frames = cur_frame_idx - self.last_frame_idx
        self.last_frame_idx = cur_frame_idx
        self.frame_count += passed_frames
        num_det = len(det_boxes)
        num_pred = len(self.tracks)
        alpha = detections['alpha']
        bbox = detections['bbox']  # 2d bounding box
        # for the first frame
        if num_pred == 0:
            det_lens = []
            for i in range(num_det):
                det_lens.append(points_split[i + 1] - points_split[i])
            det_scores, det_features = self.model(images, points, det_lens)
            # add in tracks
            for d in range(num_det):
                self.tracks.append(Track(det_boxes[d], det_scores[d], feature=det_features[:, :, d],
                                         info={'alpha': alpha[d], 'bbox': bbox[d]}))
            return self.track_management()
        # get predictions of the current frame.
        pred_boxes = np.empty((num_pred, 7))
        pred_scores = torch.empty(num_pred)
        pred_features = torch.empty((3, 512, num_pred))
        for i, trk in enumerate(self.tracks):
            box, score, feature = trk.predict(passed_frames)
            pred_boxes[i] = box
            pred_features[:, :, i] = feature
            pred_scores[i] = score
        pred_scores = pred_scores.cuda()
        pred_features = pred_features.cuda()
        # ===========================
        det_lens = []
        for i in range(num_det):
            det_lens.append(points_split[i + 1] - points_split[i])
        det_scores, det_features = self.model(images, points, det_lens)
        link_scores, new_scores, end_scores = self.model.scoring(pred_features, det_features)
        matched, unmatched_dets, tentative_dets = ortools_solve(
            det_boxes,
            pred_boxes,
            torch.cat((pred_scores, det_scores)),
            link_scores,
            new_scores,
            end_scores,
            w_app=self.w_app,
            w_iou=self.w_iou,
            w_motion=self.w_motion
        )
        # update matched tracks
        for t, d in matched:
            self.tracks[t].update_with_feature(det_boxes[d],
                                               det_features[:, :, d],
                                               det_scores[d],
                                               {'alpha': alpha[d], 'bbox': bbox[d]})
        # init new tracks for unmatched detections
        for i in unmatched_dets:
            trk = Track(bbox=det_boxes[i], feature=det_features[:, :, i],
                        score=det_scores[i], info={'alpha': alpha[i], 'bbox': bbox[i]})
            self.tracks.append(trk)

        for i in tentative_dets:
            trk = Track(bbox=det_boxes[i], feature=det_features[:, :, i],
                        score=det_scores[i], info={'alpha': alpha[i], 'bbox': bbox[i]})
            trk.misses += 1
            self.tracks.append(trk)
        return self.track_management()
