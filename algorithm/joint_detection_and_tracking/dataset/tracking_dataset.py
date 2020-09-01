import pickle
import torch


def get_sequence_data(feature_pkl: str, result_pkl: str, link_pkl: str):
    """
    The dataset used for testing mot
    Args:
        feature_pkl (str): [tensor (N, 27648), ...]
        result_pkl (str): detection annotation list
        link_pkl (str): map sample_id to sequence frame
    """
    with open(feature_pkl, 'rb') as f:
        all_features = pickle.load(f)
    with open(result_pkl, 'rb') as f:
        det_results = pickle.load(f)
    with open(link_pkl, 'rb') as f:
        link_dict = pickle.load(f)

    assert len(all_features) == len(det_results) == len(link_dict)

    n_samples = len(all_features)
    sequence = {}
    for idx in range(n_samples):
        cur_features = all_features[idx]
        if cur_features is None:
            cur_features = torch.zeros((0, 27648))
        cur_detection = det_results[idx]
        assert len(cur_detection['name']) == len(cur_features)
        cur_detection['feature'] = cur_features.squeeze(-1)
        cur_frame = cur_detection['frame_id']
        real_seq, real_frame = link_dict[cur_frame]

        if real_seq not in sequence:
            sequence[real_seq] = {real_frame: cur_detection}
        else:
            sequence[real_seq][real_frame] = cur_detection

    return sequence
