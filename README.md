# Multi-Object Tracking with Multi-Modality Appearance and Bayesian Motion for Autonomous Driving

Evaluation results on KITTI tracking benchmark:

| Methods | MOTA       | MOTP       | FP      | FN       | IDS   | Frag.  | MT         | ML        | FPS     |
| ------- | ---------- | ---------- | ------- | -------- | ----- | ------ | ---------- | --------- | ------- |
| AB3DMOT | 70.26%     | **86.78%** | 1795    | 1510     | **0** | **70** | 73.15%     | 4.17 %    | **200** |
| mmMOT   | 80.08%     | 85.44      | 790     | **1411** | 13    | 208    | **76.85%** | **2.31%** | 8       |
| Ours    | **81.35%** | 85.66%     | **449** | 1621     | 3     | 170    | 69.91%     | 3.24 %    | 28      |

GPU: Nvidia GTX1070

