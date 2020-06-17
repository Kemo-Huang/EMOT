TRAIN_SEQ_ID = ['0003', '0001', '0013', '0009', '0004',
                '0020', '0006', '0015', '0008', '0012']
VALID_SEQ_ID = ['0005', '0007', '0017', '0011', '0002',
                '0014', '0000', '0010', '0016', '0019', '0018']
TEST_SEQ_ID = [f'{i:04d}' for i in range(29)]
# Valid sequence 0017 has no cars in detection,
# so it should not be included if val with GT detection
# VALID_SEQ_ID = ['0005', '0007', '0011', '0002', '0014', \
#                 '0000', '0010', '0016', '0019', '0018']
TRAINVAL_SEQ_ID = [f'{i:04d}' for i in range(21)]

val_link = "./dataset/kitti/data/val.txt"
# val_link = "./dataset/kitti/data/train.txt"
val_det = "./dataset/kitti/data/pp_val_dets.pkl"
# val_det = "./dataset/kitti/data/pp_train_dets.pkl"
