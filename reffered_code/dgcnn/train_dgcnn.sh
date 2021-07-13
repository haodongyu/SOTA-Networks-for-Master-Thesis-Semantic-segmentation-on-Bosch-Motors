# CUDA_VISIBLE_DEVICES=2,3 python train_dgcnn_semseg.py --model dgcnn --epoch 100 --log_dir dgcnn_sem_seg_0625

# vCUDA_VISIBLE_DEVICES=2 python test_semseg_dgcnn.py --log_dir dgcnn_sem_seg_0625 --root /home/ies/hyu/data/Training_set_PointNet_0611/ --test_area Validation

CUDA_VISIBLE_DEVICES=3 python test_semseg_dgcnn.py --log_dir dgcnn_sem_seg_0625 --root /home/ies/hyu/data/Zivid_Testset/labeled/ --visual --test_area Test