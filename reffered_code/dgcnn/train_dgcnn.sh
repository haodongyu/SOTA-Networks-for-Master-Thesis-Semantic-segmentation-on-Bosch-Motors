CUDA_VISIBLE_DEVICES=0,1 python train_dgcnn_semseg.py --model dgcnn --epoch 100 --log_dir dgcnn_sem_seg_0820_bw10  --root /home/ies/hyu/data/Training_set_0814_noise/ --bolt_weight 10 --k 8

# CUDA_VISIBLE_DEVICES=4,5 python test_semseg_dgcnn_new.py --log_dir dgcnn_sem_seg_0814 --root /home/ies/hyu/data/Training_set_0814_noise/ --visual --test_area Validation

# CUDA_VISIBLE_DEVICES=4,5 python test_semseg_dgcnn_new.py --log_dir dgcnn_sem_seg_0814 --root /home/ies/hyu/data/Zivid_Testset/labeled_transformed/ --visual --test_area Test