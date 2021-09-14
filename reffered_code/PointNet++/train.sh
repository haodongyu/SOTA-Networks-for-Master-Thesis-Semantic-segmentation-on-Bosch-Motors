CUDA_VISIBLE_DEVICES=4,5 python train_semseg.py --model pointnet2_sem_seg --epoch 100 --log_dir pointnet2_sem_seg_0914_weights1 --root /home/ies/hyu/data/Training_set_0814_noise/ --bolt_weight 1
# 0830 same as 0827
# CUDA_VISIBLE_DEVICES=0,1 python test_semseg_new.py --log_dir pointnet2_sem_seg_0831 --root /home/ies/hyu/data/Training_set_0827_noiseNoCover/ --visual --test_area Validation

######## Test process #############
# CUDA_VISIBLE_DEVICES=2 python test_semseg_new.py --log_dir pointnet2_sem_seg_0814 --root /home/ies/hyu/data/Zivid_Testset/labeled_transformed_0909/ --visual --test_area Test