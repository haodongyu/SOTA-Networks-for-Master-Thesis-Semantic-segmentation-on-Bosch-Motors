# CUDA_VISIBLE_DEVICES=0 python train_semseg.py --model pointnet2_sem_seg --epoch 100 --log_dir pointnet2_sem_seg_0628_weight10  --bolt_weight 10



######## Test process #############
CUDA_VISIBLE_DEVICES=1 python test_semseg.py --log_dir pointnet2_sem_seg_0628_weight10 --root /home/ies/hyu/data/Zivid_Testset/labeled/ --visual --test_area Test