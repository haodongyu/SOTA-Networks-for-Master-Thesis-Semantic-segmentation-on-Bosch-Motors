# SOTA-Networks-for-Master-Thesis-Semantic-segmentation-on-Bosch-Motors
This project is mainly about the used codes for Haodong Yu's master thesis.   

## PointNet/PointNet2
run training and test process:  
```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --test_area 5 --log_dir pointnet2_sem_seg
python test_semseg.py --log_dir pointnet2_sem_seg --test_area 5 --visual
```  
The --log_dir command will build a new dir in the path './Pointnet_Pointnet2_pytorch/log/sem_seg/'.  

## Reference by
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
