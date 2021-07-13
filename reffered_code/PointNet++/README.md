# PointNet++
## run training and test process:  

```
## Check model in ./models 
## e.g., pointnet2_ssg
python train_semseg.py --model pointnet2_sem_seg --epoch 100 --log_dir pointnet2_sem_seg_0628_weight10 --root /home/ies/hyu/data/Zivid_Testset/labeled/
python test_semseg.py --log_dir pointnet2_sem_seg --root /home/ies/hyu/data/Zivid_Testset/labeled/ --test_area 5 --visual
```   
  
You can set the your data folder by command `--root`.  
The `--log_dir` command will build a new dir in the path `./Pointnet_Pointnet2_pytorch/log/sem_seg/` and save the log and parameter used in training in this folder.  
  
For the further necessary settings please see the [train_semseg.py](./train_semseg.py) and [test_semseg.py](./test_semseg.py).

## Reference by
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
