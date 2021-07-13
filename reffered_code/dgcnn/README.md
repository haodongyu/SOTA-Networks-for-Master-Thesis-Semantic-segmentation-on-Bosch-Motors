# DGCNN
## run training and test process:  

```
## Check model in ./models 
## e.g., dgcnn.py 
python python train_dgcnn_semseg.py --model dgcnn --epoch 100 --log_dir dgcnn_sem_seg_0625 --root /home/ies/hyu/data/Training_set_PointNet_0611/
python test_semseg_dgcnn.py --log_dir dgcnn_sem_seg_0625 --root /home/ies/hyu/data/Training_set_PointNet_0611/ --visual --test_area Validation
```   
  
You can set the your data folder by command `--root`.  
The `--log_dir` command will build a new dir in the path `./Pointnet_Pointnet2_pytorch/log/sem_seg/` and save the log and parameter used in training in this folder.
The `--visual` command will build the predictied results into .obj file.
  
For the further necessary settings please see the [train_semseg.py](./train_semseg.py) and [test_semseg.py](./test_semseg.py).  

## Visualization  
### Using show3d_balls.py
```
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```  
### Using MeshLab
You can download MeshLab APP from here: [MeshLab](http://www.meshlab.net/).

## Reference by
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
