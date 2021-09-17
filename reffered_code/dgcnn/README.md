# DGCNN

This repo is implementation for [DGCNN](https://github.com/WangYueFt/dgcnn)in pytorch.

## run training and test process:  

```
## Check model in ./models 
## e.g., dgcnn.py 
python train_dgcnn_semseg.py --model dgcnn --epoch 100 --log_dir dgcnn_sem_seg --root /home/ies/hyu/data/
python test_semseg_dgcnn_new.py --log_dir dgcnn_sem_seg --root /home/ies/hyu/data/ --visual --test_area Validation
```   
  
You can set your data folder by command `--root`.  
The `--log_dir` command will build a new dir in the path `./Pointnet_Pointnet2_pytorch/log/sem_seg/` and save the log and parameter used in training in this folder.
The `--visual` command will build the predictied results into .obj file.  

Visualization results will save in `log/sem_seg/pointnet2_sem_seg/visual/` and you can visualize these .obj file by [MeshLab](http://www.meshlab.net/).  

If you want to run the evaluation process with other data, just reset the `--root dir`. Besides that, remember the name of your data must have str `Validation` or `test`. And then change the `--test_area str` order in corresponding ways.  


## Visualization  
### Using MeshLab
You can download MeshLab APP from here: [MeshLab](http://www.meshlab.net/).



## Citation
If you find this repo useful in your research, please consider citing it and our other works:
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
```
@InProceedings{yan2020pointasnl,
  title={PointASNL: Robust Point Clouds Processing using Nonlocal Neural Networks with Adaptive Sampling},
  author={Yan, Xu and Zheng, Chaoda and Li, Zhen and Wang, Sheng and Cui, Shuguang},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
```
@InProceedings{yan2021sparse,
  title={Sparse Single Sweep LiDAR Point Cloud Segmentation via Learning Contextual Shape Priors from Scene Completion},
  author={Yan, Xu and Gao, Jiantao and Li, Jie and Zhang, Ruimao, and Li, Zhen and Huang, Rui and Cui, Shuguang},
  journal={AAAI Conference on Artificial Intelligence ({AAAI})},
  year={2021}
}
```

## Reference by
[halimacc/pointnet3](https://github.com/halimacc/pointnet3)<br>
[fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch)<br>
[charlesq34/PointNet](https://github.com/charlesq34/pointnet) <br>
[charlesq34/PointNet++](https://github.com/charlesq34/pointnet2)
