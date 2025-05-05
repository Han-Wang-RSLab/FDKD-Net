# FDKD-Net: Diffusion-Based Feature Fusion with Dual-Path Knowledge Transfer for Aerial Remote Sensing Detection

By Han Wang, Yiqing Li, Wen Zhou, Zhuoyan Lei 

This repository contains the implementation accompanying our paper FDKD-Net: Diffusion-Based Feature Fusion with Dual-Path Knowledge Transfer for Aerial Remote Sensing Detection.

If you find this project helpful, please consider giving it a star ⭐


![](https://github.com/Han-Wang-RSLab/FDKD-Net/blob/main/FDKD_Net/figs/overview.pdf)

 We leave our system information for reference.

    python: 3.8.16
    torch: 1.13.1+cu117
    torchvision: 0.14.1+cu117
    timm: 0.9.8
    mmcv: 2.1.0
    mmengine: 0.9.0

Other operating environments    

pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 tidecv PyWavelets -i https://pypi.tuna.tsinghua.edu.cn/simple

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources. 
You can download the processed xView and VisDrone-Datasets and HIT-UAV-Datasets  from this Web [link](https://github.com/VisDrone/VisDrone-Dataset) and [link](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset).

- Convert the annotation files into TXT-format annotations.

- Modify the dataset path setting within the script.

```
'dateset's name': {
    'train_img'  : '',  #train image dir
    'train_Label' : '',  #train txt format label file
    'val_img'    : '',  #val image dir
    'val_label'   : '',  #val txt format label file
},
```
- Add domain adaptation direction within the script [__init__.py](./datasets/__init__.py). During training, the domain adaptation direction will be automatically parsed and corresponding data will be loaded. In our paper, we provide four adaptation directions for remote sensing scenarios.
```

```

## Training / Evaluation / Inference
We provide training script on single node as follows.
- Training with single GPU
```
python train.py
```
- Valing with dataset
```
python val.py
```
- Self-distillation process 
```
python distill.py
```
- get_COCO_metrice
```
get_COCO_metrice.py
```
<<<<<<< HEAD

=======
>>>>>>> 1be07be56748639f4f7369145d45f03632c50646
## Result Visualization 
```
![](https://github.com/Han-Wang-RSLab/FDKD-Net/blob/master/FDKD_Net/figs/experiment.png)
```

## Demo prediction
```
![](/figs/prediction.png)
```
