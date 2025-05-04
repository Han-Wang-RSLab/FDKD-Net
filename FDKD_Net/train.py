import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('/root/autodl-tmp/RTDETR-main/ultralytics/cfg/models/rt-detr/my/rtdetr-C2f-SMPCGLU-FDPN-DASI-SRFD.yaml')
    #  model.load('') # loading pretrain weights
    model.train(data='/root/autodl-tmp/RTDETR-main/dataset/hit-uav/dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                lr0=0.0001,
                workers=4, 
                project='runs/train',
                
                )