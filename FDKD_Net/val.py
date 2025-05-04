import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    model = RTDETR('runs/train/exp/weights/best.pt')
    model.val(data='dataset/data.yaml',
              split='test', # or val
              imgsz=640,
              batch=4,
              save_json=True, 
              project='runs/val',
              name='exp',
              )