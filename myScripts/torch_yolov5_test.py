import torch
import time

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
print('CUDA available: ',torch.cuda.is_available())

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', classes=2)
model.load_state_dict(torch.load("C:/VirEnvs/YoloV5_Env/yolov5/runs/train/yolo_imp_det_v0/weights/best.pt"))

# model = model.fuse().autoshape()