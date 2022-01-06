import torch
# from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
import time
import math

def square_pad_img_grey(img,grey = 114):
    h, w, d = img.shape
    max_dim = max(h,w)
    pad_img = np.ones_like(img,shape=(max_dim,max_dim,3))*grey
    if max_dim == w:
        total_padding = max_dim - h
        if total_padding %2 ==0:
            half_pad = int(total_padding/2)
            pad_img[half_pad:half_pad + h,:,:] = img
        else:
            half_pad = int(math.floor(total_padding/2))
            pad_img[half_pad:half_pad + h,:,:] = img
    else:
        total_padding = max_dim - w
        if total_padding %2 ==0:
            half_pad = int(total_padding/2)
            pad_img[:,half_pad:half_pad + w,:] = img
        else:
            half_pad = int(math.floor(total_padding/2))
            pad_img[:,half_pad:half_pad + w,:] = img
    return pad_img
    

def process_detections1(outputs, img):
    ''' https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html '''
    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3,top_k=1)
    colors = np.random.uniform(0, 255, size=(len(classIDs), 3))
    if len(indices) > 0:
        for i in indices.flatten():
            # print(classIDs[i],confidences[i])
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classIDs[i], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # cv2.imshow('DET',img)
    # cv2.waitKey(0)
    return indices

random.seed(108)

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
print('CUDA available: ',torch.cuda.is_available())
device = torch.cuda.get_device_properties(0)

t0 = time.time()
print('Loading model...', end=' ')
full_model_path = "C:/VirEnvs/YoloV5_Env/yolov5/runs/train/yolo_imp_det_v0/weights/best.onnx"
model = cv2.dnn.readNetFromONNX(full_model_path)
t1 = time.time()

print('complete.',round(t1 -t0,3),'seconds')

txt_dir = "D:/Documents/TensorFlow2/workspace/ImpDetectYolo/annot_convert_test"
img_dir = txt_dir

class_names = {0:'Impalement',1:'Miss'}
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

txts = [f for f in os.listdir(txt_dir) if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.txt)$', f)]
txts = [txts[0], txts[1], txts[2]]
images = [ele[:-4]+".jpg" for ele in txts]
for img_name in images:
    t0 = time.time()
    print('Inference on', img_name,'...', end = ' ')
    image = cv2.imread(os.path.join(img_dir,img_name))
    image_pad = square_pad_img_grey(image)
    image_height, image_width, d = image.shape
    # blob = cv2.dnn.blobFromImage(image=image, size=(640, 640), swapRB=True)
    blob = cv2.dnn.blobFromImage(image=image_pad, scalefactor = 1.0/255, size=(640, 640), swapRB=True,crop=False)
    model.setInput(blob)
    output = model.forward()
    t1 = time.time()
    print('complete.',round(t1 -t0,3),'seconds' )
    # print('BLOB. Type:', type(blob), '\tDtype:',blob.dtype,'\tShape:',blob.shape)
    # for ii in range(0,3):
    #     print(blob[0,ii,0:3,0:3])
    # blobshow = blob[0].reshape(640,640,3)
    # cv2.imshow('BLOB', blobshow)
    # cv2.waitKey(0)

    rets = process_detections1(output,image)
    # print(len(rets)) 
    # print(output)
    # print(output[0,0])
    # Loop on the outputs
    # rows = image_height
    # H = rows
    # cols = image_width
    # W = cols
    # boxes = []
    # confidences = []
    # classIDs = []
    # detections=output
    # for output in detections:
    #     for detection in output:
    #         scores = detection[5:]
    #         classID = np.argmax(scores)
    #         confidence = scores[classID]
    #         if confidence > .99:
    #             # W, H are the dimensions of the input image
    #             box = detection[0:4] * np.array([W, H, W, H])
    #             (centerX, centerY, width, height) = box.astype("int")
    #             x = int(centerX - (width / 2))
    #             y = int(centerY - (height / 2))
    #             boxes.append([x, y, int(width), int(height)])
    #             confidences.append(float(confidence))
    #             classIDs.append(classID)
    # print(classIDs)

    # for detection in output[0]:
    #     print(detection)
    #     score = float(detection[2])
    #     if score > 0.2:

    #         left = detection[3] * cols
    #         top = detection[4] * rows
    #         right = detection[5] * cols
    #         bottom = detection[6] * rows

    #         #draw a red rectangle around detected objects
    #         cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    # # loop over each of the detection
    # for detection in output[0, :, :]:
    #     # extract the confidence of the detection
    #     confidence = detection[2]
    #     # draw bounding boxes only if the detection confidence is above...
    #     # ... a certain threshold, else skip
    #     if confidence > .4:
    #         # get the class id
    #         class_id = detection[1]
    #         # map the class id to the class
    #         class_name = class_names[int(class_id)-1]
    #         color = COLORS[int(class_id)]
    #         # get the bounding box coordinates
    #         box_x = detection[3] * image_width
    #         box_y = detection[4] * image_height
    #         # get the bounding box width and height
    #         box_width = detection[5] * image_width
    #         box_height = detection[6] * image_height
    #         # draw a rectangle around each detected object
    #         cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
    #         # put the FPS text on top of the frame
    #         cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # img_resize = cv2.resize(image,dsize=(int(image_width/3),int(image_height/3)))
    # cv2.imshow('IMAGE',img_resize)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
