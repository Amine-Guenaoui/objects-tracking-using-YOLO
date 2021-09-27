from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time 
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching #for the association matrix 
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet #to help detect objects 

#loading wieghts
class_names = [c.strip() for c in open('data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('weights/yolov3.tf')

max_cosine_distance = 0.5 # features similarity 
nn_budget = None #to create the libraries 
nms_max_overlap = 0.8 #to avoid many detections on the same object

#application using nearest neighbor as metric for the tracker of features  
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine',max_cosine_distance,nn_budget)
tracker = Tracker(metric)

#loading the video 

vid = cv2.VideoCapture('./data/video/test2.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/result.avi',codec,vid_fps,(vid_width,vid_height))

#trajectoire
from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

#counter 
counter = []


while True:
    _,img = vid.read()
    if img is None:
        print('Completed')
        break
    #pretraitement de l'image
    img_in = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in,0) #adding more dimension to the original image
    img_in = transform_images(img_in,416)


    t1 = time.time()
    #prediction using YOLO
    boxes,scores,classes,nums = yolo.predict(img_in)

    #boxes has 3D shape (1, 100, 4)
    #scores has 2D shape  (1, 100)
    # classes has  2D shape( 1 ,100)
    #nums has 1D shape (1,)
    #getting the classes
    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    #getting the names and boxes and extracting features 
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    calsses = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs,classes,nms_max_overlap,scores)
    detections = [detections[i] for i in indices]
    #predictions and then update  using KALMAN 
    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    #objects that are passing by 
    current_count = int(0)
    #for each tracked object draw the box and it's path 
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        color = colors[int(track.track_id) % len(colors) ]
        color = [i * 55 for i in color]

        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img,(int(bbox[0]),int(bbox[1]-30)),(int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17,int(bbox[1])), color, -1)
        cv2.putText(img,class_name+"-"+str(track.track_id),(int(bbox[0]),int(bbox[1]-10)),0,0.75,(255,255,255),2)

        #hisotry part / trajectoire
        center = (int(((bbox[0])+(bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2) ) 
        pts[track.track_id].append(center)
        #drawing the PATH
        
        for j in range(1 , len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color , thickness)
        
        #drawing a line that tells how many objects have passed by zone or line
        height, width , _ = img.shape
        cv2.line(img, (0,int(3*height/6+height/20)), (width,int(3*height/6+height/20)), (0,255,0),thickness=2)
        cv2.line(img, (0,int(3*height/6-height/20)), (width,int(3*height/6-height/20)), (0,255,0),thickness=2)
        
        #for each box check if it's center is passing by the line 
        
        center_y = int( ( (bbox[1])+(bbox[3]) ) / 2)
        
        if center_y <= int(3*height/6+height/30) and center_y >= int(3*height/6-height/30):
            if class_name == 'car' or class_name == 'truck' or class_name == "person":
                
                counter.append(int(track.track_id))
                current_count +=1
    
    #related to the objects that passed by a line 
    total_count = len(str(counter))
    cv2.putText(img , "Current Vehicles and Persons  Count: " + str(current_count), (0,160), 0, 1 , (0,255,0), 2)
    cv2.putText(img , "Total Vehicles and Persons  Count: " + str(total_count), (0,130), 0, 1 , (0,255,0), 2)
    #passed 
    #counting fps 
    fps = 1./(time.time()-t1)
    cv2.putText(img,"FPS: {:.2f}".format(fps), (0,30),0,1,(0,255,0),2)
    #showing the frame and writing it on the file 
    cv2.imshow('output',img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()
