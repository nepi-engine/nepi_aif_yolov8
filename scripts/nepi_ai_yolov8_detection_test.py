#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_apps) repo
# (see https://https://github.com/nepi-engine/nepi_apps)
#
# License: nepi applications are licensed under the "Numurus Software License", 
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment block.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com
#

import os
import time
import copy
import sys
import torch
import cv2
import numpy as np


from ultralytics import YOLO


####################

image_path="/mnt/nepi_storage/sample_data/Images/HardHats/hats_img_4.jpg"

model_path="/mnt/nepi_storage/ai_models/yolov8/common_objects_yolov8_640_tiny.pt"

threshold=0.3
  
####################              

if __name__ == '__main__':

    print("RUNNING YOLOV8 DETECTION TEST")


   ######################################
    # Load image

    print('')
    print("Loading image: " + str(image_path))

    # Convert BGR image RGB

    cv2_img = cv2.imread(image_path)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    cv2_img_shape = cv2_img.shape
    cv2_img_width = cv2_img_shape[1]
    cv2_img_height = cv2_img_shape[0]
    cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]

    print("image size: " + str(cv2_img.shape))


    ######################################
    # Load Model

    weight_file_path=model_path
    print('')
    print("Loading model: " + str(weight_file_path))

    device = 'cpu'
    has_cuda = torch.cuda.is_available()
    print("CUDA available: " + str(has_cuda))
    if has_cuda == True:
        cuda_count = torch.cuda.device_count()
        print("CUDA GPU Count: " + str(cuda_count))
        if cuda_count > 0:
            device = 'cuda'

    print("Loading model: " + str(weight_file_path))
    ai_model = YOLO(weight_file_path)
    ai_model.conf = threshold  # Confidence threshold (0-1)


    ######################################
    # Run Detection

    print('')
    print("Priming the Model")

    detect_dict_list = []


    # Prime the model
    try:
        # Inference
        results = ai_model(cv2_img, conf=threshold, verbose=False, device=device)
        #print("Got Yolov8 detection results: " + str(results[0].boxes))

        #print("Got Yolov8 detection results: " + str(results[0].boxes))
        ids = results[0].boxes.cls.to('cpu').tolist()
        #print("Got Yolov8 detection ids: " + str(ids))
        boxes = results[0].boxes.xyxy.to('cpu').tolist()
        #print("Got Yolov8 detection boxes: " + str(boxes))
        confs = results[0].boxes.conf.to('cpu').tolist()
        #print("Got Yolov8 detection confs: " + str(confs))
    
    except Exception as e:
        print("Failed to process detection with exception: " + str(e))

    print('')
    print("Running Detections")

    start_time = time.time()

    for i in range(1, 100):
        try:
            # Inference
            results = ai_model(cv2_img, conf=threshold, verbose=False, device=device)
        except Exception as e:
            print("Failed to process detection with exception: " + str(e))
    


    end_time = time.time()




    rescale_ratio = 1
    for i, idf in enumerate(ids):
        id = int(idf)
        det_name = id #classes[id]
        det_id = id
        det_prob = confs[i]
        det_box = boxes[i]
        det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
        detect_dict = {
            'name': det_name, # Class String Name
            'id': det_id, # Class Index from Classes List
            'uid': '', # Reserved for unique tracking by downstream applications
            'prob': det_prob, # Probability of detection
            'xmin': int(det_box[0] ),
            'ymin': int(det_box[1] ) ,
            'xmax': int(det_box[2] ),
            'ymax': int(det_box[3]),
            'area_pixels': int(det_area),
            'area_ratio': det_area / cv2_img_area
        }
        # Rescale to orig image size
        detect_dict['xmin'] = int(detect_dict['xmin'] * rescale_ratio)
        detect_dict['ymin'] = int(detect_dict['ymin'] * rescale_ratio)
        detect_dict['xmax'] = int(detect_dict['xmax'] * rescale_ratio)
        detect_dict['ymax'] = int(detect_dict['ymax'] * rescale_ratio)
        detect_dict_list.append(detect_dict)

    print("Got detect dict list: " + str(detect_dict_list))



    elapsed_time = end_time - start_time
    drate=float(1.0)/elapsed_time * 100
    print("")
    print("")
    print(f"Ran 100 detections in: {elapsed_time:.6f} seconds")
    print(f"Average detection rate: {drate:.2f} hz")

