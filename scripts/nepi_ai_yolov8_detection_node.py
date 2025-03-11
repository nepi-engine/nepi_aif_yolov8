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
import rospy
import torch
import cv2
import numpy as np


from ultralytics import YOLO

from nepi_sdk import nepi_ros
from nepi_sdk import nepi_msg
from nepi_sdk import nepi_img

from nepi_sdk.ai_detector_if import AiDetectorIF

# Define your PyTorch model and load the weights
# model = ...


TEST_DETECTION_DICT_ENTRY = {
    'name': 'TEST_DATA', # Class String Name
    'id': 1, # Class Index from Classes List
    'uid': '', # Reserved for unique tracking by downstream applications
    'prob': .3, # Probability of detection
    'xmin': 10,
    'ymin': 10,
    'xmax': 50,
    'ymax': 50,
    'width_pixels': 40,
    'height_pixels': 40,
    'area_pixels': 16000,
    'area_ratio': 0.22857
}



class Yolov8Detector():
    defualt_config_dict = {'threshold': 0.3,'max_rate': 5}

    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_yolov8" # Can be overwitten by luanch command
    def __init__(self):
        #### APP NODE INIT SETUP ####
        nepi_ros.init_node(name= self.DEFAULT_NODE_NAME)
        self.node_name = nepi_ros.get_node_name()
        self.base_namespace = nepi_ros.get_base_namespace()
        self.node_namespace = self.base_namespace + self.node_name
        nepi_msg.createMsgPublishers(self)
        nepi_msg.publishMsgInfo(self,"Starting Initialization Processes")
        ##############################
        # Initialize parameters and fields.
        node_params = nepi_ros.get_param(self,"~")
        nepi_msg.publishMsgInfo(self,"Starting node params: " + str(node_params))
        self.all_namespace = nepi_ros.get_param(self,"~all_namespace","")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace
        self.weight_file_path = nepi_ros.get_param(self,"~weight_file_path","")
        if self.weight_file_path == "":
            nepi_msg.publishMsgWarn(self,"Failed to get required node info from param server: ")
            rospy.signal_shutdown("Failed to get valid model info from param")
        else:
            # The ai_models param is created by the launch files load network_param_file line
            model_info = nepi_ros.get_param(self,"~ai_model","")
            if model_info == "":
                nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: ")
                rospy.signal_shutdown("Failed to get valid model file paths")
            else:
                try: 
                    model_framework = model_info['framework']['name']
                    model_type = model_info['type']['name']
                    model_description = model_info['description']['name']
                    self.classes = model_info['classes']['names']
                    self.proc_img_width = model_info['image_size']['image_width']['value']
                    self.proc_img_height = model_info['image_size']['image_height']['value']
                except Exception as e:
                    nepi_msg.publishMsgWarn(self,"Failed to get required model info from params: " + str(e))
                    rospy.signal_shutdown("Failed to get valid model file paths")

                if model_framework != 'yolov8':
                    nepi_msg.publishMsgWarn(self,"Model not a yolov8 model: " + model_framework)
                    rospy.signal_shutdown("Model not a valid framework")


                if model_type != 'detection':
                    nepi_msg.publishMsgWarn(self,"Model not a valid type: " + model_type)
                    rospy.signal_shutdown("Model not a valid type")

                nepi_msg.publishMsgInfo(self,"Loading model: " + self.node_name)
                self.model = YOLO(self.weight_file_path)

                #nepi_msg.publishMsgInfo(self,"Waiting " + str(800) + " seconds for model to load")
                #nepi_ros.sleep(800)

                nepi_msg.publishMsgInfo(self,"Starting ai_if with defualt_config_dict: " + str(self.defualt_config_dict))
                self.ai_if = AiDetectorIF(model_name = self.node_name,
                                    framework = model_framework,
                                    description = model_description,
                                    proc_img_height = self.proc_img_height,
                                    proc_img_width = self.proc_img_width,
                                    classes_list = self.classes,
                                    defualt_config_dict = self.defualt_config_dict,
                                    all_namespace = self.all_namespace,
                                    preprocessImageFunction = self.preprocessImage,
                                    processDetectionFunction = self.processDetection,
                                    has_img_tiling = False)

                #########################################################
                ## Initiation Complete
                nepi_msg.publishMsgInfo(self,"Initialization Complete")
                # Spin forever (until object is detected)
                nepi_ros.spin()
                #########################################################        
              



    def preprocessImage(self,cv2_img,options_dict):
        height, width = cv2_img.shape[:2]
        
        # For Future
        '''
        tile = False
        if 'tile'  in options_dict.keys():
            tile = options_dict['tile']
        '''
        cv2_img = nepi_img.resize_proportionally(cv2_img, self.proc_img_width,self.proc_img_height,interp = cv2.INTER_NEAREST)
        
        
        # Convert BW image to RGB
        if nepi_img.is_gray(cv2_img):
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)

        # Create image dict with new image
        img_dict = dict()
        img_dict['cv2_img'] = cv2_img
        img_shape = cv2_img.shape
        img_dict['org_width'] = width 
        img_dict['orig_height'] = height 
        img_dict['tiling'] = False

        return img_dict



    def processDetection(self,img_dict, threshold):
        detect_dict_list = []
        tile = False
        # For Future
        #if 'tile'  in options_dict.keys():
        #    tile = options_dict['tile']
        if img_dict is not None:
            if 'cv2_img' in img_dict.keys():
                cv2_img = img_dict['cv2_img']
                if cv2_img is not None:

                    # Convert BGR image RGB
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

                    cv2_img = img_dict['cv2_img']
                    cv2_img_shape = cv2_img.shape
                    cv2_img_width = cv2_img_shape[1]
                    cv2_img_height = cv2_img_shape[0]
                    cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]
                    #nepi_msg.publishMsgWarn(self,"Original image size: " + str(orig_size))

                    # Update model settings
                    self.model.conf = threshold  # Confidence threshold (0-1)

                    try:
                        # Inference
                        results = self.model(cv2_img, conf=threshold, verbose=False)
                        #nepi_msg.publishMsgWarn(self,"Got Yolov8 detection results: " + str(results[0].boxes))
                
                        #nepi_msg.publishMsgWarn(self,"Got Yolov8 detection results: " + str(results[0].boxes))
                        ids = results[0].boxes.cls.to('cpu').tolist()
                        #nepi_msg.publishMsgWarn(self,"Got Yolov8 detection ids: " + str(ids))
                        boxes = results[0].boxes.xyxy.to('cpu').tolist()
                        #nepi_msg.publishMsgWarn(self,"Got Yolov8 detection boxes: " + str(boxes))
                        confs = results[0].boxes.conf.to('cpu').tolist()
                        #nepi_msg.publishMsgWarn(self,"Got Yolov8 detection confs: " + str(confs))
                    
                    except Exception as e:
                        nepi_msg.publishMsgInfo(self,"Failed to process detection with exception: " + str(e))
                

                    for i, idf in enumerate(ids):
                        id = int(idf)
                        det_name = self.classes[id]
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
                            'width_pixels': cv2_img_width,
                            'height_pixels': cv2_img_height,
                            'area_pixels': int(det_area),
                            'area_ratio': det_area / cv2_img_area
                        }
                        detect_dict_list.append(detect_dict)
                        #nepi_msg.publishMsgInfo(self,"Got detect dict entry: " + str(detect_dict))
            
        return detect_dict_list



if __name__ == '__main__':
    Yolov8Detector()
