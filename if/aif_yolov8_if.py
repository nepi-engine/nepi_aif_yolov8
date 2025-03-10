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

import sys
import os
import os.path
# ROS namespace setup for stand alone testing. Comment out for deployed version
#NEPI_BASE_NAMESPACE = '/nepi/s2x/'
#os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script
import rospy

import glob
import subprocess
import yaml
import time
import numpy as np


from nepi_sdk import nepi_ros
from nepi_sdk import nepi_msg


from std_msgs.msg import Empty, Float32, Int32, String, Bool


TEST_AI_DICT = {
'description': 'Yolov8 ai framework support', 
'pkg_name': 'nepi_aif_yolov8', 
'if_file_name': 'aif_yolov8_if.py', 
'if_path_name': '/opt/nepi/ros/share/nepi_aifs', 
'if_module_name': 'aif_yolov8_if', 
'if_class_name': 'Yolov8AIF', 
'models_folder_name': 'yolov8', 
'launch_pkg_name': 'nepi_aif_yolov8',
'launch_file_name': 'yolov8_ros.launch', 
'node_file_name': 'nepi_ai_yolov8_detection_node.py',  
'active': True
}

TEST_LAUNCH_NAMESPACE = "/nepi/yolov8_test"
TEST_MGR_NAMESPACE = "/nepi/ai_detector_mgr"
TEST_MODELS_LIB_PATH = "/mnt/nepi_storage/ai_models/"



class Yolov8AIF(object):
    TYPICAL_LOAD_TIME_PER_MB = 3.5

    ai_node_dict = dict()
    def __init__(self, ai_dict,launch_namespace, mgr_namespace, models_lib_path):
      if launch_namespace[-1] == "/":
        launch_namespace = launch_namespace[:-1]
      self.launch_namespace = launch_namespace  
      #nepi_msg.printMsgWarn("Launch Namespace: " + self.launch_namespace)
      if mgr_namespace[-1] == "/":
        mgr_namespace = mgr_namespace[:-1]
      self.mgr_namespace = mgr_namespace
      self.models_lib_path = models_lib_path
      self.pkg_name = ai_dict['pkg_name']
      self.node_file_dict = ai_dict['node_file_dict']
      self.launch_pkg = ai_dict['launch_pkg_name']
      self.launch_file = ai_dict['launch_file_name']
      self.models_folder = ai_dict['models_folder_name']
      self.models_folder_path =  os.path.join(self.models_lib_path, self.models_folder)
      nepi_msg.printMsgInfo("Yolov8 models path: " + self.models_folder_path)

    
    #################
    # Yolov8 Model Functions

    def getModelsDict(self):
        models_dict = dict()
        # Try to obtain the path to Yolov8 models from the system_mgr
        nepi_msg.printMsgInfo("ai_yolov8_if: Looking for model files in folder: " + self.models_folder_path)
        # Grab the list of all existing yolov8 cfg files
        if os.path.exists(self.models_folder_path) == False:
            nepi_msg.printMsgInfo("ai_yolov8_if: Failed to find models folder: " + self.models_folder_path)
            return models_dict
        else:
            self.cfg_files = glob.glob(os.path.join(self.models_folder_path,'*.yaml'))
            nepi_msg.printMsgInfo("ai_yolov8_if: Found network config files: " + str(self.cfg_files))
            # Remove the ros.yaml file -- that one doesn't represent a selectable trained neural net
            for f in self.cfg_files:
                cfg_dict = dict()
                success = False
                try:
                    #nepi_msg.printMsgWarn("ai_yolov8_if: Opening yaml file: " + f) 
                    yaml_stream = open(f, 'r')
                    success = True
                    #nepi_msg.printMsgWarn("ai_yolov8_if: Opened yaml file: " + f) 
                except Exception as e:
                    nepi_msg.printMsgWarn("ai_yolov8_if: Failed to open yaml file: " + str(e))
                if success:
                    try:
                        # Validate that it is a proper config file and gather weights file size info for load-time estimates
                        #nepi_msg.printMsgWarn("ai_yolov8_if: Loading yaml data from file: " + f) 
                        cfg_dict = yaml.load(yaml_stream)  
                        model_keys = list(cfg_dict.keys())
                        model_key = model_keys[0]
                        #nepi_msg.printMsgWarn("ai_yolov8_if: Loaded yaml data from file: " + f) 
                    except Exception as e:
                        nepi_msg.printMsgWarn("ai_yolov8_if: Failed load yaml data: " + str(e)) 
                        success = False 
                try: 
                    #nepi_msg.printMsgWarn("ai_yolov8_if: Closing yaml data stream for file: " + f) 
                    yaml_stream.close()
                except Exception as e:
                    nepi_msg.printMsgWarn("ai_yolov8_if: Failed close yaml file: " + str(e))
                
                if success == False:
                    nepi_msg.printMsgWarn("ai_yolov8_if: File does not appear to be a valid A/I model config file: " + f + "... not adding this model")
                    continue
                #nepi_msg.printMsgWarn("ai_yolov8_if: Import success: " + str(success) + " with cfg_dict " + str(cfg_dict))
                cfg_dict_keys = cfg_dict[model_key].keys()
                #nepi_msg.printMsgWarn("ai_yolov8_if: Imported model key names: " + str(cfg_dict_keys))
                #nepi_msg.printMsgWarn("ai_yolov3_if: Imported model key names: " + str(cfg_dict_keys))
                if ("framework" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_yolov8_if: Framework does not specified in model yaml file: " + f + "... not adding this model")
                    continue
                if ("weight_file" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_yolov8_if: File does not appear to be a valid A/I model config file: " + f + "... not adding this model")
                    continue
                #nepi_msg.printMsgWarn("ai_yolov3_if: Imported model key names: " + str(cfg_dict_keys))
                if ("image_size" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_yolov3_if: File does not specify a image size: " + f + "... not adding this model")
                    continue
                #nepi_msg.printMsgWarn("ai_yolov3_if: Imported model key names: " + str(cfg_dict_keys))
                if ("classes" not in cfg_dict_keys):
                    nepi_msg.printMsgWarn("ai_yolov3_if: File does not specify a classes: " + f + "... not adding this model")
                    continue

                param_file = os.path.basename(f)
                framework = cfg_dict[model_key]["framework"]["name"]
                model_name = os.path.splitext(param_file)[0]

                
                if framework != 'yolov8':
                    nepi_msg.printMsgWarn("ai_yolov8_if: Model " + model_name + " not a yolov3 model" + framework + "... not adding this model")
                    continue



               
                weight_file = cfg_dict[model_key]["weight_file"]["name"]
                weight_file_path = os.path.join(self.models_folder_path,weight_file)
                #nepi_msg.printMsgWarn("ai_yolov8_if: Checking that model weights file exists: " + weight_file_path + " for model name " + model_name)
                if not os.path.exists(weight_file_path):
                    nepi_msg.printMsgWarn("ai_yolov8_if: Model " + model_name + " specifies non-existent weights file " + weight_file_path + "... not adding this model")
                    continue
                model_type = cfg_dict[model_key]['type']['name']
                if model_type not in self.node_file_dict.keys():
                    nepi_msg.printMsgWarn("ai_yolov8_if: Model " + model_name + " specifies non-supported model type " + model_type + "... not adding this model")
                    continue
                else:
                    node_file_name = self.node_file_dict[model_type]
                model_size = int(os.path.getsize(weight_file_path) / 1000000)
                model_dict = dict()
                try:
                    model_dict['param_file'] = param_file
                    model_dict['framework'] = framework
                    model_dict['model_name'] = model_name
                    model_dict['model_path'] = self.models_folder_path
                    model_dict['type'] = model_type
                    model_dict['description'] = cfg_dict[model_key]['description']['name']
                    model_dict['img_height'] = cfg_dict[model_key]['image_size']['image_height']['value']
                    model_dict['img_width'] = cfg_dict[model_key]['image_size']['image_width']['value']
                    model_dict['classes'] = cfg_dict[model_key]['classes']['names']
                    model_dict['weight_file']= weight_file
                    model_dict['node_file_name'] = node_file_name
                    model_dict['size'] = model_size
                    model_dict['load_time'] = self.TYPICAL_LOAD_TIME_PER_MB * model_size / 1000000
                    nepi_msg.printMsgInfo("ai_yolov11_if: Model dict create for model : " + model_name)
                except Exception as e:
                    nepi_msg.printMsgInfo("ai_yolov11_if: Failed to get model info : " + str(e))
                models_dict[model_name] = model_dict
            #nepi_msg.printMsgWarn("Model returning models dict" + str(models_dict))
        return models_dict


    def loadModel(self, model_dict):
        success = False
        model_name = model_dict['model_name']
        node_name = model_name
        node_namespace = os.path.join(self.launch_namespace, node_name)
        # Build Darknet new model_name launch command
        launch_cmd_line = [
            "roslaunch", self.launch_pkg, self.launch_file,
            "pkg_name:=" + self.launch_pkg,
            "node_name:=" + node_name,
            "node_namespace:=" + self.launch_namespace,
            "node_file_name:=" + model_dict['node_file_name'],
            "mgr_namespace:=" + self.mgr_namespace, 
            "param_file_path:=" + os.path.join(model_dict['model_path'],model_dict['param_file']),
            "weight_file_path:=" + os.path.join(model_dict['model_path'],model_dict['weight_file'])
        ]
        nepi_msg.printMsgInfo("ai_yolov8_if: Launching Yolov8 AI node " + model_name + " with commands: " + str(launch_cmd_line))
        node_process = subprocess.Popen(launch_cmd_line)
        self.ai_node_dict[model_name] = {'namesapce':node_namespace, 'process':node_process}
        success = True
        return success, node_namespace


    def killModel(self,model_name):
        if model_name in self.ai_node_dict.keys():
            node_process = self.ai_node_dict[model_name]['process']
            nepi_msg.printMsgInfo("ai_yolov8_if: Killing Yolov8 AI node: " + model_name)
            if not (None == node_process):
                node_process.terminate()
            del self.ai_node_dict[model_name]


 
   

if __name__ == '__main__':
    node_name = "ai_yolov8_test"
    while nepi_ros.check_for_node(node_name):
        nepi_msg.printMsgInfo("ai_yolov8_if: Trying to kill running node: " + node_name)
        nepi_ros.kill_node(node_name)
        nepi_ros.sleep(2,10)
    Yolov8AIF(TEST_AI_DICT,TEST_LAUNCH_NAMESPACE,TEST_MGR_NAMESPACE,TEST_MODELS_LIB_PATH)
