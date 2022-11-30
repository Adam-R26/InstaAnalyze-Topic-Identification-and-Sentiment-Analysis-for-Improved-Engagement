# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 21:04:57 2022

@author: adamr
"""
import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class DatasetBuilder:
    def __init__(self, data_dir=r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\Dataset'):
        self.data_dir = data_dir
        
    def generate_dataset(self, class_list, image_height, image_width):
        training_data = []
        training_labels  = []
        
        for _class in class_list:
            #Get the names of all of the files in the class directory.
            class_path = os.path.join(self.data_dir, _class)
            image_names = os.listdir(class_path)
            for image_name in image_names:
                #Load in each file in the dir, and perform preprocessing on it.
                full_image_path = os.path.join(class_path, image_name)
                image = self.load_image(full_image_path, image_height, image_width)
                
                #Add to training sets.
                training_data.append(image)
                training_labels.append(_class)
        
        #Label encode the target
        encoding={k: v for v, k in enumerate(np.unique(training_labels))}
        training_labels = [encoding[i] for i in training_labels]
        
        return np.array(training_data, dtype=np.float32), np.array(training_labels, dtype=np.int32), encoding
                
    def load_image(self, image_path, image_height, image_width):
        image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image= cv2.resize(image, (image_height, image_width), interpolation=cv2.INTER_AREA)
        image= np.array(image)
        image = image.astype('float32')
        image /= 255 
        return image