# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:21:41 2022

@author: adamr
"""
import os

class PipelineConfiguation:
    def __init__(self, class_list, data_dir, output_dir, two_model_ensemble_list, two_model_weights, three_model_ensemble_list, three_model_weights, img_width=256, img_height=320):
        self.set_class_list(class_list)
        self.num_classes = len(self.class_list)
        self.set_path(data_dir, 'data_dir')
        self.set_path(output_dir, 'output_dir')
        self.set_ensemble_list(three_model_ensemble_list, 3)
        self.set_ensemble_list(two_model_ensemble_list, 2)
        self.set_weight_list(two_model_weights, 2)
        self.set_weight_list(three_model_weights, 3)
        self.set_image_dimension(img_height, 'height')
        self.set_image_dimension(img_width, 'width')
    
    def set_ensemble_list(self, ensemble_list, numModels):
        #Check that lists are of length 2 or 3.
        listLen = len(ensemble_list[0])
        if listLen != numModels:
            raise ValueError('Error Setting ' + str(numModels)+ ' Model Ensembles: Expected lists of length, '+str(numModels)+', got length '+str(listLen))
        
        #Check that the length of each internal list is the same.
        for _list in ensemble_list:
            if len(_list)!=listLen:
                raise ValueError('Error Setting ' +str(numModels)+' Model Ensemble List: Not all internal lists are the same length.')
        
        #If checks suceed set the ensemble list
        if listLen == 2:
            self.two_model_ensemble_list = ensemble_list
        else:
            self.three_model_ensemble_list = ensemble_list
    
    def set_weight_list(self, weight_list, numModels):
        #Check that lists are of length 2 or 3.
        listLen = len(weight_list[0])
        if listLen != numModels:
            raise ValueError('Error Setting ' + str(numModels)+ ' Model Weights: Expected lists of length, '+str(numModels)+', got length '+str(listLen))
        
        #Check that the length of each internal list is the same.
        for _list in weight_list:
            if len(_list)!=listLen:
                raise ValueError('Error Setting ' +str(numModels)+' Model Weight List: Not all internal lists are the same length.')
        
        #If checks suceed set the ensemble list
        if listLen == 2:
            self.two_model_weights = weight_list
        else:
            self.three_model_weights = weight_list
    
    def set_path(self, path, path_type):
        if path_type not in ['data_dir', 'output_dir']:
            raise ValueError('Invalid Path Type, please use one of the following "input_dir", "output_dir".')
        
        pathExistsFlag = os.path.exists(path)
        if pathExistsFlag:
            if path_type == 'data_dir':
                self.data_dir = path
            else:
                self.output_dir = path
                
    def set_class_list(self, class_list):
        marked_map  = dict.fromkeys(class_list, 0)
        for _class in class_list:
            if _class.lower() not in ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior','pet', 'travel']:
                raise ValueError('Invalid class, please choose from the following classes: "beauty", "family", "fashion", "fitness", "food", "interior","pet", "travel"')
            else:
                marked_map[_class] += 1
                if marked_map[_class] >1:
                    raise ValueError('Duplicate class found in class list: ' + str(_class))

        self.class_list = class_list
    
    
    def set_image_dimension(self, dim, dim_type):
        if int(dim) == dim:
            if dim > 0:
                if dim_type == 'height':
                    self.img_height = dim
                elif dim_type == 'width':
                    self.img_width = dim
                else:
                    raise ValueError('Dimension type must be one of "height" or "width" got: '+str(dim_type))
            else:
                raise ValueError('Image ' +str(dim_type) + ' must be a positive int, got negative int: ' + str(dim))
        else:
            raise ValueError('Image ' +str(dim_type) + ' must be of type int, got type: ' + str(type(dim)))
                