# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 12:36:09 2022

@author: adamr
"""
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class ModelEvaluator:
    def __init__(self, x_test, y_test, trainer_obj, decoders):
        self.models = trainer_obj.trained_models
        self.x_test, self.y_test = x_test, y_test
        self.decoders = decoders
        self.trainer = trainer_obj
    
    def evaluate_models(self):
        model_names = list(self.models.keys())
        results = {}
        for model_name in model_names:
            results[model_name] = self.model_evaluator(model_name)
        return results
    
    def evaluate_ensembles(self, ensemble_dict, weights=None):
        ensemble_names = list(ensemble_dict.keys())
        for ensemble_name in ensemble_names:
            ensemble_dict[ensemble_name] = self.ensemble_evaluator(ensemble_dict[ensemble_name][0], ensemble_dict[ensemble_name][1], ensemble_dict[ensemble_name][2], weights)
            
            
        return ensemble_dict
            
    def model_evaluator(self, model_name='SVC'):
        if model_name in ['inception', 'cnn', 'effinet']:
            predictions = self.models[model_name].predict(self.x_test)
        else:
            predictions = self.models[model_name].predict(self.trainer.trad_x_test)

        metrics = classification_report(self.y_test, predictions, output_dict=True)
        
        #Get list of all classes in the metric dict
        encoded_classes = list(metrics.keys())
        encoded_classes.remove('accuracy')
        encoded_classes.remove('macro avg')
        encoded_classes.remove('weighted avg')
        
        #Iterate through encoded classes restoring class names.
        for key in encoded_classes:
            metrics[self.decoders[int(key)]] = metrics.pop(key) 
            
        return metrics
    
    def ensemble_evaluator(self, estimators, label_encoder, x_test_list, weights):        
        predictions = self._predict_from_multiple_estimators(estimators, label_encoder, x_test_list, weights)
        metrics = classification_report(self.y_test, predictions, output_dict=True)
        
        #Get list of all classes in the metric dict
        encoded_classes = list(metrics.keys())
        encoded_classes.remove('accuracy')
        encoded_classes.remove('macro avg')
        encoded_classes.remove('weighted avg')
        
        #Iterate through encoded classes restoring class names.
        for key in encoded_classes:
            metrics[self.decoders[int(key)]] = metrics.pop(key) 
            
        return metrics
    
    def _predict_from_multiple_estimators(self, estimators, label_encoder, X_list, weights = None):
        '''Code for this function inspired from following source:
            Kumar, V. (2017) Votingclassifier: Different feature sets, Stack Overflow. 
            Available at: https://stackoverflow.com/questions/45074579/votingclassifier-different-feature-sets (Accessed: November 10, 2022). '''
        # Predict 'soft' voting with probabilities
        pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
        pred2 = np.average(pred1, axis=0, weights=weights)
        pred = np.argmax(pred2, axis=1)
    
        # Convert integer predictions to original labels:
        return label_encoder.inverse_transform(pred)
    
    