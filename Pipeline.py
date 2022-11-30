# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:28:09 2022

@author: adamr
"""

from ModelTrainer import ModelTrainer
from DatasetBuilder import DatasetBuilder
from ModelEvaluator import ModelEvaluator
from sklearn.model_selection import train_test_split
import pickle

class Pipeline:
    def __init__(self, pipelineConfiguration):
        self.config = pipelineConfiguration
        
    #Store results
    def _save_item(self, file_name, item):
        path = self.config.output_dir + file_name +'.p'
        with open(path, "wb" ) as f:
        	pickle.dump(item, f)
            
    def _evaluate_ensembles(self, evaluator, ensembles, weights):
        output = {}
        numModels = len(weights[0])
        
        #Permutate through all weight combinations for the given ensemble.
        for i in range(len(weights)):
            ensemble_result = evaluator.evaluate_ensembles(ensembles.copy(), weights[i])
            output[str(numModels)+'-Model-W'+str(i)] = ensemble_result
        
        return output
    
    def _save_models(self, models):
        for key in models:
            if key in ['inception', 'effinet', 'cnn']:
                models[key].model.save(key+'.h5')
            else:
                self._save_item(key, models[key])
        return 1
    
            
    def run(self):
        #Build dataset.
        builder = DatasetBuilder(self.config.data_dir)
        global training_data
        global training_labels
        training_data, training_labels, encoders = builder.generate_dataset(self.config.class_list, self.config.img_height, self.config.img_width)
        decoders = {v: k for k, v in encoders.items()}
        
        x_test, x_train, y_test, y_train = train_test_split(training_data, training_labels, test_size=0.8, random_state=123, stratify=training_labels)
        print(x_test.shape)
        print(x_train.shape)
        print(y_test.shape)
        print(y_train.shape)
        
        #Train models
        trainer = ModelTrainer(x_train, x_test, y_train, self.config.num_classes, self.config.img_height, self.config.img_width)
        models = trainer.train_all_models()
        
        #Evaluate models
        evaluator = ModelEvaluator(x_test, y_test, trainer, decoders)
        results = evaluator.evaluate_models()
        self._save_item('Individual Model Results', results)
        
        #Configure and evaluate ensembles from trained models.
        if self.config.two_model_ensemble_list != None:
            two_model_ensembles = trainer.configure_multiple_ensembles(self.config.two_model_ensemble_list)
            two_model_results = self._evaluate_ensembles(evaluator, two_model_ensembles.copy(), self.config.two_model_weights)
            self._save_item('Two Model Ensemble Results', two_model_results)
            
        if self.config.three_model_ensemble_list !=None:
            three_model_ensembles = trainer.configure_multiple_ensembles(self.config.three_model_ensemble_list)
            three_model_results = self._evaluate_ensembles(evaluator, three_model_ensembles.copy(), self.config.three_model_weights)
            self._save_item('Three Model Ensemble Results', three_model_results)
        
        del models['effinet']
        self._save_models(models)
        self._save_item('History', trainer.history_objects)
        return results, two_model_results, three_model_results
        