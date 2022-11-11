# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:18:18 2022

@author: adamr
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin

#from HyperparameterGridConfigs import HyperparameterGridConfigs

#Reproducible results from NNs
tf.random.set_seed(123)

class ModelTrainer:
    def __init__(self, df, test_size=0.2, img_height=200, img_width=200):
        self.x_train, self.x_test, self.y_train, self.y_test = self._generateLearningSets(df, test_size) 
        self.x_train = np.asarray(self.x_train).astype(np.float32)
        self.y_train = np.asarray(self.y_train).astype(np.float32)
        self.img_height = img_height
        self.img_width  = img_width
        
    def train_all_models():
        pass
    
    def _generateLearningSets(self, df, test_size):
        #Split data into target and non-target
        Y = df['Category'].copy()
        X = df.drop('Category', axis=1).copy()
        
        #Random state set for reproducible results.
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=123, stratify=Y)
        
        return X_train, X_test, Y_train, Y_test
    
    def _configure_cnn_model(self, num_layers:int) -> models.Sequential:
        '''Builds CNN model using keras layers.'''
        model = models.Sequential()
        for i in range(num_layers):
            if i == 0:
                model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)))
                model.add(layers.MaxPooling2D((2, 2)))
            else:
                model.add(layers.Conv2D(64, (3, 3), activation='relu'))
                model.add(layers.MaxPooling2D((2, 2)))
                if i == num_layers-1:
                        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10))
        
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
        model.summary()
        return model
    
    def _configure_inception_model(self):
        model = tf.keras.applications.InceptionV3(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=(self.img_height, self.img_width, 3),
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )
        
        model.trainable = False
        augmented_model = models.Sequential()
        augmented_model.add(model)
        augmented_model.add(layers.GlobalAveragePooling2D())
        augmented_model.add(layers.Dropout(0.5))
        augmented_model.add(layers.Dense(9, 
                            activation='softmax'))
        
        augmented_model.compile(loss='categorical_crossentropy', 
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model
        
    
    def _configure_efficient_network_model(self):
        model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=(self.img_height,self.img_width,3),
            pooling=None,
            classes=1000,
            classifier_activation='softmax'
        )
        augmented_model = models.Sequential()
        augmented_model.add(model)
        augmented_model.add(layers.GlobalAveragePooling2D())
        augmented_model.add(layers.Dropout(0.5))
        augmented_model.add(layers.Dense(9, 
                            activation='softmax'))
        
        augmented_model.compile(loss='categorical_crossentropy', 
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        return model
        
    
    def _hyperparamter_tuning(self, model: ClassifierMixin, hyperparameter_grid: dict, training_data: np.array, training_labels: np.array) -> dict:
        '''Optimizes hyperparamters for inputted model using repeated stratified k-folds cross validation, returning these optimal hyperparameters'''
        #Set up grid search with 5 fold cross validation.
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameter_grid, n_jobs=7, cv=cv, scoring='f1', error_score=0)
        
        #Execute grid search
        grid_result = grid_search.fit(training_data, training_labels)
        
        #Summarize the results
        print("Model Training Performance:")
        print('------------------------------------------------------------------')
        print(f"F1: {grid_result.best_score_:3f} using {grid_result.best_params_}")
        print('------------------------------------------------------------------')
        
        parameters = grid_result.best_params_
        
        return parameters