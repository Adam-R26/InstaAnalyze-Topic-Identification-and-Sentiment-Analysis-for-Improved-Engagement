# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:18:18 2022

@author: adamr
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
from HyperparameterGridConfigs import HyperparameterGridConfigs

#Reproducible results from NNs
tf.random.set_seed(123)

class ModelTrainerV2:
    def __init__(self, x_train, x_test, y_train, y_test, img_height=260, img_width=320): #x_train, x_test, y_train, y_test,
        self.img_height, self.img_width = img_height, img_width
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def train_all_models(self):
        '''Pipelines to train all machine learning models and output the results, returns the models in a dictionary.'''
        print("----------------------------------------------------------------------")
        print("Optimizing Hyperparamters for Each Model")
        print('Random Forest:')
        hyperparameterConfig = HyperparameterGridConfigs()
        #rf = self._hyperparamter_tuning(RandomForestClassifier(random_state=123, n_jobs=7), hyperparameterConfig.get_rf_hyperparam_grid(), self.x_train.reshape(self.x_train.shape[0], -1) , self.y_train)
        
        print('\nSupport Vector Machine:')
        svm = SVC(random_state=123, verbose=True, class_weight='balanced',)
        svm.fit(self.x_train.reshape(self.x_train.shape[0], -1), self.y_train)
        
        predictions = svm.predict(self.x_test.reshape(self.x_test.shape[0], -1))
        accuracy = accuracy_score(self.y_test, predictions)
        
    
        print("----------------------------------------------------------------------")
        return {'SVM': svm, 'Accuracy':accuracy} #'RF': rf,
    
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
        #model.add(layers.Input(batch_shape=(self.img_height, self.img_width, 3)))
        #model.add(layers.Dropout(0.3))
        for i in range(num_layers):
            if i == 0:
                model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(self.img_height, self.img_width, 3)))#
                model.add(layers.MaxPooling2D((2, 2)))
            else:
                model.add(layers.Conv2D(64, (3, 3), activation='relu'))
                model.add(layers.MaxPooling2D((2, 2)))
                if i == num_layers-1:
                        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(9))
        
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
        grid_search = GridSearchCV(estimator=model, param_grid=hyperparameter_grid, n_jobs=7, cv=cv, scoring='f1_macro', error_score=0)
        
        #Execute grid search
        grid_result = grid_search.fit(training_data, training_labels)
        
        #Summarize the results
        print("Model Training Performance:")
        print('------------------------------------------------------------------')
        print(f"F1: {grid_result.best_score_:3f} using {grid_result.best_params_}")
        print('------------------------------------------------------------------')
        
        parameters = grid_result.best_params_
        
        return parameters