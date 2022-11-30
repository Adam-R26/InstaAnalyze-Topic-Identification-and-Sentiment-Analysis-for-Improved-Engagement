# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:18:18 2022

@author: adamr
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from HyperparameterGridConfigs import HyperparameterGridConfigs
from tensorflow.keras import callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

#Reproducible results from NNs
tf.random.set_seed(123)

class ModelTrainer:
    def __init__(self, x_train, x_test, y_train, num_classes, img_width=256, img_height=320, pca_variance=0.95):
        self.img_height, self.img_width = img_height, img_width
        self.x_train, self.y_train, = x_train, y_train
        self.x_test  = x_test
        self.num_classes = num_classes
    
        #Prepare Design Matrix for Traditional Models.
        pca = PCA(pca_variance)
        
        #Flatten training data
        trad_x_train = x_train.reshape(x_train.shape[0], -1,)
        trad_x_test = x_test.reshape(x_test.shape[0], -1)
        
        #Apply z-score normalization (Mean:0, SD:1)
        scale= StandardScaler()
        scale.fit(trad_x_train)
        trad_x_train = scale.transform(trad_x_train)
        trad_x_test = scale.transform(trad_x_test)
        
        print(trad_x_train.shape)
        print(trad_x_train.dtype)
        print(trad_x_test.shape)
        
        #Apply dimensionality reduction.            
        pca.fit(trad_x_train)
        self.trad_x_train = pca.transform(trad_x_train)
        self.trad_x_test = pca.transform(trad_x_test)
            
            
    def train_all_models(self):
        '''Pipelines to train all machine learning models and output the results, returns the models in a dictionary.'''        
        #Attain optimal hyperparamters
        print('Tuning Hyperparamters')
        self.best_hyperparamters = self._tune_traditional_hyperparamters(self.trad_x_train, self.y_train)
        
        print('Training Traditional Models')
        #Train traditional models on same train set.
        rf = RandomForestClassifier(random_state=123, n_jobs=7, n_estimators=self.best_hyperparamters['RF']['n_estimators'], max_features=self.best_hyperparamters['RF']['max_features'])
        rf.fit(self.trad_x_train, self.y_train)
        svc = SVC(random_state=123, C=self.best_hyperparamters['SVM']['C'], kernel=self.best_hyperparamters['SVM']['kernel'], probability=True)
        svc.fit(self.trad_x_train, self.y_train)
        
        #Train deep learning models
        print('Training DL Models')
        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
        
        inception = tf.keras.wrappers.scikit_learn.KerasClassifier(
                            self._configure_inception_model,
                            epochs=32,
                            verbose=False)
        history_inception = inception.fit(self.x_train, self.y_train , epochs=32, batch_size=16, verbose=1, validation_split=0.10, callbacks =[earlystopping])
        
        effinet = tf.keras.wrappers.scikit_learn.KerasClassifier(
                            self._configure_efficient_network_model,
                            epochs=32,
                            verbose=False)
        history_effinet = effinet.fit(self.x_train, self.y_train, epochs=32, batch_size=16, verbose=1, validation_split=0.10, callbacks=[earlystopping])
        
        cnn = tf.keras.wrappers.scikit_learn.KerasClassifier(
                            self._configure_cnn_model, last_dense=32, dropout_value=0.3,num_layers=2,
                            epochs=32,
                            verbose=False)
        history_cnn = cnn.fit(self.x_train, self.y_train, epochs=32, batch_size=16, verbose=1, validation_split=0.10, callbacks=[earlystopping])
         
        self.history_objects = {'inception':history_inception.history, 'cnn': history_cnn.history, 'effinet': history_effinet.history}
        self.trained_models = {'svc':svc, 'rf':rf, 'inception': inception, 'cnn':cnn, 'effinet': effinet}
        return self.trained_models
    
    def configure_multiple_ensembles(self, ensemble_list):
        ensemble_dict = {}
        for ensemble in ensemble_list:
            ensemble_name = '_'.join(str(e) for e in ensemble)
            ensemble_dict[ensemble_name] = self.configure_model_ensemble(ensemble)
        return ensemble_dict
            
    def configure_model_ensemble(self, model_names):
        '''Some code for this function inspired from following source:
            Kumar, V. (2017) Votingclassifier: Different feature sets, Stack Overflow. 
            Available at: https://stackoverflow.com/questions/45074579/votingclassifier-different-feature-sets (Accessed: November 10, 2022). '''
    
        model_dict = {'rf':self.trained_models['rf'],
                      'svc': self.trained_models['svc'],
                      'cnn': self.trained_models['cnn'],
                      'inception': self.trained_models['inception'],
                      'effinet': self.trained_models['effinet'],
                      }
        
        #Generate estimator list from params
        fitted_estimators = []
        x_test_list = []
        for model in model_names:
            fitted_estimators.append(model_dict[model])
            if model in ('inception', 'cnn', 'effinet'):
                x_test_list.append(self.x_test)
            else:
                x_test_list.append(self.trad_x_test)
            
            
        label_encoder = LabelEncoder()
        label_encoder.fit(self.y_train)
        
        return [fitted_estimators, label_encoder, x_test_list]
        
        
    def _tune_traditional_hyperparamters(self, x_train, y_train):
        #Get hyperparameter grids
        hyperparameterConfig = HyperparameterGridConfigs()
        
        #Tune RF Hyperparamters
        rf_opt_hyperparam = self._hyperparamter_tuning(RandomForestClassifier(random_state=123, n_jobs=7), hyperparameterConfig.get_rf_hyperparam_grid(), x_train, y_train)
        
        #Tune SVM Hyperparamters
        svm_opt_hyperparam = self._hyperparamter_tuning(SVC(random_state=123, verbose=True, class_weight='balanced', probability=True), hyperparameterConfig.get_svm_hyperparam_grid(), x_train, y_train)
    
        return {'RF': rf_opt_hyperparam, 'SVM': svm_opt_hyperparam}
    
    
    def _configure_cnn_model(self, num_layers=2, dropout_value=0.3, last_dense=64) -> models.Sequential:
        '''Builds CNN model using keras layers.'''
        model = models.Sequential()
        model.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(self.img_height, self.img_width, 3))),
        model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)),
        model.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)),
        for i in range(num_layers):
            if i == 0:
                model.add(layers.Conv2D(32, (3, 3), activation='relu'))#,input_shape=(self.img_height, self.img_width, 3)
                model.add(layers.MaxPooling2D((2, 2)))
                model.add(layers.Dropout(dropout_value))
            else:
                model.add(layers.Conv2D(64, (3, 3), activation='relu'))
                model.add(layers.MaxPooling2D((2, 2)))
                if i == num_layers-1:
                        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            
        model.add(layers.Dropout(dropout_value)),
        model.add(layers.Flatten())
        model.add(layers.Dense(last_dense, activation='relu'))
        #model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        model.summary()
        return model
    
    def _configure_inception_model(self):
            model = tf.keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(self.img_height, self.img_width, 3),
                pooling=None,
                classifier_activation="relu",
            )
            
            model.trainable = False
            augmented_model = models.Sequential()
            augmented_model.add(model)
            augmented_model.add(layers.GlobalAveragePooling2D())
            augmented_model.add(layers.Dropout(0.5))
            augmented_model.add(layers.Dense(self.num_classes, 
                                activation='softmax'))
            
            augmented_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          metrics=['accuracy'])
            #Learning rate: 0.01
            #model.summary()
            return augmented_model
        
    
    def _configure_efficient_network_model(self):
        '''Code for this function inspired by the following tensorflow documentation:
            Fu, Y. (2020) Keras documentation: Image Classification via fine-tuning with EfficientNet, Keras. 
            Available at: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ (Accessed: November 13, 2022).'''
        img_augmentation = models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.experimental.preprocessing.RandomFlip(),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
        )
        inputs = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = img_augmentation(inputs)
        model = tf.keras.applications.EfficientNetB1(include_top=False, input_tensor=x, weights="imagenet")#drop_connect_rate=0.4
    
        # Freeze the pretrained weights
        model.trainable = True
    
        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)
    
        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="pred")(x)
        
        for layer in model.layers:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        
        for layer in model.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
            
        
        # Compile
        model = tf.keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)#ie-2
        model.compile(
            optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        return model
            
    def _hyperparamter_tuning(self, model: ClassifierMixin, hyperparameter_grid: dict, training_data: np.array, training_labels: np.array) -> dict:
        '''Optimizes hyperparamters for inputted model using repeated stratified k-folds cross validation, returning these optimal hyperparameters'''
        #Set up grid search with 5 fold cross validation.
        cv = KFold(n_splits=4)
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