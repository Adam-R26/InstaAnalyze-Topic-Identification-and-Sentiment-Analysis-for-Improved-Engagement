# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:58:36 2022

@author: adamr
"""
#Stop CUDA from taking over
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Import other required libraries
from DatasetBuilderV2 import DatasetBuilderV2
from ModelTrainerV2 import ModelTrainerV2
from SentimentClassifier import SentimentClassifier
from tensorflow.keras import callbacks
import time

#Global variables
lookup_link= r'https://drive.google.com/file/d/1vYd_Q9hdnssV2xWT8694v-j3KodWwrUr/view'
image_folder_path = r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\image'
mapping_path = r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\JSON-Image_files_mapping.txt'
opinion_lexicon_neg_path = r'C:\Users\adamr\Documents\UniversityWork\COMP591\Opinion Lexicon\negative-words.txt'
opinion_lexicon_pos_path = r'C:\Users\adamr\Documents\UniversityWork\COMP591\Opinion Lexicon\positive-words.txt'

#def main():
start_time = time.time()
builder = DatasetBuilderV2(lookup_link, mapping_path, image_folder_path)
x_train, x_test, y_train, y_test, encoders, df = builder.generate_dataset(1000)

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 4, 
                                        restore_best_weights = True)
            
trainer = ModelTrainerV2(x_train, x_test, y_train, y_test)
traditional_models = trainer.train_all_models()
# cnn = trainer._configure_cnn_model(2)
# cnn.fit(x_train, y_train , epochs=35, batch_size=32, verbose=1, validation_split=0.12, callbacks =[earlystopping])

print("--- %s seconds ---" % (time.time() - start_time))
    

#main()
    
    
    
    
    
# sentimentClassifier = SentimentClassifier(opinion_lexicon_pos_path, opinion_lexicon_neg_path)
# setniment = sentimentClassifier.classifiy_sentiment('The movie was very bad', 'vader')








