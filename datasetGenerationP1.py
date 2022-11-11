# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:31:10 2022

@author: adamr
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import ast

#Get appropriate link to data lookup.
lookup_link='https://drive.google.com/file/d/1vYd_Q9hdnssV2xWT8694v-j3KodWwrUr/view'
formatted_link_influencer_lookup= 'https://drive.google.com/uc?export=download&id='+lookup_link.split('/')[-2]


image_folder = os.listdir(r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\image')
image_folder = [i.split("-")[1] for i in image_folder]
#print(image_folder)

#Read in lookup from web.
influencer_lookup = pd.read_csv(formatted_link_influencer_lookup, delimiter='\t')
influencer_lookup = influencer_lookup.rename({'Username':'influencer_name'},axis=1)
#Read in mapping file
mapping_file = pd.read_csv(r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\JSON-Image_files_mapping.txt', delimiter='\t')
#print(mapping_file.columns)
mapping_file.columns = mapping_file.columns.str.replace(' ', '')
mapping_file['Image_file_name'] = mapping_file['Image_file_name'].apply(lambda x: ast.literal_eval(x)[0])
mapping_file = mapping_file[mapping_file['Image_file_name'].isin(image_folder)]



#Perform inner join on dataset
df = pd.merge(influencer_lookup, mapping_file, on='influencer_name')


#Investigate number of samples for each category
df['Category'].value_counts().plot(kind='barh')
print(df['Category'].value_counts())


#Create dataset for classifiers
df_rand = [] #Store samples from each category 
df_follower_strata = []
df_popularity = []


for value in df['Category'].unique():
    category_df = df[df['Category']==value].copy()
    df_rand.append(category_df.sample(1000, random_state=123)) #Reproducibility
               

for value in df['Category'].unique():
    category_df = df[df['Category']==value].copy()
    category_df['Follower Strata'] = pd.qcut(category_df['#Followers'], 10)
    print(category_df['Follower Strata'].nunique())
    for strata in category_df['Follower Strata'].unique():
        strata_sample = category_df[category_df['Follower Strata']==strata].copy().sample(100)
        df_follower_strata.append(strata_sample)
    
for value in df['Category'].unique():
    category_df = df[df['Category']==value].copy().sort_values('#Followers', ascending=False)
    df_popularity.append(category_df[0:1000]) #Reproducibility
               

df_rand = pd.concat(df_rand)    
df_follower_strata = pd.concat(df_follower_strata)
df_popularity = pd.concat(df_popularity)


image_files = [i.split("-")[1] for i in os.listdir(r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\image')]
main_list = list(set(df_rand['Image_file_name'].tolist()) - set(image_files))

#Read in images
IMG_HEIGHT, IMG_WIDTH = 256, 256

unmatched = []

def create_dataset(img_folder, img_ids):
    img_data_array = []
    image_files = [i.split("-")[1] for i in os.listdir(img_folder)]
    raw_image_files = os.listdir(img_folder)
    
    # for i in range(len(image_files)):
    #     if image_files[i] in img_ids:
    #         image_path= os.path.join(img_folder, raw_image_files[i])
    #         #print(image_path)
    #         image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
    #         #image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
    #         image=np.array(image)
    #         image = image.astype('float32')
    #         image /= 255 
    #         img_data_array.append(image)    
    

    for i in range(len(image_files)):
        found = False
        for img_name in img_ids:
            if image_files[i] == img_name:
                image_path= os.path.join(img_folder, raw_image_files[i])
                #print(image_path)
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                #image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                image=np.array(image)
                image = image.astype('float32')
                image /= 255 
                img_data_array.append(image)
                found=True
        if found == False:
            unmatched.append(img_name)
            
    
            
    
    return img_data_array

images = create_dataset(r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\image', df_rand['Image_file_name'].tolist())
print(images)
print(images[0])
print('hELLO')
plt.imshow(images[0])
plt.show()

    
    


