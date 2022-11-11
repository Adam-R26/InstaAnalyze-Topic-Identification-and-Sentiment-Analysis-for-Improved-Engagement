#Import required libraries
import pandas as pd 
import numpy as np
import os
import cv2
import ast
import tensorflow as tf

class DatasetBuilder:
    def __init__(self, lookup_link, mapping_path, image_folder_path):
        self.set_influencer_lookup(lookup_link)
        self.set_mapping_file(mapping_path)
        self.set_image_folder(image_folder_path)
        
    def set_influencer_lookup(self, link):
        #Format link to lookup, and fetch from web.
        self.lookup_link = 'https://drive.google.com/uc?export=download&id='+link.split('/')[-2]
        influencer_lookup = pd.read_csv(self.lookup_link, delimiter='\t')
        
        #Wrangle name to make consistent with json lookup.
        influencer_lookup = influencer_lookup.rename({'Username':'influencer_name'},axis=1)
        self.influencer_lookup = influencer_lookup
    
    def set_mapping_file(self, mapping_path):
        mapping_path = r'C:\Users\adamr\Documents\UniversityWork\COMP591\Data\JSON-Image_files_mapping.txt'
        mapping_file = pd.read_csv(mapping_path, delimiter='\t')
        mapping_file.columns = mapping_file.columns.str.replace(' ', '')
        mapping_file['Image_file_name'] = mapping_file['Image_file_name'].apply(lambda x: ast.literal_eval(x)[0])
        self.mapping_file = mapping_file
        
    def set_image_folder(self, image_folder_path):
        self.img_folder_path = image_folder_path
        self.raw_img_names = os.listdir(image_folder_path)
        self.fmt_img_names = [i.split("-")[1] for i in self.raw_img_names]
        
    def generate_dataset(self, n_class_samples, method='popularity', return_info=True, img_height=200, img_width=200):
        #Perform inner joins on influencer lookup, mapping_file and take set of these for which we have images.
        df = pd.merge(self.mapping_file, self.influencer_lookup, on='influencer_name')
        df = df[df['Image_file_name'].isin(self.fmt_img_names)].copy()
        
        #Extract data using method of sampling.
        df = self.perform_sampling_procedure(df, method, n_class_samples)
        
        #Get related images using ids.
        df['ImageData'] = self.image_mapper(df['Image_file_name'].tolist(), (img_height,img_width))
        
        non_df_shape = self.image_mapper(df['Image_file_name'].tolist(), (img_height,img_width))
        print('Non DF Shape:', non_df_shape.shape)
        print('DF Shape', np.array(tf.convert_to_tensor(df['ImageData']).shape))
        
        #Separate dataset from info
        dataset = df[['ImageData', 'Category']].copy()
        dataset['ImageData'] = tf.convert_to_tensor(dataset['ImageData'], dtype=tf.float32)
        
        #Label encoding for dataset
        encoders={k: v for v, k in enumerate(list(df['Category'].unique()))}
        dataset['Category'] = dataset['Category'].apply(lambda x: encoders[x])
        
        
        #Return desired information
        if return_info:
            return dataset, df, encoders
        else:
            return dataset, encoders
        
    def perform_sampling_procedure(self, df, method, n_class_samples):
        output = []
        
        for value in df['Category'].unique():
            #Isolate specific category data.
            category_df = df[df['Category']==value].copy()
            
            #Perform appropriate sampling procedure.
            if method == 'random sampling':
                output.append(category_df.sample(n_class_samples, random_state=123)) #Reproducibility
                
            elif method == 'popularity':
                category_df = category_df.sort_values('#Followers', ascending=False)
                output.append(category_df[0:n_class_samples])
            
            elif method == 'stratification':
                category_df['Follower Strata'] = pd.qcut(category_df['#Followers'], 10)
                for strata in category_df['Follower Strata'].unique():
                    strata_sample = category_df[category_df['Follower Strata']==strata].copy().sample(n_class_samples/10, random_state=123)
                    output.append(strata_sample)
        
        output = pd.concat(output)
        return output
        
    def image_mapper(self, img_ids, resize=(200,200)):
        img_data_array = []
        
        for i in range(len(self.fmt_img_names)):
            for img_name in img_ids:
                if self.fmt_img_names[i] == img_name:
                    image_path= os.path.join(self.img_folder_path, self.raw_img_names[i])
                    image= cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                    if resize:
                        image=cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
                    image=np.array(image)
                    image = image.astype('float32')
                    image /= 255 
                    img_data_array.append(image)
                    print(image.shape)
        
        return np.array(img_data_array, np.float32)
        