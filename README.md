# InstaAnalyze: Topic Indentification and Sentiment Analysis for Improved Engagement
### Summary
- An End-to-End Machine Learning Pipeline, Leveraging the Fine-Tuning of Large Pre-Trained Computer Vision Networks. (Inception V3, Efficient Network (B1))
- Consists of Multi Class Classification (8 Class) Classification of Instagram Photos as either Beauty, Family, Fashion, Fitness, Food, Interior, Pet and Travel.
- As well as the use of Lexicon Based Classifiers: VADER, AFINN, Opinion Lexicon, TextBlob.
- Enables the topics of instagram users to be identified, as well as the sentiment towards them via sentiment analysis of comments section.
- As well as this a dashboard in plotly was developed to showcase the findings.

### Approach

#### Dataset Generation
- Initially the plan was to use a dataset of instagram images provided by Seungbae Kim from the University of South Florida to train the classifiers. However, upon investigation of this dataset it was realised that the labels provided within it corresponded only to Instagram usernames and not individual posts. Anecdotally it was hypothesized that this could induce mass mislabelling for the given application, hence to gauge the magnitude of the implications this would have on trained model performance. It was therefore decided to train the models on this dataset using three different methods to extract training sets to assess this.

- The first approach was to use random sampling to select 1000 images from each class to form the training dataset. The second approach was to use a popularity heuristic and take 1000 images from the most popular influencers within each category. Finally, the third approach was to split the set of images for each class into a set of 10 strata based upon the number of followers the accounts had, and to equally sample from these strata, to try to get the most representative sample of the population. The results of the model training showed that these all of these procedures were ineffective at achieving accurate ground truth labels, with the accuracy of the best trained classifier (inception V3) being 45%, which was produced on the training dataset formed using the popularity heuristic.

- Armed with this information, the literature related to this particular dataset was re-examined and it was found that the curators had also attempted to create classifiers to predict the classes of individual photos. But in order to do so they had to manually label 1000 images from each class as well as add two more classes to the dataset namely, product and other. Inspired by this it was decided to manually label images, however due to the time constraints of the 
project our approach was to stick to the original 8 categories and to manually label as many of the Instagram posts as possible and supplement some classes with photos from Kaggle house interior dataset (Reni, 2020), and photos collected from the non-copyright platform unsplashed to speed up the labelling process. Note that the reason for collecting Unsplashed photos for some categories rather than just manually Instagram photos, was that because of the huge variation we observed between influencer and photo labels. This meant that using the method of manually labelling the Instagram photos resulted in around one in every five photos being of a desired category, whereas the probability of the photos being labelled correctly when taken from Unsplashed was higher than this. Meaning that for any photos taken from unsplashed the process of labelling involved more of removing outliers rather than manual labelling one by one. Employing this approach, a new dataset consisting of 8000 images; 1000 images for each of the 8 original classes was created.

- To test the effectiveness of our new dataset we trained our modelson it and found the accuracy of our best classifier increased to 87% from 45%, an increase of 42% overall. This <b>highlights the importance of the quality of the training dataset on overall model performance</b> and shows that in general the best way to improve one’s models is to improve the quality of the data underpinning them.

#### Image Pre-Processing
- Due to the different sources used to generate the dataset, and variation in size some platforms would allow, the images needed to be resized for classification. To do this, the frequency of different sizes of images in the dataset was analyzed and the size that most accurately represented most of the dataset was chosen. Using the figures below.
- Considering that the project is Instagram photo focused, and most of the population lied between 251-300 pixels in width, we decided the appropriate size for width to be the median pixel width of 256 pixels. Moreover, an equivalent analysis was carried out for image height to determine that the best compromise there was the median height of 320 pixels.

<img width="1300" alt="image" src="https://github.com/Adam-R26/InstaAnalyze-Topic-Identification-and-Sentiment-Analysis-for-Improved-Engagement/assets/53123097/5eda0c39-4e2b-4877-942d-00d1a10fa4bb">
<img width="1300" alt="image" src="https://github.com/Adam-R26/InstaAnalyze-Topic-Identification-and-Sentiment-Analysis-for-Improved-Engagement/assets/53123097/6f2c8b5a-94bf-47e7-8d79-6c8fb2b29463">

#### Model Training and Evaluation 
A number of models were trained for image classification task and their macro metrics are reported below:
<img width="1300" alt="image" src="https://github.com/Adam-R26/InstaAnalyze-Topic-Identification-and-Sentiment-Analysis-for-Improved-Engagement/assets/53123097/9163d47a-6ac1-4f26-98ed-502bf7366980">




