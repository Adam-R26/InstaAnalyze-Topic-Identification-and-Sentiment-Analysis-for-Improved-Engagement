# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:28:38 2022

@author: adamr
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
import pandas as pd
import re
import string

class SentimentClassifier:
    def __init__(self, opinion_lexicon_pos_path, opinion_lexicon_neg_path):
        self.opinion_lexicon_pos_path = opinion_lexicon_pos_path
        self.opinion_lexicon_neg_path = opinion_lexicon_neg_path
        self.return_int=False
       
    def classify_sentiment(self, comment: str, method='vader'):
        if method == 'vader':
            return self.get_vader_sentiment(comment)
        elif method == 'afinn':
            return self.get_afinn_sentiment(comment)
        elif method == 'textblob':
            return self.get_textblob_sentiment(comment)
        elif method == 'opinion lexicon':
            return self.get_opinion_lexicon_sentiment(comment)
    
    def get_vader_sentiment(self, comment: str):
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(comment)
        sentiment = self._get_sentiment_from_polarity(sentiment_dict['compound'])
        return sentiment, sentiment_dict['compound']
    
    def get_textblob_sentiment(self, sentence):
        text_blob = TextBlob(sentence)
        polarity = text_blob.sentiment[0]
        sentiment = self._get_sentiment_from_polarity(polarity)
        return sentiment, polarity
    
    def get_afinn_sentiment(self, sentence):
        afn = Afinn()
        polarity = afn.score(sentence)
        polarity = polarity/5
        sentiment = self._get_sentiment_from_polarity(polarity, 0.05)
        return sentiment, polarity
    
    def get_opinion_lexicon_sentiment(self, sentence):
        #Init counters for positive and negative words.
        positive_count = 0
        negative_count = 0
        
        #Read in positive and negative terms
        neg_words = pd.read_csv(self.opinion_lexicon_neg_path)['Words'].tolist()
        pos_words = pd.read_csv(self.opinion_lexicon_pos_path)['Words'].tolist()
        
        #Clean sentence of punctuation and make all elements lower case.
        sentence = self._pre_process_string(sentence)
        
        #Tokenize
        sentence = sentence.split()
        sentence = [i.strip() for i in sentence]
        
        #Count number of negative occurances
        for word in sentence:
            for neg_word in neg_words:
                if word == neg_word:
                    negative_count+=1
                    
        #Count number of positive occurances
        for word in sentence:
            for pos_word in pos_words:
                if word == pos_word:
                    positive_count+=1
                    
        #Normalise and return score
        polarity = (positive_count - negative_count)/len(sentence)
        
        #Get sentiment from polarity
        sentiment = self._get_sentiment_from_polarity(polarity)
        return sentiment, polarity
        
    def _get_sentiment_from_polarity(self, polarity, threshold=0.05):
        if polarity >= threshold :
            sentiment = "Positive"
     
        elif polarity <= -threshold :
            sentiment = "Negative"
     
        else :
            sentiment = 'Neutral'
        return sentiment
        
    def _pre_process_string(self, text: str) -> str:
        '''Function pre-processes text into a form that can go into the lemmatization step.'''
        #Lower case to match format or neg and pos words.
        text = text.lower()
        
        #Remove punctuation only leaving behind words.
        text = re.sub('[%s]' % re.escape(string.punctuation.replace("-","")), '', text)
        return text
        