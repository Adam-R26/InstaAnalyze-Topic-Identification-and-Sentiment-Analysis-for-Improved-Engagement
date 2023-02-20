# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 19:22:16 2022

@author: adamr
"""

from SentimentClassifier import SentimentClassifier

#NB: Paths only required if you want to use method='opinion lexicon' otherwise can just pass dummy strings.
opinion_lexicon_neg_path = 'negative-words.txt'
opinion_lexicon_pos_path = 'positive-words.txt'

classifier = SentimentClassifier(opinion_lexicon_pos_path, opinion_lexicon_neg_path)

negative_sentence = 'God that movie was absolutely awful!'
positive_sentence = 'Wow that an amazing movie!'
neutral_sentence = 'it was alright'

print('Negative')
print('-------------------------------------------------------------------------')
print(classifier.classify_sentiment(negative_sentence, method='vader'))
print(classifier.classify_sentiment(negative_sentence, method='afinn'))
print(classifier.classify_sentiment(negative_sentence, method='textblob'))
print(classifier.classify_sentiment(negative_sentence, method='opinion lexicon'))
print('-------------------------------------------------------------------------')
print()
print('Positive')
print('-------------------------------------------------------------------------')
print(classifier.classify_sentiment(positive_sentence, method='vader'))
print(classifier.classify_sentiment(positive_sentence, method='afinn'))
print(classifier.classify_sentiment(positive_sentence, method='textblob'))
print(classifier.classify_sentiment(positive_sentence, method='opinion lexicon'))
print('-------------------------------------------------------------------------')
print()
print('Neural')
print('-------------------------------------------------------------------------')
print(classifier.classify_sentiment(neutral_sentence, method='vader'))
print(classifier.classify_sentiment(neutral_sentence, method='afinn'))
print(classifier.classify_sentiment(neutral_sentence, method='textblob'))
print(classifier.classify_sentiment(neutral_sentence, method='opinion lexicon'))
print('-------------------------------------------------------------------------')
print()



