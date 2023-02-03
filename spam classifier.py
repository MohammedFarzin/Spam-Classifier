# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:48:59 2023

@author: FARZIN
"""

import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['labels', 'message'])


import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


stemmer = PorterStemmer()
corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', " ", messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer()
X = bow.fit_transform(corpus)


y = pd.get_dummies(messages['labels'])
y = y.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_model.predict(X_test)

from sklearn.metrics import confusion_matrix

confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)