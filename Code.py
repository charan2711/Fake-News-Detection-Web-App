#!/usr/bin/env python
# coding: utf-8

# In[9]:


#IMPORTING ALL LIBRARIES

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# In[11]:


#Reading the data

news=pd.read_csv(r"C:\Users\CHARAN R\Desktop\4TH SEM\IT290 - Seminar\news.csv")
X = news['text']
y = news['label']


# In[12]:


#Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[13]:


#Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('nbmodel', MultinomialNB())])


# In[15]:


#Training the data

pipeline.fit(X_train, y_train)

#Predicting the label for the test data

pred = pipeline.predict(X_test)


# In[16]:


#Checking the performance of our model

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))


# In[17]:


#Serialising the file

with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




