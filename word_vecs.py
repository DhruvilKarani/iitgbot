# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 17:42:24 2019

@author: DHRUVIL
"""

import numpy as np
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
STOPWORDS = set(stopwords.words('english'))

path = 'C:/Users/DHRUVIL/Desktop/Website/data/'
data_files = os.listdir("C:/Users/DHRUVIL/Desktop/Website/data")
paragraphs = []
for data_file in data_files:
    with open(path + data_file, 'r' , encoding = 'utf-8') as f:
        content = f.readlines()
    f.close()
    paragraphs+=[line for line in content]




tokens = []
for para in paragraphs:
    para = re.sub(r'https\S+','',para)
    tokens+= word_tokenize(para)
    
''' Exploratory data analysis '''

average_length = np.mean([len(x) for x in tokens])
print("Average length of a token = ",average_length)

punctuations_as_tokens = len([x for x in tokens if x in string.punctuation])
print("Percent of tokens as punctations = ",punctuations_as_tokens/len(tokens)*100)

token_count = Counter(tokens)
print("Top 10 frequent words = ",list(sorted(token_count.items(),key =lambda x: x[1], reverse=True))[:10])
print("Mean frequency of words = ",np.mean(list(token_count.values())))


def wordprocess(word):
    word=word.lower()
    lemmatizer=WordNetLemmatizer()
    word=lemmatizer.lemmatize(word)
    return word
    
    
class Tokens():
    def __init__(self,tokens):
        self.tokens=tokens
    def removebrackets(self):
        ''' Removes {}, [], () and everything 
        in between'''
        _tokens=[]
        _bracketopen = {string:_id for _id, string in enumerate(['{','(','<','['])}
        _bracketclose = {string:_id for _id, string in enumerate(['}',')','>',']'])}
        for i,token in enumerate(self.tokens):
            if token in _bracketopen.keys():
                _idx=_bracketopen[token]
                flag=1
                continue
            if token != _bracketclose[_idx] and flag==1:
                continue
            elif token==_bracketclose[_idx] and flag==1:
                flag=0
                continue
            _tokens.append(token)
        return _tokens
    def preprocess(self):
        _tokens=[]
        remove_punct_except_dot = ''.join([c for c in string.punctuation if c!='.'])
        for token in self.tokens:
            if token not in remove_punct_except_dot and token not in STOPWORDS:
                _tokens.append(wordprocess(token))
        return _tokens
                


processed_tokens = Tokens(tokens)
processed_tokens = processed_tokens.preprocess()
sentences=[]
temp=[]
processed_counts = Counter(processed_tokens)
for token in processed_tokens:
    if token =='.':
        sentences.append(temp)
        temp=[]
    elif processed_counts[token]<100: temp.append(token)
if temp:
    sentences.append(temp)
    
model = gensim.models.Word2Vec(sentences,size=100, min_count=1)






    