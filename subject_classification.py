# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:46:32 2019

@author: DHRUVIL
"""
import numpy as np
import nltk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim
import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
import sys
from scipy import spatial
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from autocorrect import spell
from itertools import chain
lemmatizer=WordNetLemmatizer()

path = 'C:/Users/DHRUVIL/Desktop/iitgchatbot/data_categorized/'
folders = os.listdir(path)

sys.path.append(path)

STOPWORDS = set(stopwords.words('english'))

count=0
datadict={}
for folder in folders:
    documents = []
    data_files = os.listdir(path+folder)
    for data_file in data_files:
#        print(data_file)
        with open(path +folder+ '/'+ data_file, 'r', encoding='utf-8',errors="surrogateescape") as f:
            content = f.readlines()
        f.close()
        documents+=[line for line in content]
#        count+=len(content)
#        print(data_file, len(content))
    datadict[folder]=documents

def remove_non_ascii(text):
    return unidecode(str(text))

tokens = []
for label,document in datadict.items():
    datadict[label] = [re.sub(r'https\S+','',piece) for piece in document]
    datadict[label] = [remove_non_ascii(piece) for piece in document]

#    datadict[label] = [re.sub(r'\\\w{1}','',piece) for piece in document]


#def advancecorrect(word,word_list=word_list):
#    spell_correct = spell(word)
#    synonyms = list(set(synsets(spell_correct)).intersection(word_list))
#    if synonyms:
#        return synonyms[0]
#    return spell_correct


def wordprocess(sentence):
    '''processes a token. lowercasing, lemmatizing'''
    tokenized_sentence = word_tokenize(sentence)
    tokenized_sentence = [lemmatizer.lemmatize(word.lower()) for word in tokenized_sentence]
    tokenized_sentence = [word for word in tokenized_sentence if word not in string.punctuation]
    tokenized_sentence = [word for word in tokenized_sentence if word not in STOPWORDS]

    return tokenized_sentence


sent=[]       
for label, document in datadict.items():
    splitted=' '.join(document).split('.')
    processedtext =[' '.join(wordprocess(line)) for line in splitted if 25>len(wordprocess(line))>3]
    datadict[label]=processedtext
   # print(np.array(processedtext).shape)
    sent+=splitted

        
#tok2id = {tok:idx for idx,tok in enumerate(tokens)}
#id2tok = {idx:tok for idx,tok in tok2id.items()}

y = np.array([0]*len(list(datadict.values())[0]) + [1]*len(list(datadict.values())[1]) +[2]*len(list(datadict.values())[2]))
X = np.array(list(datadict.values())[0]+list(datadict.values())[1]+list(datadict.values())[2])

maxl=[]
for document in datadict.values():
    maxl.append(max([len(l) for l in document]))


# list of text documents
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(X)
# summarize


X_tfidf = vectorizer.transform(X).toarray()
pca =PCA(n_components = 100)
X_pca = pca.fit_transform(X_tfidf)
seed = 7
#test_size = 
#X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size, random_state=seed)
model = XGBClassifier(ets=0.1)
model.fit(X_pca, y)



# make predictions for test data
#y_pred = model.predict(X_test)
#
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

def generator(question, model=model, tfidf=vectorizer, wordprocess=wordprocess, pca = pca ):
    question_tag=['how','when','what','who','why']
    processed_question = [word for word in wordprocess(question) if word not in question_tag]
    processed_question = [' '.join(processed_question)]
    vector=tfidf.transform(processed_question).toarray()
    pca_vector = pca.transform(vector)
    prediction = model.predict(pca_vector)
    
    return folders[prediction[0]],pca_vector


acads_vectors = np.array([vec for idx,vec in zip(y,X_pca) if idx==0])
hostel_vectors = np.array([vec for idx,vec in zip(y,X_pca) if idx==1])
others_vectors = np.array([vec for idx,vec in zip(y,X_pca) if idx==2])

search_vector = {folders[0]:acads_vectors,folders[1]:hostel_vectors,folders[2]:others_vectors}


acads_test = list(datadict.items())[0]
hostel_test = list(datadict.items())[1]
others_test = list(datadict.items())[2]

search_dict = {folders[0]:acads_test, folders[1]:hostel_test, folders[2]:others_test}

question = 'mess in iitg?'
folder, question_vector = generator(question)

search_in_text = list(chain.from_iterable(search_dict[str(folder)]))
search_in_vector = search_vector[str(folder)]
scores = np.array([spatial.distance.cosine(question_vector, answer) for answer in search_in_vector])

print(search_in_text[np.argmax(scores)])




