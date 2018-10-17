# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Imports
"""

import numpy 
import pandas 
import sklearn
import nltk
import sklearn
import matplotlib.pyplot as pyplot

# nltk.download()
from nltk.corpus import reuters
from nltk.corpus import stopwords
#from nltk import word_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Function Definitions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Ugly, but gets the job done
# Takes the NLTK data and creates a binary indicator array for hte categories
def Y_Trainer():
    y_train = [[0 for x in range(90)] for y in range(7769)]
    x = 0
       
    # Cycle through each doc and check for training
    for doc in range(len(doc_list)):
        if doc_list[doc].startswith('training'):
            # If it is training reset Y to 0 and cycle through categories
            for cat in range(len(categories)):
                if categories[cat] in reuters.categories(doc_list[doc]):
                    y_train[x][cat] = 1
            x += 1
            
    return numpy.asarray(y_train)

def Y_Tester():
    y_test = [[0 for x in range(90)] for y in range(3019)]
    x = 0
    
    # Cycle through each doc and check for training
    for doc in range(len(doc_list)):
        if doc_list[doc].startswith('test'):
            # If it is training reset Y to 0 and cycle through categories
            for cat in range(len(categories)):
                if categories[cat] in reuters.categories(doc_list[doc]):
                    y_test[x][cat] = 1
            x += 1
            
    return numpy.asarray(y_test)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Show Basic Stats for Reuters (Mod Apte Split)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("The reuters corpus has {} tags".format(len(reuters.categories())))
print("The reuters corpus has {} documents".format(len(reuters.fileids())))

# create counter to summarize
categories = []
file_count = []

documents = reuters.fileids()

# count each tag's number of documents
for i in reuters.categories():
    file_count.append(len(reuters.fileids(i)))
    categories.append(i)

# create a dataframe out of the counts
df = pandas.DataFrame( {'categories': categories, "file_count": file_count}).sort_values('file_count', ascending = False)
print(df.head())

# For later if we decide to do some filtering
# category_filter = df.iloc[1:4, 0].values.tolist()

# Examining the distribution of categories
# This plot is realllllllly busy
CategoryPlot = pyplot.barh(df.loc[:,"categories"], df.loc[:,"file_count"])

# Reduce the number temporarily
category_filter2 = []

for index, row in df.iterrows():
    if row['file_count'] >= 50:
        category_filter2.append(row['categories'])
        
df2 = df[df.categories.isin(category_filter2)].sort_values('file_count', ascending = False)

CategoryPlot2 = df2.plot(x = df2.loc[:,"categories"], kind = 'barh', title = 'Reduced Reuters (Only Categories >= 50 instances)')
CategoryPlot2.invert_yaxis()

# Create lists of test and training docs
doc_list = numpy.array(reuters.fileids())
test_doc = doc_list[['test' in x for x in doc_list]]
train_doc = doc_list[['training' in x for x in doc_list]]
print("test_doc is created with following document names: {} ...".format(test_doc[0:5]))
print("train_doc is created with following document names: {} ...".format(train_doc[0:5]))

# Create the corpus for later use
test_corpus = [" ".join([t for t in reuters.words(test_doc[t])]) for t in range(len(test_doc))]
train_corpus = [" ".join([t for t in reuters.words(train_doc[t])]) for t in range(len(train_doc))]
print("test_corpus is created, the first line is: {} ...".format(test_corpus[0][:100]))
print("train_corpus is created, the first line is: {} ...".format(train_corpus[0][:100]))

# Create a vectorizer (NOT CURRENTLY USING.  PLAY WITH IN A BIT)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_corpus)
X_test_counts = count_vect.fit_transform(test_corpus)
print("Reuters Training BOW Matrix shape:", X_train_counts.shape)
print("Reuters Training BOW Matrix shape:", X_test_counts.shape)

# Following in the footsteps of Sean
# Warning: Very long
print(count_vect.vocabulary_)

# Creating the output shapes for use
Y_Train = Y_Trainer()
Y_Test = Y_Tester()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Creating some Baseline Accuracies

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Let's vectorize this stuff.  WOOHOO!!!
stop_words = stopwords.words("english")
vectorizer = HashingVectorizer(stop_words = stop_words, alternate_sign = False)
X_Train = vectorizer.transform(train_corpus)
X_Test = vectorizer.transform(test_corpus)

# Now, we're going throw in a little training on KNN
MrRogers = KNeighborsClassifier(n_neighbors = 5)
MrRogers.fit(X_Train, Y_Train)
pred = MrRogers.predict(X_Test)
myScore = accuracy_score(Y_Test, pred)
print("KNN accuracy score was: " + str(myScore))

# One vs Rest Classifier
OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)
myScore = accuracy_score(Y_Test, pred)
print("OVR accuracy score was: " + str(myScore))

# Try out a Neural Network.  This one takes a while.
NN = MLPClassifier()
NN.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)
myScore = accuracy_score(Y_Test, pred)
print("NB accuracy score was: " + str(myScore))

# Try out a decision tree. 
Tree = DecisionTreeClassifier()
Tree.fit(X_Train, Y_Train)
pred = Tree.predict(X_Test)
myScore = accuracy_score(Y_Test, pred)
print("NB accuracy score was: " + str(myScore))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Random Working Area

""""""""""""""""""""""""""""""""""""""""""""""""""



reuters.words(categories = ['acq', 'money-fx', 'grain'])

# Based on NLTK tutorials
all_words = nltk.FreqDist(w.lower() for w in reuters.words())
word_features = list(all_words)[:2000]
print(word_features)

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def target_vars(document):
    categories = {}
    for i in len(test_doc):
        print(reuters.categories[test_doc])


print(reuters.words('training/3482'))
print(reuters.raw('training/3482'))
print(reuters.categories('training/3482'))

"""