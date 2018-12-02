# -*- coding: utf-8 -*-
"""
Spyder Editor

Authors:
    Shiyan Deng
    Geoffrey Lucas
    
Revisions:
    First Publicaion
        12/1/2018
        
        
"""

"""
Imports
"""

import numpy 
import pandas 
import sklearn
import nltk
import matplotlib.pyplot as pyplot
import csv
import re
import copy

# nltk.download()
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import reuters
from nltk.corpus import stopwords
#from nltk import word_tokenizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from scipy.sparse import vstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier 

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MultiLabelBinarizer


ros = RandomOverSampler(random_state = 0)
rus = RandomUnderSampler(random_state = 0)
mlb = MultiLabelBinarizer()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Function Definitions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Ugly, but gets the job done
# Takes the NLTK data and creates a binary indicator array for hte categories
# I ended up finding the MultiLabelBinarizer later.  It produces identical output

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

# Courtesy of http://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/counting-and-stemming/
def snowball_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [snow.stem(word) for word in words]
    return words

def underSample(a, b, reduceTo):
    colSums = b.sum(axis =0)
    
    undersampleRows = []
    countAdds = []
    cols = []
    
    for col in range(len(colSums)):
        if colSums[col] > reduceTo:
            temp = numpy.ndarray.copy(colSums)
            for col2 in range(len(temp)):
                if (col2 != col):
                    temp[col2] = 0
                if (col2 == col):
                    temp[col2] = 1
            
            undersampleRows.append(temp)
            countAdds.append(0)
            cols.append(col)
    
    new_a = []
    temp = []
    
    for row in range(len(a)):
        
        match = False
        
        for underRow in range(len(undersampleRows)):
            if numpy.array_equal(b[row,:], undersampleRows[0]):
                match = True
            
        if not(match):
            new_a.append(a[row])
            temp.append(b[row,:])
        else:
            if countAdds[underRow] < reduceTo:
                new_a.append(a[row])
                temp.append(b[row,:])
                countAdds[underRow] = countAdds[underRow]  + 1
    
        # del new_b
        new_b = numpy.stack(temp, axis = 0)
    
    return new_a, new_b       

"""
This isn't randomly oversampling.

I choose against randomly oversampling in this case due to the connections between categories and could easily 
oversaturate the already large categories in some cases
""" 

def overSample(a, b, increaseTo):
    colSums = b.sum(axis = 0)
    oversample = numpy.ndarray.copy(colSums)
    
    for col in range(len(colSums)):
        if colSums[col] < increaseTo:
            oversample[col] = 1
        else:
            oversample[col] = 0

    new_a = copy.deepcopy(a)
    temp = []
    
    for row in range(len(a)):
        for col in range(len(oversample)):
            if b[row,col] == oversample[col] and colSums[col] < increaseTo:
                new_a.append(a[row])
                temp.append(b[row,])
                colSums[col] = colSums[col] + 1
                
    if (temp):
        temp2 = numpy.stack(temp, axis = 0)
        new_b = numpy.concatenate((numpy.ndarray.copy(b), temp2), axis = 0)
        new_a, new_b = overSample(new_a, new_b, increaseTo)
    else:
        new_b = b
    
    new_b.sum(axis = 0)            
    b.sum(axis = 0)                

    return new_a, new_b       

def damnSMOTE(a, b, increaseTo):
    
    from imblearn.over_sampling import SMOTE
    from scipy.sparse import vstack
    smote = SMOTE(random_state = 0, k_neighbors = 4)
    
    colSums = b.sum(axis = 0)
    oversample = numpy.ndarray.copy(colSums)
     
    for col in range(len(colSums)):
        if colSums[col] < increaseTo:
            oversample[col] = 1
        else:
            oversample[col] = 0

    if 'a_new' in dir():
        del a_new
        
    if 'b_new' in dir():
        del b_new

    b_new = []
    
    extraRows = []
    target = []
        
    for col in range(increaseTo):
        extraRows.append(a[0])
        target.append(-1)
    
    for col in range(len(oversample)):
        
        a_temp = []
        b_temp = []
        cat_holder = []
                
        """
        This section accomplishes the SMOTE oversampling if the # of occurences is below the supplied threshold
        
        It basically pads a list with fake entries (SMOTE equalizes the # of entries to the highest) and 
        appends the collected entries from the dataset.  It then generates the synthetic entreis and appends
        them to the master collection.
        """
        
        if oversample[col] == 1:
            for row in range(a.shape[0]):
                if b[row,col] == 1:
                    a_temp.append(a[row])
                    b_temp.append(col)
                    cat_holder.append(b[row])
                    
            temp = extraRows + a_temp
            X_array = vstack(temp)
            Y_array = target + b_temp   
            smote_a, smote_b = smote.fit_sample(X_array, Y_array)
        
            if 'a_new' in dir():
                a_new = vstack((a_new, smote_a[increaseTo:]))
            else:
                a_new = smote_a[increaseTo:]
            
            for row in range(0, len(a_temp)):
                b_new.append(cat_holder[row])
                
            for row in range((increaseTo + len(a_temp)), len(smote_b)):
                b_new.append(cat_holder[0])
    
        else:
            for row in range(a.shape[0]):
                if b[row,col] == 1:
                    a_temp.append(a[row])
                    b_temp.append(b[row])
                    
            X_array = vstack(a_temp)
        
            if 'a_new' in dir():
                a_new = vstack((a_new, X_array))
            else:
                a_new = X_array
            
            for row in range(len(b_temp)):
                b_new.append(b_temp[row])
    
    b_hot = numpy.stack(b_new, axis = 0)
    
    return a_new, b_hot
             
    
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
print(df.head)

# Plot used in presentation
df2 = df.head(10)
df2 = df2.append(df.tail(10))
df2 = df2.sort_values('file_count', ascending = False)
CategoryPlot2 = df2.plot(kind = 'barh', y = "file_count", x = "categories", title = 'Top 10 & Bottom 10 Categories')
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
full_corpus =  [" ".join([t for t in reuters.words(doc_list[t])]) for t in range(len(doc_list))]
print("test_corpus is created, the first line is: {} ...".format(test_corpus[0][:50]))
print("train_corpus is created, the first line is: {} ...".format(train_corpus[0][:50]))

"""
Stemmers
"""

ps = PorterStemmer()
for article in range(len(train_corpus)):
    train_corpus[article] = ps.stem(train_corpus[article])

snow = SnowballStemmer("english")
for article in range(len(train_corpus)):
    train_corpus[article] = snow.stem(train_corpus[article])

# Creating the output shapes for use
Y_Train = Y_Trainer()
Y_Test = Y_Tester()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Creating some Baseline Accuracies

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
vectorizer = TfidfVectorizer(ngram_range = (1,2), tokenizer = snowball_tokenizer, max_features = 50000)

X_Train = vectorizer.fit_transform(train_corpus)
X_Test = vectorizer.transform(test_corpus)

# Now, we're going throw in a little training on KNN
MrRogers = KNeighborsClassifier(n_neighbors = 3)
MrRogers.fit(X_Train, Y_Train)
pred = MrRogers.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

# One vs Rest Classifier with SVM
OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Try out a Neural Network.  This one takes a while, too long for general testing

NN = MLPClassifier()
NN.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)
myScore = accuracy_score(Y_Test, pred)
print("NB accuracy score was: " + str(myScore))
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# One vs Rest Classifier with Decision Tree
Tree = OneVsRestClassifier(DecisionTreeClassifier())
Tree.fit(X_Train, Y_Train)
pred = Tree.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Baselines

Stemmed, bigrams, max features = 50K

KNN (K = 3), with bigrams

Accuracy score was:  0.7413050679032792
F1 Macro Avg was:    0.43039396595950874
Recall score was:    0.40079717231411505
Precision score was: 0.541675454482741

OVR with SVC, with bigrams

Accuracy score was:  0.7992712818814177
F1 Macro Avg was:    0.37654057406208574
Recall score was:    0.30884038249890994
Precision score was: 0.5456678277600946

Decision Treee, with bigrams

Accuracy score was:  0.7065253395163962
F1 Macro Avg was:    0.5529887446053556
Recall score was:    0.5511745138865697
Precision score was: 0.5901060650642271

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""""""""""""""""""""""""""""""""""""""""""""""
Y matrices

Note: They are in list formats so MLB can convert them
"""""""""""""""""""""""""""""""""""""""""""""""

Y_Train_Cats = []
for doc in range(len(doc_list)):
    if doc_list[doc].startswith('train'):
         Y_Train_Cats.append(reuters.categories(doc_list[doc]))
            
Y_Test_Cats = []
for doc in range(len(doc_list)):
    if doc_list[doc].startswith('test'):
        Y_Test_Cats.append(reuters.categories(doc_list[doc]))

"""
Binarizing the labels
"""

# Fit the model for binarizing
Y_Train = mlb.fit_transform(Y_Train_Cats)
# It's okay to binarize the Testing set now, need to add to Training set

Y_Test = mlb.transform(Y_Test_Cats)

"""
Cutting down the list of categories to small ones
These will be the ones we oversample on
Note: Teh OS methods here bring hte smallest up to the ration ofh te majority in the class
"""

category_filter2 = []
for index, row in df.iterrows():
    if row['file_count'] <= 500:
        category_filter2.append(row['categories'])

"""""""""""""""""""""""""""""""""""""""""""""""
Regular OverSampling
"""""""""""""""""""""""""""""""""""""""""""""""

new_a, new_b = overSample(train_corpus, Y_Train, 500)
new_b.sum(axis = 0)

X_Train = vectorizer.fit_transform(new_a)
X_Test = vectorizer.transform(test_corpus)

Y_Train = new_b
mlb.fit(Y_Train_Cats)
Y_Test = mlb.transform(Y_Test_Cats)

MrRogers = KNeighborsClassifier(n_neighbors = 3)
MrRogers.fit(X_Train, Y_Train)
pred = MrRogers.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7469360715468698
F1 Macro Avg was:    0.43768087816037454
Recall score was:    0.43014454752119213
Precision score was: 0.49836124219430444
"""

OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7979463398476316
F1 Macro Avg was:    0.4094575021061617
Recall score was:    0.342660260096975
Precision score was: 0.5853041569643807
"""

Tree = OneVsRestClassifier(DecisionTreeClassifier())
Tree.fit(X_Train, Y_Train)
pred = Tree.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.69195097714475
F1 Macro Avg was:    0.5789644444914774
Recall score was:    0.5803204006216225
Precision score was: 0.6043165892335032

"""

"""""""""""""""""""""""""""""""""""""""""""""""
UnderSampling

Note: This doesn't do as good of a job as some fancier undersamplers do
Note: I'm deleting column 21 as it's too intertwined with other categories and
didn't undersample well.  I forget what it is, maybe 'trade'
"""""""""""""""""""""""""""""""""""""""""""""""

Y_Train = mlb.fit_transform(Y_Train_Cats)
x_us, y_us = underSample(train_corpus, Y_Train, 500)
y_new = numpy.delete(y_us, 21,1)
y_new.sum(axis =0)

X_Train = vectorizer.fit_transform(x_us)
X_Test = vectorizer.transform(test_corpus)

Y_Train = y_new
Y_Test = numpy.delete(Y_Test,21,1)

# Now, we're going throw in a little training on KNN
MrRogers = KNeighborsClassifier(n_neighbors = 3)
MrRogers.fit(X_Train, Y_Train)
pred = MrRogers.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7045379264657171
F1 Macro Avg was:    0.4250157157966988
Recall score was:    0.397377609079728
Precision score was: 0.5270260601740671
"""

# One vs Rest Classifier
OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7903279231533621
F1 Macro Avg was:    0.3718850491730583
Recall score was:    0.30257986905633444
Precision score was: 0.5396764473459428
"""

Tree = OneVsRestClassifier(DecisionTreeClassifier())
Tree.fit(X_Train, Y_Train)
pred = Tree.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7028817489234847
F1 Macro Avg was:    0.5379982440516561
Recall score was:    0.5536374102283341
Precision score was: 0.5521205148650596

"""

"""""""""""""""""""""""""""""""""""""""""""""""
SMOTE OverSampling
"""""""""""""""""""""""""""""""""""""""""""""""

mlb = MultiLabelBinarizer()

"""
I'm SMOTEing hot - TSSSSCH 

I was really tired when I wrote this.  Don't hold it too much against me.
I can't seem to make myself delte it though.
"""

# Ensure there are enough entries for SMOTE (needs at least 2 (default is 5), but I'm creating at least 10)
# Note that this doesn't really help the smaller (< 5 example categories)

Y_Train = mlb.fit_transform(Y_Train_Cats)
# It's okay to binarize the Testing set now, need to add to Training set

Y_Test = mlb.transform(Y_Test_Cats)

forSMOTEX, forSMOTEY = overSample(train_corpus, Y_Train, 10)
forSMOTEX = vectorizer.fit_transform(forSMOTEX)
X_Test = vectorizer.transform(test_corpus)

X_Train_SMOTE, Y_Train_SMOTE = damnSMOTE(forSMOTEX, forSMOTEY, 500)

# Now, we're going throw in a little training on KNN
MrRogers = KNeighborsClassifier(n_neighbors = 3)
MrRogers.fit(X_Train_SMOTE, Y_Train_SMOTE)
pred = MrRogers.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.4368996356409407
F1 Macro Avg was:    0.3304782202214542
Recall score was:    0.4856602198788088
Precision score was: 0.2954683025552146
"""

# One vs Rest Classifier
OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_Train_SMOTE, Y_Train_SMOTE)
pred = OVR.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7618416694269625
F1 Macro Avg was:    0.3996356947927968
Recall score was:    0.3684180637240032
Precision score was: 0.5017517581061008
"""

Tree = OneVsRestClassifier(DecisionTreeClassifier())
Tree.fit(X_Train_SMOTE, Y_Train_SMOTE)
pred = Tree.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.6498840675720438
F1 Macro Avg was:    0.45970018812146085
Recall score was:    0.4967188625693827
Precision score was: 0.4702078470965117
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Web Search Oversampling

Note: The webscraper I created is in another file.  It isn't in this one
as the libraries I used require python 2.7 and a couple of the libraries in here
require python 3.7 and I'm frankly not good enough with python yet to 
easily combine this in a single bit of work

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

with open('Extras_Articles.csv', encoding = "ISO-8859-1") as csvfile:
    readCSV = csv.reader(csvfile, delimiter = ',')
    cats = []
    arts = []
    for row in readCSV:
        arts.append(row[0])
        cats.append(row[1])

train_corpus = [" ".join([t for t in reuters.words(train_doc[t])]) for t in range(len(train_doc))]
for i in range(len(arts)):
   train_corpus.append(arts[i])

from scipy.sparse import vstack

# Stemming the words to try and cut down on training time.
ps = PorterStemmer()
for article in range(len(train_corpus)):
    train_corpus[article] = ps.stem(train_corpus[article])

"""
Y matrices

Note: They are in list formats so MLB can convert them
"""

Y_Train_Cats = []
for doc in range(len(doc_list)):
    if doc_list[doc].startswith('train'):
         Y_Train_Cats.append(reuters.categories(doc_list[doc]))
            
Y_Test_Cats = []
for doc in range(len(doc_list)):
    if doc_list[doc].startswith('test'):
        Y_Test_Cats.append(reuters.categories(doc_list[doc]))

# Fit the model for binarizing
mlb.fit(Y_Train_Cats)
# It's okay to binarize the Testing set now, need to add to Training set

temp_cats = []
for i in range(len(cats)):
   temp_cats.append([cats[i]])
temp = mlb.transform(temp_cats)
Y_Test = mlb.transform(Y_Test_Cats)

Y_Train = Y_Trainer()
Y_Train = vstack((Y_Train, temp))
Y_Train = Y_Train.toarray()

vectorizer = TfidfVectorizer(ngram_range = (1,2), tokenizer = snowball_tokenizer, max_features = 50000)
X_Train = vectorizer.fit_transform(train_corpus)
X_Test = vectorizer.transform(test_corpus)

# KNN Classifier
MrRogers = KNeighborsClassifier(n_neighbors = 3)
MrRogers.fit(X_Train, Y_Train)
pred = MrRogers.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7456111295130838
F1 Macro Avg was:    0.4583441009891684
Recall score was:    0.41637313125518793
Precision score was: 0.5891895899571761
"""

# One vs Rest Classifier
OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7919841006955946
F1 Macro Avg was:    0.4080992746074331
Recall score was:    0.3304576717822968
Precision score was: 0.6184156729080164
"""

# Try out a decision tree. 
Tree = OneVsRestClassifier(DecisionTreeClassifier())
Tree.fit(X_Train, Y_Train)
pred = Tree.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.699238158330573
F1 Macro Avg was:    0.587500649818784
Recall score was:    0.599299345097768
Precision score was: 0.6175743473267834
"""


"""""""""""""""""""""""""""""""""""""""""""""
Combining Regular Oversampling and Web Oversampling

I'm just relying on the incorporation above

"""""""""""""""""""""""""""""""""""""""""""""

new_a, new_b = overSample(train_corpus, Y_Train, 500)
new_b.sum(axis = 0)

X_Train = vectorizer.fit_transform(new_a)
X_Test = vectorizer.transform(test_corpus)

Y_Train = new_b

mlb.fit(Y_Train_Cats)
Y_Test = mlb.transform(Y_Test_Cats)

MrRogers = KNeighborsClassifier(n_neighbors = 3)
MrRogers.fit(X_Train, Y_Train)
pred = MrRogers.predict(X_Test)


print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7462736005299768
F1 Macro Avg was:    0.4553178448675798
Recall score was:    0.4678707816284285
Precision score was: 0.49852987915544317
"""

OVR = OneVsRestClassifier(LinearSVC(random_state=0))
OVR.fit(X_Train, Y_Train)
pred = OVR.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.7962901623053992
F1 Macro Avg was:    0.48190666185231523
Recall score was:    0.40436583467301007
Precision score was: 0.6883178501880015
"""

Tree = OneVsRestClassifier(DecisionTreeClassifier())
Tree.fit(X_Train, Y_Train)
pred = Tree.predict(X_Test)

print("Accuracy score was:  " + str(accuracy_score(Y_Test, pred)))
print("F1 Macro Avg was:    " + str(f1_score(Y_Test, pred, average = 'macro')))
print("Recall score was:    " + str(recall_score(Y_Test, pred, average = 'macro')))
print("Precision score was: " + str(precision_score(Y_Test, pred, average = 'macro')))

"""
Accuracy score was:  0.6886386220602848
F1 Macro Avg was:    0.6063027167013276
Recall score was:    0.6243513159506756
Precision score was: 0.6370432613828773

"""