# -*- coding: utf-8 -*-

import fasttext
from loadData import loadData
from nltk.corpus import reuters

doc_id = reuters.fileids()
train_id = [d for d in doc_id if d.startswith('training/')]
test_id = [d for d in doc_id if d.startswith('test/')]
data, label = loadData()

for cat in reuters.categories():
    train = open('./training/' + cat + '_train', 'w+')
    num = 0
    for d in train_id:
        if cat in reuters.categories(d):
            num += 1
    for d in train_id:
        if cat in reuters.categories(d):
            train.write('__label__' + cat)
        elif num > 0:
            train.write('__label__NOT' + cat)
            num -= 1
        text = reuters.raw(d)
        text.replace('\t', ' ').replace('\n', ' ')
        train.write(' ' + u' '.join(text.split()).encode('utf-8') + '\n')

    train.close()

    clf = fasttext.supervised('./training/' + cat + '_train', './model_OneVSAll/' + cat + '.model', label_prefix = '__label__')
