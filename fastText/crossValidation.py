# -*- coding: utf-8 -*-

import fasttext
from nltk.corpus import reuters
import numpy as np

def crossValidation(folds, data, label):
    train = data['train']
    train_label = label['train']

    proportion = len(train) / folds

    for i in range(0, folds - 1):
        f = open('./crossValidation/cross_' + str(i), 'w+')
        for j in range(i * proportion, (i + 1) * proportion):
            for cat in train_label[j]:
                f.write('__label__' + cat + ' ')
            text = train[j]
            text.replace('\t', ' ').replace('\n', ' ')
            f.write(u' '.join(text.split()).encode('utf-8') + '\n');
        f.close()

    f = open('./crossValidation/cross_' + str(folds - 1), 'w+')
    for j in range((folds - 2) * proportion, len(train)):
        for cat in train_label[j]:
            f.write('__label__' + cat + ' ')
        text = train[j]
        text.replace('\t', ' ').replace('\n', ' ')
        f.write(u' '.join(text.split()).encode('utf-8') + '\n');
    f.close()
