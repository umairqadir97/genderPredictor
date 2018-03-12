# -*- coding: utf-8 -*-
import random
import string
import math
import numpy as np
import argparse
import time
import nltk
import pickle
from nltk.classify import apply_features
import pandas as pd

DATASET_FN = 'arabic.xlsx'
TRAIN_SPLIT = 0.85
VAL_SPLIT = 0
alif = unicode("ىءهونملقكفغعظطضصشسرذدخحجثتبا", "utf-8")
ALL_LETTERS = string.ascii_lowercase + alif
ALL_GENDERS = ["male", "female"]
N_LETTERS = len(ALL_LETTERS)
N_GENDERS = len(ALL_GENDERS)
weight_file = 'naive_weights'


def features(name):
    features_list = {}
    if name:
        features_list["last_letter"] = name[-1]
        features_list["first_letter"] = name[0]
        for letter in ALL_LETTERS:
            features_list["count(%s)" % letter] = name.count(letter)
            features_list["has(%s)" % letter] = (letter in name)
        #names ending in -yn are mostly female, names ending in -ch ar mostly male, so add 2 more features
        features_list["suffix2"] = name[-2:]
        features_list["suffix3"] = name[-3:]
        features_list["suffix4"] = name[-4:]
    return features_list


def load_dataset(filename=DATASET_FN):
    names = []
    genders = []
    data = pd.read_excel(filename, encoding='utf-8')
    for i in range(len(data) - 1):
        names.append(data.iloc[i][0])
        genders.append(data.iloc[i][1])
    namelist = list(zip(names, genders))
    #random.shuffle(namelist)
    return namelist


def dataset_dicts(dataset=load_dataset()):
    name_gender = {}
    gender_name = {}
    for name, gender in dataset:
        name_gender[name] = gender
        gender_name.setdefault(gender, []).append(name)
    return name_gender, gender_name


def split_dataset(train_pct=TRAIN_SPLIT, val_pct=VAL_SPLIT, filename=DATASET_FN):
    dataset = load_dataset(filename)
    n = len(dataset)
    tr = int(n * train_pct)
    va = int(tr + n * val_pct)
    return dataset[:tr], dataset[va:]  # Trainset, Valset, Testset


def time_since(since):
    now = time.time()
    s = now - since
    hours, rem = divmod(now-since, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}h {:0>2}m {:0>2}s".format(int(hours), int(minutes), int(seconds))


# Han ve yara time thi gya....model train kar gino
def train(trainset, weight=weight_file):
    start = time.time()
    print("Training Naive Bayes Classifer on %d examples (%s)" % (len(trainset), time_since(start)))
    trainset = apply_features(features, trainset, labeled=True)
    classifier = nltk.NaiveBayesClassifier.train(trainset)
    print("Training complete. (%s)" % (time_since(start)))
    # save weights
    with open(weight, 'wb') as f:
        pickle.dump(classifier, f)
        f.close()
        print('Weights saved to "%s"' % weight)


# TEsting karo bhai oye....training Thi gye !
def test(testset, weight=weight_file):
    start = time.time()
    classifier = load_classifier(weight)
    print("Testing Naive Bayes Classifer on %d examples (%s)" % (len(testset), time_since(start)))
    testset = apply_features(features, testset, labeled=True)
    acc = nltk.classify.accuracy(classifier, testset)
    print("Testing accuracy is %.2f%% on %d examples (%s)" % (acc * 100, len(testset), time_since(start)))
    return acc


def load_classifier(weight=weight_file, verbose=True):
    with open(weight, 'rb') as f:
        classifier = pickle.load(f)
        f.close()
    if verbose: print('Loaded weights from "%s"...\n' % weight_file)
    return classifier


def _classify(name, classifier, verbose=True):
    _name = features(name.lower())
    dist = classifier.prob_classify(_name)
    m, f = dist.prob("male"), dist.prob("female")
    d = {m: "male", f: "female"}
    prob = max(m, f)
    guess = d[prob]
    if verbose: print("%s -> %s (%.2f%%)" % (name, guess, prob * 100))
    return guess, prob


def classify(names, weight=weight_file):
    classifier = load_classifier(weight)
    for name in names:
        _classify(unicode(name, 'utf-8'), classifier)
    print("\nClassified %d names" % (len(names)))


'''
text = unicode("العدبى", "utf-8")
print(text)
-------------
import codecs
data = codecs.open("names.txt", "r", "utf-8")
names = data.read()
print(names)
-----------
import pandas as pd
df = pd.read_excel('arabic.xlsx', encoding='utf-8')
df.shape
df.iloc[945]
-------------
records = [tuple(x) for x in df.to_records(index=False)]
type(records[0])
--------------
import openpyxl
wb = openpyxl.load_workbook("Empty.xlsx")
ws = wb['sheet_name']
for row in ws.rows:
        for cell in row:
              print cell.value

for row in ws.rows:
        for cell in row:
              cell.value = new_value
----------------
alif = unicode("ىءهونملقكفغعظطضصشسرذدخحجثتبا", "utf-8")
print alif
'''
