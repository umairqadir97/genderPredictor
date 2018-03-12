# -*- coding: utf-8 -*-

import time
import nltk
import pickle
from nltk.classify import apply_features
import pandas as pd

TRAINED_MODEL = 'gender_features_naive'


# LOAD already trained model
def load_classifier(weight=TRAINED_MODEL):
    with open(weight, 'rb') as f:
        classifier = pickle.load(f)
        f.close()
    print('Loaded weights from "%s"...\n' % weight)
    return classifier


def features(name):
    alif = unicode("ىءهونملقكفغعظطضصشسرذدخحجثتبا", "utf-8")
    all_letters = 'abcdefghijklmnopqrstuvwxyz' + alif
    features_list = {}
    if name:
        features_list["last_letter"] = name[-1]
        features_list["first_letter"] = name[0]
        for letter in all_letters:
            features_list["count(%s)" % letter] = name.count(letter)
            features_list["has(%s)" % letter] = (letter in name)
        features_list["suffix2"] = name[-2:]
        features_list["suffix3"] = name[-3:]
        features_list["suffix4"] = name[-4:]
    return features_list


# Classify names
def classify(name, classifier):
    # names should be an array
    dist = classifier.prob_classify(features(name))
    m, f = dist.prob("male"), dist.prob("female")
    d = {m: "male", f: "female"}
    guess = d[max(m, f)]
    prob = max(m,f)
    print("%s -> %s (%.2f%%)" % (name, guess, prob*100))


def run(weight=TRAINED_MODEL):
    classifier = load_classifier(weight)
    while(True):
        name = raw_input("Enter Any Name to Classify:    ")
        if type(name)!=unicode:
            name = unicode(name.lower(), 'utf-8')
        classify(name, classifier)
        next = raw_input("enter 'y' to test another name, any other key to exit:    ")
        if next != 'y':
            break

if __name__ == '__main__':
    run()
