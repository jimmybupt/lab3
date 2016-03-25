
# -*- coding: utf-8 -*-
"""
@author: Kun Liu， Zhe Dong
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import time


#build classifer and fit training data
def KNN_classifier(training_data, class_labels):
    print ""
    print "Classification using KNN"
    start_time = time.time()

    neigh = KNeighborsClassifier()
    neigh.fit(training_data, class_labels)
    
    return neigh, (time.time() - start_time)
    

def tree_classifier(training_data, class_labels):
    print ""
    print "Classification using decision tree"
    start_time = time.time()
    
    clf = DecisionTreeClassifier()
    clf.fit(training_data, class_labels)
    
    return clf, (time.time() - start_time)
    
    
def bayes_classifier(training_data, class_labels):
    print ""
    print "Classification using Gaussian Naive Bayes"
    start_time = time.time()
    
    gnb = GaussianNB()
    gnb.fit(training_data, class_labels)
    
    return gnb, (time.time() - start_time)
