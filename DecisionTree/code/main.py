# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np                             
import pandas as pd                            
import matplotlib.pyplot as plt                
from scipy.optimize import approx_fprime       
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

# user-defined
import utils
from decision_stump import DecisionStumpEquality, DecisionStumpErrorRate, DecisionStumpInfoGain
from decision_tree import DecisionTree
from simple_decision import SimpleDecision

with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
    dataset = pickle.load(f)
    
    X = dataset["X"]
    y = dataset["y"]
    print("n = %d" % X.shape[0])
    
    depths = np.arange(1,15) # depths to try
    
    #p1 = plt.figure(0) #added for figure
    plt.figure(1,figsize=(12,8))
    plt.subplot(224)
    
    t = time.time()
    my_tree_errors = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        model = DecisionTree(max_depth=max_depth)
        model.fit(X, y)
        y_pred = model.predict(X)
        my_tree_errors[i] = np.mean(y_pred != y)
    model_ER = model
    print("Our decision tree with DecisionStumpErrorRate took %f seconds" % (time.time()-t))
    
    plt.plot(depths, my_tree_errors, label="errorrate")
    
    
    t = time.time()
    my_tree_errors_infogain = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        model = DecisionTree(max_depth=max_depth,stump_class=DecisionStumpInfoGain)
        model.fit(X, y)
        y_pred = model.predict(X)
        my_tree_errors_infogain[i] = np.mean(y_pred != y)
    print("Our decision tree with DecisionStumpInfoGain took %f seconds" % (time.time()-t))
    model_IG = model
    plt.plot(depths, my_tree_errors_infogain, label="infogain")
    
    t = time.time()
    sklearn_tree_errors = np.zeros(depths.size)
    for i, max_depth in enumerate(depths):
        model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
        model.fit(X, y)
        y_pred = model.predict(X)
        sklearn_tree_errors[i] = np.mean(y_pred != y)

        #if max_depth == 14:
        #    print("sklearn error: %.8f" % sklearn_tree_errors[i])
        #    
        #    fname1 = os.path.join("..", "figs", "q6_5_sklearn.pdf")
        #    plt.savefig(fname1)
        #    print("\nFigure saved as '%s'" % fname1)
    model_SKL = model
    print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

    plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)
    plt.xlabel("Depth of tree")
    plt.ylabel("Classification error")
    plt.legend()
    
    plt.subplot(221)
    utils.plotClassifier(model_ER, X, y)
    plt.title('Method: Error Rate')
    plt.xlabel('')
    
    plt.subplot(222)
    utils.plotClassifier(model_IG, X, y)
    plt.title('Method: Information Gain')
    #frame1.axes.get_xaxis().set_visible(False)
    
    
    plt.subplot(223)
    utils.plotClassifier(model_SKL, X, y)
    plt.title('Method: sklearn Decision Tree')
    
    fname = os.path.join("..", "figs", "DecisionTree_summary.pdf")
    plt.savefig(fname)

    fname = os.path.join("..", "figs", "DecisionTree_summary.png")
    plt.savefig(fname)
    
    plt.show()
