import os
import pickle
import gzip
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from neural_net import NeuralNet
from manifold import MDS, ISOMAP
import utils

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', required=True)

    io_args = parser.parse_args()
    task = io_args.question

    elif task == "2":

        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        print(X.shape)
        
        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        hidden_layer_sizes = [50]
        model = NeuralNet(hidden_layer_sizes, sgd=0)

        t = time.time()
        model.fit(X,Y)
        print("Fitting took %d seconds" % (time.time()-t))

        # Comput training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)
        
        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    elif task == "2.1":
        W = np.array([[-2,2,-1],[1,-2,0]])
        x = np.array([-3,2,2])
        v = np.array([[3],[1]])
        
        z = x@W.T
        hz = 1/(1+np.exp(-1*z))

        y = hz@v
        
        print("z_i = ", z)
        print("h(z_i) = ", hz)
        print("y = ",y[0])

        #print(W-1)
        
    elif task == "2.4":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        print("n =", X.shape[0])
        print("d =", X.shape[1])        

        model = MLPClassifier()
        #Layer=50,80 no good
        #layer=150, alpha=0 pretty good
        #sgd no good with the above condition
        model.fit(X,y)

        #note for condition
        #print("no modification")
        #print("hidden_layer_sizes=(1000, )")
        #print("activation='logistic'")
        #print("hidden_layer_sizes=(200, ),early_stopping=True")
        #print("hidden_layer_sizes=(500, )")
        
        # Compute training error
        yhat = model.predict(X)
        trainError = np.mean(yhat != y)
        print("Training error = ", trainError)
        
        # Compute test error
        yhat = model.predict(Xtest)
        testError = np.mean(yhat != ytest)
        print("Test error     = ", testError)

    else:
        print("Unknown task: %s" % task)    
