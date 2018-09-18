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
import bigO

# user-defined
import utils
import grads
from decision_stump import DecisionStumpEquality, DecisionStumpErrorRate, DecisionStumpInfoGain
from decision_tree import DecisionTree
from simple_decision import SimpleDecision

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "3.3":
        # Here is some code to test your answers to Q3.3
        # Below we test out example_grad using scipy.optimize.approx_fprime, which approximates gradients.
        # if you want, you can use this to test out your foo_grad and bar_grad

        def check_grad(fun, grad):
            x0 = np.random.rand(5) # take a random x-vector just for testing
            diff = approx_fprime(x0, fun, 1e-4)  # don't worry about the 1e-4 for now
            print("\n** %s **" % fun.__name__)
            print("My gradient     : %s" % grad(x0))
            print("Scipy's gradient: %s" % diff)

        #check_grad(grads.example, grads.example_grad)
        check_grad(grads.foo, grads.foo_grad)
        check_grad(grads.bar, grads.bar_grad)

    elif question == "4.3":
        N=2000
        bigO.func4(N)
        N=3000
        bigO.func4(N)
        N=4000
        bigO.func4(N)
        N=5000
        bigO.func4(N)
        N=6000
        bigO.func4(N)
    elif question == "5.1":
        # Load the fluTrends dataset
        df = pd.read_csv(os.path.join('..','data','fluTrends.csv'))
        X = df.values
        names = df.columns.values

        Xravel = X.ravel()
        
        Xmax = np.amax(X)
        Xmin = np.amin(X)
        Xmean = np.mean(X)
        Xmedi = np.median(X)
        Xmode = stats.mode(Xravel)
        Xmode2 = utils.mode(X)
        pctl05 = np.percentile(X,5)
        pctl25 = np.percentile(X,25)
        pctl50 = np.percentile(X,50)
        pctl75 = np.percentile(X,75)
        pctl95 = np.percentile(X,95)

        Xtr = np.transpose(X)
        
        regMean = np.zeros(shape=(1,np.size(X,1)))
        regVar = np.zeros(shape=(1,np.size(X,1)))
        for i in range(np.size(X,1)):
            regMean[0,i] = np.mean(Xtr[i,:])
            regVar[0,i] = np.var(Xtr[i,:])

            
        print(pctl05)
        print(pctl25)
        print(pctl50)
        print(pctl75)                
        print(pctl95)

        print(Xmode)
        print(Xmode2)

        print(names)
        print(regMean)
        print(f'Array Min:{Xmin:.3f}, Max:{Xmax:.3f}, Mean:{Xmean:.3f}, Median:{Xmedi:.3f}, Mode:{Xmode2:.3f}')

        print(f'Quantile 5%:{pctl05:.3f}, 25%:{pctl25:.3f}, 50%:{pctl50:.3f}, 75%:{pctl75:.3f}, 95%:{pctl95:.3f}')

        print(f'Region with highest mean: {names[np.argmax(regMean)]}')
        print(f'Region with lowest mean: {names[np.argmin(regMean)]}')
        print(f'Region with highest variance: {names[np.argmax(regVar)]}')
        print(f'Region with lowest variance: {names[np.argmin(regVar)]}')

        n, bins, patches = plt.hist(Xravel,bins=80,range=(0,4))
        plt.axis([0,4,0,50])
        plt.show()

        # YOUR CODE HERE

    elif question == "6":
        # 1Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 2Evaluate majority predictor model
        y_pred = np.zeros(y.size) + utils.mode(y)

        error = np.mean(y_pred != y)
        print("Mode predictor error: %.3f" % error)

        # 3Evaluate decision stump
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error (ErrorRate): %.3f"
              % error)

        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q6_decisionBoundary_ErrorRate.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        # 3Evaluate decision stump
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error (InfoGain): %.3f"
              % error)

        
        # Plot result
        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "q6_decisionBoundary_InfoGain.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    elif question == "6.2":
        # Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # Evaluate decision stump
        model = DecisionStumpErrorRate()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # Plot result
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_2_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "6.3":
        # 1. Load citiesSmall dataset
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        # 3. Evaluate decision stump
        model = DecisionStumpInfoGain()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with info gain rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_3_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
    
    elif question == "6.4":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        model = DecisionTree(max_depth=2,stump_class=DecisionStumpInfoGain)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_4_decisionBoundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "6.4h":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset["X"]
        y = dataset["y"]

        model = SimpleDecision()

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
        
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q6_4_decisionBoundary_hc.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
    elif question == "6.5":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)
        
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        p1 = plt.figure(0) #added for figure
        
        
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

        plt.figure(1,figsize=(10,7))
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
        
        plt.show()


    else:
        print("No code to run for question", question)
