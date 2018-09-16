import numpy as np
import utils


class DecisionStumpEquality:

    def __init__(self):
        pass


    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y,None,minlength=2)    
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 
        
        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        # Loop over features looking for the best split
        X = np.round(X)     
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]
                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] == value])
                y_not = utils.mode(y[X[:,d] != value])
                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] != value] = y_not
                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat





class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        """ YOUR CODE HERE """
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y,None,minlength=2)    
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 
        
        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)
        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]
                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] >= value])
                y_not = utils.mode(y[X[:,d] < value])
                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] < value] = y_not
                # Compute error
                errors = np.sum(y_pred != y)
                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

        #print(self.splitVariable)
        #print(self.splitValue)
        #print(self.splitSat)
        #print(self.splitNot)
        #print(y_pred)
        
    def predict(self, X):
        M, D = X.shape
        #X = np.round(X)
        #print(self.splitVariable)
        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] >= self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        #print(yhat)
        return yhat


"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true (you can show this with l'Hopital's rule), 
but numerically results in NaN because log(0) returns -Inf.
"""
def entropy(p):
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)
    
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
class DecisionStumpInfoGain:

    def __init__(self):
        pass

    def fit(self, X, y):
        """ YOUR CODE HERE """
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y,None,minlength=2)    
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 
        
        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        #(start) This part can be a function
        iniProb = np.bincount(y)/np.size(y)
        #print(iniProb)
        iniEntropy = entropy(iniProb)
        #(end)
        minInfoGain = iniEntropy
        maxInfoGain = 0
        #print(iniEntropy)
        minError = np.sum(y != y_mode)
        #print(y)
        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]
                # Find most likely class for each split
                y_spl0 = y[X[:,d] >= value]
                y_spl1 = y[X[:,d] < value]
                #print(y_spl1)
                if np.size(y_spl0)!=0:
                    spl0Prob = np.bincount(y_spl0,None,2)/np.size(y_spl0)
                elif np.size(y_spl0)==0:
                    spl0Prob = np.zeros(2)
                spl0Entropy = entropy(spl0Prob)
                wSpl0Entropy = (np.size(y_spl0)/np.size(y))*spl0Entropy                
                if np.size(y_spl1)!=0:
                    spl1Prob = np.bincount(y_spl1,None,2)/np.size(y_spl1)
                elif np.size(y_spl1)==0:
                    spl1Prob = np.zeros(2)
                spl1Entropy = entropy(spl1Prob)
                wSpl1Entropy = (np.size(y_spl1)/np.size(y))*spl1Entropy                
                infoGain = iniEntropy - wSpl0Entropy - wSpl1Entropy
                
                #print(infoGain)
                
                #print(testarray)
                #print(np.size(testarray))
                y_sat = utils.mode(y[X[:,d] > value])
                y_not = utils.mode(y[X[:,d] < value])
                #if(y_sat==0):print(y_not)
                #print(y_not)
                # Make predictions
                y_pred = y_sat * np.ones(N)
                #if(y_sat==1):print(y_pred)
                y_pred[X[:, d] < value] = y_not
                # Compute error
                errors = np.sum(y_pred != y)
                # Compare to minimum error so far
                if (infoGain!=0 and infoGain > maxInfoGain):
                    # This is the lowest error, store this value
                    minError = errors
                    maxInfoGain = infoGain
                    #print(minInfoGain)
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

        #print(minInfoGain)
        #print(self.splitVariable)
        #print(self.splitValue)
        #print(self.splitSat)
        #print(self.splitNot)
        #print(y_pred)
        
    def predict(self, X):
        M, D = X.shape
        #X = np.round(X)
        #print(self.splitVariable)
        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] >= self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        #print(yhat)
        return yhat
