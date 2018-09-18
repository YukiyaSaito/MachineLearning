#This is a hard-coded version of a depth-2 decision tree optimized by information gain (DecisionStumpInfoGain class in decision_stump.py).

import numpy as np

class SimpleDecision:

    def __init__(self):
        pass

    def predict(self,X):
        M, D = X.shape
        y = np.zeros(M)

        splitVar1 = 0
        splitVal1 = -80.248086
        splitSat1 = 0
        splitVar2 = 1
        splitVal2 = 36.813883
        splitSat2 = 0
        splitVar3 = 1
        splitVal3 = 37.695206
        splitSat3 = 0

        splitIndex1 = np.zeros(M, dtype=bool)
        splitIndex0 = np.zeros(M, dtype=bool)
        for i0 in range(M):
            if X[i0, splitVar1] > splitVal1:
                splitIndex1[i0] = True
            elif X[i0, splitVar1] <= splitVal1:
                splitIndex0[i0] = True

        splitIndex1_1 = np.zeros(M, dtype=bool)
        splitIndex1_0 = np.zeros(M, dtype=bool)
        splitIndex0_1 = np.zeros(M, dtype=bool)
        splitIndex0_0 = np.zeros(M, dtype=bool)
        for i1 in range(M):
            if splitIndex1[i1] == True:
                if X[i1, splitVar2] > splitVal2:
                    splitIndex1_1[i1] = True
                elif X[i1, splitVar2] <= splitVal2:
                    splitIndex1_0[i1] = True
            if splitIndex0[i1] == True:
                if X[i1, splitVar3] > splitVal3:
                    splitIndex0_1[i1] = True
                elif X[i1, splitVar3] <= splitVal3:
                    splitIndex0_0[i1] = True
                    
        y[splitIndex1_0] = 1
        y[splitIndex0_0] = 1
        #print(y[splitIndex0_0])
        return y
