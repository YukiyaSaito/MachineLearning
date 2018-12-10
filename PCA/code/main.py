import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split

import utils
import logReg
from logReg import logRegL2, kernelLogRegL2
from pca import PCA, AlternativePCA, RobustPCA

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--task', required=True)

    io_args = parser.parse_args()
    task = io_args.task

    if task == '1': 
        X = load_dataset('highway.pkl')['X'].astype(float)/255
        #print(X[0].reshape(64,64).T)
        n,d = X.shape
        print(n,d)
        h,w = 64,64      # height and width of each image

        k = 5            # number of PCs
        threshold = 0.1  # threshold for being considered "foreground"
        #threshold = 0.07
        
        model = AlternativePCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_pca = model.expand(Z)

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat_robust = model.expand(Z)

        fig, ax = plt.subplots(2,3)
        for i in range(10):
            ax[0,0].set_title('$X$')
            ax[0,0].imshow(X[i].reshape(h,w).T, cmap='gray')

            ax[0,1].set_title('$\hat{X}$ (L2)')
            ax[0,1].imshow(Xhat_pca[i].reshape(h,w).T, cmap='gray')
            
            ax[0,2].set_title('$|x_i-\hat{x_i}|$>threshold (L2)')
            ax[0,2].imshow((np.abs(X[i] - Xhat_pca[i])<threshold).reshape(h,w).T, cmap='gray')

            ax[1,0].set_title('$X$')
            ax[1,0].imshow(X[i].reshape(h,w).T, cmap='gray')
            
            ax[1,1].set_title('$\hat{X}$ (L1)')
            ax[1,1].imshow(Xhat_robust[i].reshape(h,w).T, cmap='gray')

            ax[1,2].set_title('$|x_i-\hat{x_i}|$>threshold (L1)')
            ax[1,2].imshow((np.abs(X[i] - Xhat_robust[i])<threshold).reshape(h,w).T, cmap='gray')

            utils.savefig('highway_{:03d}.png'.format(i))

    else:
        print("Unknown task: %s" % task)    
