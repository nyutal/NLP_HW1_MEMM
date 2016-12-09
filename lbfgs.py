
# load required libraries
import numpy as np
import scipy as sp
import scipy.optimize
import pandas as pd

# hyperlink to data location
urlSheatherData = "http://www.stat.tamu.edu/~sheather/book/docs/datasets/MichelinNY.csv"

# read in the data to a NumPy array
arrSheatherData = np.asarray(pd.read_csv(urlSheatherData))
print(arrSheatherData)

# slice the data to get the dependent variable
vY = arrSheatherData[:, 0].astype('float64')

# slice the data to get the matrix of predictor variables
mX = np.asarray(arrSheatherData[:, 2:]).astype('float64')

# add an intercept to the predictor variables
intercept = np.ones(mX.shape[0]).reshape(mX.shape[0], 1)
mX = np.concatenate((intercept, mX), axis = 1)

# the number of variables and obserations
iK = mX.shape[1]
iN = mX.shape[0]

# logistic transformation
def logit(mX, vBeta):
    return((np.exp(np.dot(mX, vBeta))/(1.0 + np.exp(np.dot(mX, vBeta)))))

# stable parametrisation of the cost function
def logLikelihoodLogitStable(vBeta, mX, vY):
    return(-(np.sum(vY*(np.dot(mX, vBeta) -
    np.log((1.0 + np.exp(np.dot(mX, vBeta))))) +
                    (1-vY)*(-np.log((1.0 + np.exp(np.dot(mX, vBeta))))))))

# score function
def likelihoodScore(vBeta, mX, vY):
    return(np.dot(mX.T,
                  (logit(mX, vBeta) - vY)))

#====================================================================
# optimize to get the MLE using the L-BFGS optimizer (analytical derivatives)
#====================================================================
x,f,d = sp.optimize.fmin_l_bfgs_b(logLikelihoodLogitStable,
                                  x0 = np.array([0, 0, 0, 0, 0]),
                                    args = (mX, vY), fprime = likelihoodScore,
                                    pgtol =  1e-3, disp = True)

print(x) # print the results of the optimisation
print(f)
print(d)
# [-11.19744679   0.40484533   0.09997348  -0.19242313   0.09171951]