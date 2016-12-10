
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
    return(np.dot(mX.T, (logit(mX, vBeta) - vY)))

#====================================================================
# optimize to get the MLE using the L-BFGS optimizer (analytical derivatives)
#====================================================================

# with numerical gradient estimation:
x1,f1,d1 = sp.optimize.fmin_l_bfgs_b(logLikelihoodLogitStable,
                                  x0 = np.zeros(5),
                                    args = (mX, vY),approx_grad=True,
                                    pgtol =  1e-3, disp = True)

# with fprime:
x2, f2, d2 = sp.optimize.fmin_l_bfgs_b(logLikelihoodLogitStable,
                                       x0=np.zeros(5),
                                       args=(mX, vY),fprime=likelihoodScore ,
                                       pgtol=1e-3, disp=True)
print('x1:', x1)
print('x2:', x2)
print('diff:', x2-x1)
print('f1:', f1)
print('f2:', f2)
print('d1:', d1)
print('d2:', d2)
# [-11.19744679   0.40484533   0.09997348  -0.19242313   0.09171951]
# [-11.19743592   0.40484564   0.09997341  -0.19242314   0.09171917]