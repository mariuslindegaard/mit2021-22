from plotBoundary import *
import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np

# import your logistic regression training code

# parameters
name = '3'

print('======Training======')
# load data from csv files
train = np.loadtxt('data/data'+name+'_train.csv')


# use deep copy to be safe
X = train[:, 0:2].copy()
Y = (train[:, 2:3].copy() + 1) / 2.

# Carry out training, 
### TODO ###

# Compute training accuracy
### TODO ###

# Define the predictLog(x) function, which uses the trained parameters
# predictLog(x) should return the score that the classifier assigns to point x
#   e.g. for linear classification, this means sigmoid(w^T x + w0)
### TODO ###

# Compute algorithmic stability
### TODO ###

# plot training results
plotDecisionBoundary(X, Y, predictLog, [0, 0.5, 1], title = 'LR Train')
pl.show()

print('======Validation======')
# load data from csv files
validate = np.loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLog, [0, 0.5, 1], title = 'LR Validate')
pl.show()


print('======Testing======')
# load data from csv files
test = np.loadtxt('data/data'+name+'_test.csv')
X = test[:, 0:2]
Y = (test[:, 2:3] + 1) / 2

# Compute testing accuracy of the predictions of your model
### TODO ###
