import plotBoundary
from typing import Union, Optional
import dataclasses

import matplotlib.pyplot as plt
import scipy.optimize
import scipy.special
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg



@dataclasses.dataclass
class RegularizationParams:
    norm: float
    lambda_: float

class LogReg:
    def __init__(self, param_idx: Union[int, str],
            regularization: Optional[RegularizationParams] = None,
            feature_map: Optional[callable] = None
            ):
        self.param_idx = str(param_idx)
        self.reg = regularization
        self.feature_map = feature_map

        self.w = None  # np.ndarray of shape (n,) where n is 1+#(parameters in X)

        self._loadData()


    def _loadData(self):
        """Load and save data to train on"""
        # Load data
        train = np.loadtxt('data/data'+self.param_idx+'_train.csv')

        self.X_unmapped = train[:, 0:2].copy()
        self.Y = (train[:, 2:3].copy() + 1) / 2.
        self.Y = self.Y.flatten()

        # Apply feature map
        if self.feature_map:
            self.X = self.feature_map(self.X_unmapped)
        else:
            self.X = self.X_unmapped

    def train(self):
        """Train the classifier

        Using the training data in data/data+str(param_idx)+_train.csv"""

        # print('======Training======')
        # Initialize weights randomly to avoid cost discontinuities at 0.
        # Probably not necessary...
        data_shape = self.X.shape
        w = np.random.normal(size=(1+data_shape[1],)) 

        # Minimize and get solution
        sol = scipy.optimize.minimize(LogReg._cost, w,
                args=(self.X, self.Y, self.reg))
        self.w = sol.x

        return self.w, self._cost(self.w, self.X, self.Y, self.reg)
        
    def trainAcc(self):
        """Get the training accuracy"""
        return self._accuracy(self.w, self.X, self.Y)

    def testAcc(self):
        """Load test-data and get accuracy"""
        test = np.loadtxt('data/data'+self.param_idx+'_test.csv')
        X_unmapped = test[:, 0:2].copy()
        Y = (test[:, 2:3].copy() + 1) / 2.
        Y = Y.flatten()

        # Apply feature map
        if self.feature_map:
            X = self.feature_map(X_unmapped)
        else:
            X = X_unmapped

        return LogReg._accuracy(self.w, X, Y)

    def valAcc(self):
        """Load validation data and get accuracy"""
        # Load data
        test = np.loadtxt('data/data'+self.param_idx+'_validate.csv')
        X_unmapped = test[:, 0:2].copy()
        Y = (test[:, 2:3].copy() + 1) / 2.
        Y = Y.flatten()

        # Apply feature map
        if self.feature_map:
            X = self.feature_map(X_unmapped)
        else:
            X = X_unmapped

        return LogReg._accuracy(self.w, X, Y)

    @staticmethod
    def _accuracy(w, X, Y):
        """Get the accuracy of the predictor w on data X, Y"""
        pred_class = np.array(np.around(LogReg._predict(w, X)), dtype=int)
        return np.sum(pred_class == Y)/Y.size

    @staticmethod
    def _predict(w, X):
        """Predict the values of X using parameters w"""
        return scipy.special.expit(w[0] + w[1:] @ X.T)

    def predictLog(self, x):
        """Predict a value for a single datapoint x or an array X"""
        if len(x.shape) == 1 or x.shape[1] == 1:
            return LogReg._predict(self.w, x.reshape((1, -1))).flatten()
        else:
            return LogReg._predict(self.w, x)

    @staticmethod
    def _cost(w, X, Y, reg: Optional[RegularizationParams]):
        """Cost as a function of lin.reg. parameters w with datapoints X, labels Y"""
        if reg:
            reg_cost = reg.lambda_ * np.linalg.norm(w[1:], ord=reg.norm)
        else:
            reg_cost = 0
        pred = LogReg._predict(w, X)
        return -np.sum(Y * np.log(pred) + (1-Y) * np.log(1-pred)) + reg_cost

    def plotDecisionBoundary(self, values=[0.1, 0.5, 0.9], title="DecisionBoundary"):
        plotBoundary.plotDecisionBoundary(
                self.X, self.Y, self.predictLog, values=values, title=title
                )
        

def _main():
    np.random.seed(1)

    dataset = 2
    lreg = LogReg(dataset, regularization=RegularizationParams(2, 0.30))
    print(lreg.train())
    print(lreg.trainAcc())
    print(lreg.testAcc())
    print(lreg.valAcc())
    lreg.plotDecisionBoundary()
    plt.show()


if __name__ == "__main__":
    _main()


if False:
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

