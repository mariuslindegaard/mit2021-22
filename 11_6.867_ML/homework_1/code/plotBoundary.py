from numpy import *
# import pylab as pl
import matplotlib.pyplot as plt
pl = plt

# X is data matrix (each row is a data point)
# Y is desired output (1 or -1)
# scoreFn is a function of a data point
# values is a list of values to plot

def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    # Plot the decision boundary. For that, we asign a score to
    # each point in the mesh [x_min, m_max] x [y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)
    xx, yy = meshgrid(arange(x_min, x_max, h),
                      arange(y_min, y_max, h))
    zz = array([scoreFn(x) for x in c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    # pl.figure()  # Do not make a new figure, we will plot on whatever is active
    CS = pl.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
    pl.clabel(CS, fontsize=9, inline=1)
    # Plot the training points
    # pl.scatter(X[:, 0], X[:, 1], c=(1.-Y).ravel(), s=50, cmap = pl.cm.cool)
    # import pdb; pdb.set_trace()
    X_pos = X[(Y > 1/2).ravel(), :]
    X_neg = X[(Y < 1/2).ravel(), :]
    pl.scatter(X_pos[:, 0], X_pos[:, 1], s=50, label="Class 1")
    pl.scatter(X_neg[:, 0], X_neg[:, 1], s=50, label="Class 0")
    pl.title(title)
    plt.legend()
    pl.axis('tight')
