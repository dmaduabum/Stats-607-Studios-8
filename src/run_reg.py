#this file performs different types of regression and returns the estimated coefficients
import numpy as np
from scipy import stats
import warnings
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LinearRegression

def lin_regression(X, y):
    """Linear regression using sklearn"""

    if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        raise TypeError("Both X and y must be arrays or lists")
    # check if X is full rank
    if np.linalg.matrix_rank(X) < min(X.shape):
        warnings.warn("X is not full rank. Results may be unreliable.", UserWarning)

    model = LinearRegression()
    model.fit(X, y)
    return model.coef_

def huber_regression(X, y, epsilon=1.35):
    """Huber regression using sklearn"""

    if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        raise TypeError("Both X and y must be arrays or lists")
    
    model = HuberRegressor(epsilon=epsilon, max_iter=100000)
    model.fit(X, y)
    return model.coef_

def quantile_regression(X, y, alpha =0.5):
    """Quantile regression using sklearn"""

    if not isinstance(X, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        raise TypeError("Both X and y must be arrays or lists")
    
    model = QuantileRegressor(quantile=alpha, solver='highs')
    model.fit(X, y)
    return model.coef_


