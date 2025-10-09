#this file performs different types of regression
import numpy as np
from scipy import stats
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LinearRegression

def lin_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_

def huber_regression(X, y, epsilon=1.35):
    model = HuberRegressor(epsilon=epsilon)
    model.fit(X, y)
    return model.coef_

def quantile_regression(X, y, alpha =0):
    model = QuantileRegressor(quantile=alpha, solver='highs')
    model.fit(X, y)
    return model.coef_


