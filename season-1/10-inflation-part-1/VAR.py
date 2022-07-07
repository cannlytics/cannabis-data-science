"""
VAR Functions | Cannlytics

Author: Keegan Skeate
Contact: <keegan@cannlytics.com>
Created: Wed Apr 14 07:55:43 2021

License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Description:
    
    Crude VAR functions.

Resources:

"""


import numpy as np
import statsmodels.api as sm

def VAR(Vector, lag_order):
    """
    Inputs a Vector of dimension N x I, where N is the number of observations
    and I is the number of variables, as well as the lag order of the model.
    Returns the estimated equations from OLS as a dictionary with names 'Eq#'.
    """
    X = np.empty_like(Vector)
    for i in range(1 , 1 + lag_order):
        X = np.column_stack([X, lag(Vector, i)])
    X = np.delete(X, np.s_[0:len(Vector[0])], axis=1)
    X = sm.add_constant(X)   
    VAR_estimates = {}
    for i in range(1,len(Vector[0])+1):
        VAR_estimates["Eq{0}".format(i)] = sm.OLS(Vector[:,[i-1]][lag_order:],
                                                  X[lag_order:]).fit()       
    return VAR_estimates

    
def VAR_forecast(Vector,VAR_estimates,lag_order,horizon,shock=None):
    """
    Inputs the VAR Vector, VAR estimates, the lag order of the model,
    the forecast horizon, and the desired first period shock.
    """
    # Inital Period shock for IRF
    if shock is None:
        shock = np.zeros(len(Vector[0]))
    error = np.zeros((len(Vector),len(Vector[0])))
    error[0] = shock
    # Predictions for Forecast Horizon
    for t in np.arange(0,horizon):     
        X_hat = Vector
        for i in range(1 , lag_order):
            X_hat = np.column_stack([X_hat, lag(Vector, i)])
        X_hat = sm.add_constant(X_hat)     
        Y_hat = []
        for Eq in VAR_estimates:
            Y_hat.append(np.dot(X_hat[-1], VAR_estimates[Eq].params))
        Forecast = Y_hat + error[t]
        Vector = np.vstack((Vector,Forecast))
    return Vector[-horizon:]
    
def IRF(Vector,VAR_estimates,lag_order,horizon,shock):
    baseline = VAR_forecast(Vector,VAR_estimates,lag_order,horizon,shock=None)
    impact = VAR_forecast(Vector,VAR_estimates,lag_order,horizon,shock=shock.T)
    return (impact-baseline)

def lag(x,lag=None):
    if lag==None: lag=1
    lag_values = np.empty_like(x)
    for i in np.arange(0,len(x)):
        if i>=lag:
            lag_values[i] = x[i-lag]
    return lag_values
    
def cov_matrix(u):
    k = len(u[0])
    matrix = np.ones((k,k))
    for i in np.arange(0,k):
        for j in np.arange(0,k):
            matrix[i][j] = np.cov(u.T[i],u.T[j])[0][1]
    return matrix
