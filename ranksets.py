import numpy as np
import scipy.stats as stats

def find_ranksets(k, thetahat, Sigma, alpha):
    """
    Computes the rank-set for each model with coverage probability 1-alpha

    Parameters
    ----------
    k : int
        number of models
    thetahat : numpy.ndarray
        prediction-powered estimate of win probability of each model
    Sigma : numpy.ndarray
        sample covariance of thetahat
    alpha : float
        error probability

    Returns
    -------
    list
        Rank-set [lower rank, upper rank] for each model.

    """
    lhat = np.ones((k,1))
    uhat = np.ones((k,1))*k
    
    for m1 in range(k):
        for m2 in range(k):
            if m1 == m2:
                continue
            d = abs(thetahat[m1] - thetahat[m2])/np.sqrt(2) - np.sqrt((Sigma[m1][m1] + Sigma[m2][m2] - 2*Sigma[m1][m2])*(stats.chi2.ppf(1-alpha,k)/(2)))
            if d > 0 and thetahat[m1] < thetahat[m2]:
                lhat[m1] += 1
            elif d > 0 and thetahat[m1] > thetahat[m2]:
                uhat[m1] -= 1
    
    return [[int(lhat[m][0]), int(uhat[m][0])] for m in range(k)]