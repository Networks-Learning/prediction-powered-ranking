import numpy as np


def sample_avg(k, Dn):
    """
    Calculates the sample average win probability for each model 

    Parameters
    ----------
    k : int
        number of models
    Dn : dictionary
        Same structure as returned by data_process.summarized()
        Keys: 'winner_predicted', numpy.ndarray indicating the winner of each pairwise comparison (1:model_a won, 0:model_b won, 0.5:tie)
              'model_a_matrix', matrix where each column is a one-hot vector corresponding to each sample indicating which model was model_a
              'model_b_matrix', matrix where each column is a one-hot vector corresponding to each sample indicating which model was model_b

    Returns
    -------
    a : list of floats
        sample average win probability of each model from the samples in dataset Dn
    """
    
    n = np.shape(Dn['model_a_matrix'])[1]
    M1, M2 = Dn['model_a_matrix'], Dn['model_b_matrix']
    w = Dn['winner_predicted']
    
    a = np.matmul(np.ones((k,1)), np.transpose(np.matmul(M1+M2,np.ones((n,1)))))*np.identity(k)
    for i in range(k):
        a[i][i] = 1/a[i][i]
    a = np.matmul(a,np.matmul(M1,w) + np.matmul(M2,np.ones((n,1)) - w))
    
    return a

def estimate(k,Dn,DN):
    """
    Calculates prediction-powered estimate and sample covariance of win probability for each model

    Parameters
    ----------
    k : int
        number of models
    Dn : dictionary
        Summarized dataset of human and llm pairwise comparisons
        Same structure as returned by data_process.summarized() including human winner
        Keys: 'winner_predicted', numpy.ndarray indicating the winner of each pairwise comparison (1:model_a won, 0:model_b won, 0.5:tie)
              'model_a_matrix', numpy.ndarray where each column is a one-hot vector corresponding to each sample indicating which model was model_a
              'model_b_matrix', numpy.ndarray where each column is a one-hot vector corresponding to each sample indicating which model was model_b
              'winner_human', numpy.ndarray indicating the winner of each pairwise comparison in human_dataset (1:model_a won, 0:model_b won, 0.5:tie)
    DN : dictionary
    Summarized dataset of llm pairwise comparisons
        Same structure as returned by data_process.summarized()
        Keys: 'winner_predicted', numpy.ndarray indicating the winner of each pairwise comparison (1:model_a won, 0:model_b won, 0.5:tie)
              'model_a_matrix', numpy.ndarray where each column is a one-hot vector corresponding to each sample indicating which model was model_a
              'model_b_matrix', numpy.ndarray where each column is a one-hot vector corresponding to each sample indicating which model was model_b
        DESCRIPTION.

    Returns
    -------
    thetahat : numpy.ndarray
        prediction-powered estimate of win probability of each model from samples in datasets Dn and DN
    Sigma : numpy.ndarray, size k times k
        sample covariance of thetahat
    """
    
    n = np.shape(Dn['model_a_matrix'])[1]
    N = np.shape(DN['model_a_matrix'])[1]
    M1N, M2N = DN['model_a_matrix'], DN['model_b_matrix']
    wfN = DN['winner_predicted']
    
    a = np.matmul(np.ones((k,1)), np.transpose(np.matmul(M1N+M2N, np.ones((N,1)))))*np.identity(k)
    for i in range(k):
        assert a[i][i]>0,'There must be at least one comparison for each model in dataset Dn (human data)'
        a[i][i] = 1/a[i][i]
    a = np.matmul(a, np.matmul(M1N,wfN) + np.matmul(M2N, np.ones((N,1))-wfN))
    
    M1n, M2n = Dn['model_a_matrix'], Dn['model_b_matrix']
    assert 'winner_human' in Dn, 'dataset Dn must include "winner_human"'
    wn, wfn = Dn['winner_human'], Dn['winner_predicted']
    
    b = np.matmul(np.ones((k,1)), np.transpose(np.matmul(M1n+M2n, np.ones((n,1)))))*np.identity(k)
    for i in range(k):
        assert b[i][i]>0, 'There must be at least one comparison for each model in dataset DN (llm data)'
        b[i][i] = 1/b[i][i]
    b = np.matmul(b,np.matmul(M1n, wfn-wn) + np.matmul(M2n, wn-wfn))
    
    A = np.matmul(np.ones((k,1)), np.transpose(wfN - np.matmul(np.transpose(M1N) ,a)))*M1N
    A += np.matmul(np.ones((k,1)), np.transpose(np.ones((N,1)) - wfN - np.matmul(np.transpose(M2N), a)))*M2N
    
    B = np.matmul(np.ones((k,1)), np.transpose(wfn-wn - np.matmul(np.transpose(M1n), b)))*M1n
    B += np.matmul(np.ones((k,1)), np.transpose(wn-wfn - np.matmul(np.transpose(M2n), b)))*M2n
    
    thetahat = a - b
    Sigma = (np.matmul(A, np.transpose(A)))/(N**2) + (np.matmul(B, np.transpose(B)))/(n**2)
    return thetahat, Sigma
