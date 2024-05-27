import numpy as np

def sample_avg(k, Dn,key='llm'):
    """
    Calculates the sample average win probability for each model 

    Parameters
    ----------
    k : int
        number of models
    Dn : dictionary
        Same structure as returned by data_process.summarized()
        Keys: 'winner1_predicted' or 'winner1_human', numpy.ndarray indicating 
               whether model_a won each pairwise comparison
               (1:model_a won, 0:model_b won or tie)
              'winner2_predicted' or 'winner2_human', numpy.ndarray indicating 
               whether model_b won each pairwise comparison
               (0:model_a won or tie, 1:model_b won)
              'model_a_matrix', matrix where each column is a one-hot vector 
               corresponding to each sample indicating which model was model_a
              'model_b_matrix', matrix where each column is a one-hot vector 
               corresponding to each sample indicating which model was model_b
    key : string, optional
    Default is 'llm'. If 'human', calculate sample average from human judgments
    otherwise calculate from LLM judgments

    Returns
    -------
    a : list of floats
        sample average win probability of each model from dataset Dn
    """
    
    n = np.shape(Dn['model_a_matrix'])[1]
    M1, M2 = Dn['model_a_matrix'], Dn['model_b_matrix']
    w1, w2 = Dn['winner1_predicted'], Dn['winner2_predicted']
    if key=='human':
        w1, w2 = Dn['winner1_human'], Dn['winner2_human']
    
    a = np.matmul(np.ones((k,1)), np.transpose(np.matmul(
        M1+M2,np.ones((n,1))))) *np.identity(k)
    for i in range(k):
        a[i][i] = 1/a[i][i]
    a = np.matmul(a,np.matmul(M1,w1) + np.matmul(M2,w2))
    
    A = np.matmul(np.ones((k,1)), np.transpose(
        w1 - np.matmul(np.transpose(M1) ,a)))*M1
    A += np.matmul(np.ones((k,1)), 
                   np.transpose(w2 - np.matmul(np.transpose(M2), a)))*M2
    
    Sigma = (np.matmul(A, np.transpose(A)))/(n**2)
    
    return a, Sigma

def lhat_opt(k,Dn,DN):
    """
    Finds the optimal lambda value that minimises the variance of thetahat.

    Parameters
    ----------
    k : int
        Number of models
    Dn : dictionary
        Same structure as returned by data_process.summarized()
        Keys: 'winner1_predicted', 'winner2_predicted', 'winner1_human', 
              'winner2_human, 'model_a_matrix', 'model_b_matrix'
    DN : dictionary
        Same structure as returned by data_process.summarized(), 
        Keys: 'winner1_predicted', 'winner2_predicted','model_a_matrix', 
              'model_b_matrix'

    Returns
    -------
    lhat_opt : float
        Between 0 and 1, optimal weight of LLM judgments
    """
    n = np.shape(Dn['model_a_matrix'])[1]
    N = np.shape(DN['model_a_matrix'])[1]
    covN=sample_avg(k,DN)[1]*N
    
    a1=sample_avg(k,Dn)[0]
    a2=sample_avg(k,Dn,key='human')[0]
    
    M1, M2 = Dn['model_a_matrix'], Dn['model_b_matrix']
    assert 'winner1_human' in Dn, 'dataset Dn must include "winner1_human"'
    assert 'winner2_human' in Dn, 'dataset Dn must include "winner2_human"'
    w1, wf1 = Dn['winner1_human'], Dn['winner1_predicted']
    w2, wf2 = Dn['winner2_human'], Dn['winner2_predicted']
    
    A1 = np.matmul(np.ones((k,1)), np.transpose(
        wf1 - np.matmul(np.transpose(M1) ,a1)))*M1
    A1 += np.matmul(np.ones((k,1)), np.transpose(
        wf2 - np.matmul(np.transpose(M2), a1)))*M2
    A2 = np.matmul(np.ones((k,1)), np.transpose(
        w1 - np.matmul(np.transpose(M1) ,a2)))*M1
    A2 += np.matmul(np.ones((k,1)), np.transpose(
        w2 - np.matmul(np.transpose(M2), a2)))*M2
    
    covn=(np.matmul(A2, np.transpose(A1)))/(n)
    lhat_opt=np.trace(covn)/((1+n/N)*np.trace(covN))
    
    lhat_opt=max(lhat_opt,0)
    lhat_opt=min(lhat_opt,1)
    return lhat_opt


def estimate(k,Dn,DN,lhat=1):
    """
    Calculates prediction-powered estimate and sample covariance 
    of win probability for each model

    Parameters
    ----------
    k : int
        number of models
    Dn : dictionary
        Same structure as returned by data_process.summarized()
        Keys: 'winner1_predicted', 'winner2_predicted', 'winner1_human', 
              'winner2_human, 'model_a_matrix', 'model_b_matrix'
    DN : dictionary
        Same structure as returned by data_process.summarized(), 
        Keys: 'winner1_predicted', 'winner2_predicted','model_a_matrix', 
              'model_b_matrix'
    
    lhat : float, optional
        Default is 1. Must be between 0 and 1.
        Weight of LLM comparisons relative to human.

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
    wf1N, wf2N = DN['winner1_predicted'], DN['winner2_predicted']
    
    assert lhat<=1 and lhat>=0, "lhat must be between 0 and 1"
    
    a = np.matmul(np.ones((k,1)), np.transpose(
        np.matmul(M1N+M2N, np.ones((N,1)))))*np.identity(k)
    for i in range(k):
        assert a[i][i]>0,\
        'There must be at least one comparison for each model in dataset Dn'
        a[i][i] = 1/a[i][i]
    a = np.matmul(a, np.matmul(M1N,wf1N) + np.matmul(M2N, wf2N))
    
    M1n, M2n = Dn['model_a_matrix'], Dn['model_b_matrix']
    assert 'winner1_human' in Dn, 'dataset Dn must include "winner1_human"'
    assert 'winner2_human' in Dn, 'dataset Dn must include "winner2_human"'
    w1n, wf1n = Dn['winner1_human'], Dn['winner1_predicted']
    w2n, wf2n = Dn['winner2_human'], Dn['winner2_predicted']
    
    b = np.matmul(np.ones((k,1)), np.transpose(
        np.matmul(M1n+M2n, np.ones((n,1)))))*np.identity(k)
    for i in range(k):
        assert b[i][i]>0, \
        'There must be at least one comparison for each model in dataset DN'
        b[i][i] = 1/b[i][i]
    b = np.matmul(b, np.matmul(
        M1n, lhat*wf1n-w1n) + np.matmul(M2n, lhat*wf2n - w2n ))
    
    A = np.matmul(np.ones((k,1)), np.transpose(
        wf1N - np.matmul(np.transpose(M1N) ,a)))*M1N
    A += np.matmul(np.ones((k,1)), np.transpose(
        wf2N - np.matmul(np.transpose(M2N), a)))*M2N
    
    B = np.matmul(np.ones((k,1)), np.transpose(
        lhat*wf1n-w1n - np.matmul(np.transpose(M1n), b)))*M1n
    B += np.matmul(np.ones((k,1)), np.transpose(
        lhat*wf2n - w2n - np.matmul(np.transpose(M2n), b)))*M2n
    
    thetahat = lhat*a - b
    Sigma = (lhat**2)*(np.matmul(A, np.transpose(A)))/(N**2) + (np.matmul(B, np.transpose(B)))/(n**2)
    return thetahat, Sigma
