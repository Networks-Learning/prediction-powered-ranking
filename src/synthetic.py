import json
import sys

import numpy as np
from src import estimate as estimate
from src import ranksets as ranksets
import argparse
import os



def generate_data(k, n, theta, thetatilde):
    n = n // k * k
    winner1_human, winner2_human = np.zeros(shape=(n, 1)), np.zeros(shape=(n, 1))
    winner1_llm, winner2_llm = [np.zeros(shape=(n, 1)) for _ in range(len(thetatilde))], [np.zeros(shape=(n, 1)) for _
                                                                                          in range(len(thetatilde))]
    model_a_matrix = np.zeros(shape=(k, n))
    model_b_matrix = np.zeros(shape=(k, n))
    for i in range(n // k * k):
        m1 = i % k
        model_a_matrix[m1][i] = 1

        m2 = (i // k) % k
        model_b_matrix[m2][i] = 1

        match = np.random.rand(1)[0]

        if match < 2 * theta[m1]:
            winner1_human[i] = 1
        for j in range(len(thetatilde)):
            if match < 2 * thetatilde[j][m1]:
                winner1_llm[j][i] = 1

    summarized = [{'winner1_predicted': winner1_llm[j],
                   'winner2_predicted': winner2_llm[j],
                   'winner1_human': winner1_human,
                   'winner2_human': winner2_human,
                   'model_a_matrix': model_a_matrix,
                   'model_b_matrix': model_b_matrix} for j in range(len(thetatilde))]
    return summarized

def run_synthetic_one(k, n, N, noises, alpha, seed):
    np.random.seed(seed)
    theta=np.random.uniform(low=0.2, high=0.8, size=(k,))
    theta=sorted(theta, reverse=True)
    thetatildes=[theta+np.random.uniform(low=-noises[i], high=noises[i], size=(k,)) for i in range(len(noises))]
    thetatildes=[[max(0.01,min(0.99,thetatildes[j][i])) for i in range(k)] for j in range(len(noises))]
    theta=[theta[i]/sum(theta) for i in range(k)]
    thetatildes=[[thetatildes[j][i]/sum(thetatildes[j]) for i in range(k)] for j in range(len(noises))]

    Dn=generate_data(k,n,theta,thetatildes)
    DN=generate_data(k,N-n,theta,thetatildes)

    DnN=[{'winner1_human': np.concatenate((Dn[0]['winner1_human'],DN[0]['winner1_human'])),
        'winner2_human': np.concatenate((Dn[0]['winner2_human'],DN[0]['winner2_human'])),
        'winner1_predicted': np.concatenate((Dn[i]['winner1_predicted'],DN[i]['winner1_predicted'])),
        'winner2_predicted': np.concatenate((Dn[i]['winner2_predicted'],DN[i]['winner2_predicted'])),
        'model_a_matrix': np.concatenate((Dn[0]['model_a_matrix'],DN[0]['model_a_matrix']),axis=1),
        'model_b_matrix': np.concatenate((Dn[0]['model_b_matrix'],DN[0]['model_b_matrix']),axis=1)
    } for i in range(len(noises))]

    results=dict()
    thetas_results=dict()
    thetas_results['baseline']=theta
    for j in range(len(noises)):
        thetas_results['thetatilde '+str(noises[j])]=thetatildes[j]
        lhat_opt=estimate.lhat_opt(k,Dn[j],DN[j])
        thetahat,sigma=estimate.estimate(k, Dn[j], DN[j],lhat=lhat_opt)
        thetas_results['ppr '+str(noises[j])]=np.ndarray.flatten(thetahat).tolist()
        rank_sets=ranksets.find_ranksets(k,thetahat,sigma,alpha=alpha)
        results['ppr '+str(noises[j])]={i+1:rank_sets[i] for i in range(k)}

        thetahat,sigma = estimate.sample_avg(k, DnN[j],'llm')
        thetas_results['llm ' + str(noises[j])] = np.ndarray.flatten(thetahat).tolist()
        rank_sets = ranksets.find_ranksets(k, thetahat, sigma, alpha=alpha)
        results['llm ' + str(noises[j])] = {i + 1: rank_sets[i] for i in range(k)}

    #

    thetahat,sigma=estimate.sample_avg(k, Dn[0], 'human')
    rank_sets=ranksets.find_ranksets(k,thetahat,sigma,alpha=alpha)
    thetas_results['human only'] = np.ndarray.flatten(thetahat).tolist()
    results['human only']={i+1:rank_sets[i] for i in range(k)}
    results['baseline']={i+1:[i+1] for i in range(k)}

    return results, thetas_results
    


def main(iterations=300,
        k=8,
        n=[400,1000,5000,10000,20000],
        N=50000,
        alpha=0.05,
        noises=[0.05, 0.1, 0.3],
        seed=12345678,
        output_dir='outputs/synthetic/') -> 0:
    
    if type(n)==str:
        n=n[1:-1]
        n=[int(x) for x in n.split(',')]
    if type(noises)==str:
        noises=noises[1:-1]
        noises=[float(x) for x in noises.split(',')]

    np.random.seed(seed)
    seeds=np.random.randint(low=0,high=10000000,size=iterations*len(n))

    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for file in files:
            os.remove(os.path.join(output_dir, file))
    else:
        os.makedirs(output_dir)

    for i in range(iterations):
        if i%20==0:
            print(i)
        for nn in n:
            results, thetas = run_synthetic_one(k,nn,N,noises,alpha,seeds[i])
            results['iteration']=i
            thetas['iteration']=i
            with open(output_dir+'/k'+str(k)+'_'+str(nn)+'_'+str(alpha).split('.')[1]+'_ranksets.json', 'a') as f:
                json.dump(results, f)
                f.write('\n')
            with open(output_dir+'/k'+str(k)+'_'+str(nn)+'_'+str(alpha).split('.')[1]+'_thetas.json', 'a') as f:
                json.dump(thetas, f)
                f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run synthetic experiment.')
    parser.add_argument('--iterations', type=int, default=300, help='Number of iterations')
    parser.add_argument('--k', type=int, default=8, help='Number of models')
    parser.add_argument('--n', type=str, default='400,1000,5000,10000,20000', help='Number of pairwise comparisons by humans')
    parser.add_argument('--N', type=int, default=50000, help='Number of pairwise comparisons by LLMs')
    parser.add_argument('--alpha', type=float, default=0.05, help='Error probability')
    parser.add_argument('--noises', type=str, default='0.05,0.1,0.3', help='Noise levels')
    parser.add_argument('--seed', type=int, default=12345678, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='outputs/synthetic/sameNn', help='Output directory')

    args = parser.parse_args()

    sys.exit(main(iterations=args.iterations,
                   k=args.k, 
                   n=args.n, 
                   N=args.N, 
                   alpha=args.alpha, 
                   noises=args.noises, 
                   seed=args.seed, 
                   output_dir=args.output_dir))
    sys.exit(main())    