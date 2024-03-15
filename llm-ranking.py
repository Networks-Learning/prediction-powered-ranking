import sys
import data_process
import estimate
import ranksets

def main()->0:
    """    
    Run:
        python3 llm-ranking.py <file_human> <file_llm> <alpha> <n> <N> <ties>
    
    Args:
        file1: human pairwise comparison dataset path
        file2: llm pairwise comparison dataset path
        alpha: error probability
        
        Optional args:
            ties: if 1 ignore ties
            n: number of human comparison samples to subsample
            N: number of llm comparison samples to subsample
            
    Output: 
        on each line prints the name of the model and the corresponding rank-set
    """
    
    alpha = float(sys.argv[3])
    assert (alpha>0 and alpha<1), 'Run:\n'\
        'python3 llm-ranking.py <file_human> <file_llm> <alpha> <n> <N> <ties>\n'\
            '<alpha> must be between 0 and 1.'
    
    n, N, ignore_ties = 0, 0, False
    
    if len(sys.argv) > 4:
        if sys.argv[4] == '1':
            ignore_ties=True
    if len(sys.argv) > 5:
        n = int(sys.argv[5])
    if len(sys.argv) > 6:
        N = int(sys.argv[6])
    
    
    human_data, human_instances = data_process.input_json(sys.argv[1], ignore_ties=ignore_ties)
    llm_data, llm_instances = data_process.input_json(sys.argv[2], ignore_ties=ignore_ties)
    
    models = data_process.model_list(human_data)
    k=len(models)
    
    small_dataset_human, small_dataset_llm, big_dataset_llm = data_process.sample_data(human_data, llm_data, human_instances, llm_instances, n=n, N=N)
    
    Dn = data_process.summarize_dataset(models, small_dataset_llm, small_dataset_human)
    DN = data_process.summarize_dataset(models, big_dataset_llm)
    
    thetahat, sigma = estimate.estimate(k, Dn, DN)
    ranks = ranksets.find_ranksets(k, thetahat, sigma, alpha=alpha)
    result=[(models[i],ranks[i]) for i in range(k)]
    
    print('     model    rank-set')
    print('----------    --------')
    for i in range(k):
        print("%10s %11s"%(result[i][0], result[i][1]))

    
if __name__ == '__main__':
    sys.exit(main())
