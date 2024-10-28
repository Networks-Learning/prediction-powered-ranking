from src import data_process
from src import estimate
from src import ranksets
import json
import os

def run_experiment(seed,
                   file_human,
                   files_llm,
                   judges,
                   models=[],
                   alpha=0.1,
                   n=0,
                   ignore_ties=False,
                   methods=['human only']):
    """
    Calculates ranksets for a given list of models from a given dataset of 
    human- and LLM-judged pairwise comparisons.

    Parameters
    ----------
    seed : seed for random sampling
    file_human : string
        file path of json file with human-judged pairwise comparisons
        json file must include columns 
        "question_id", "winner", "model_a", "model_b"
    files_llm : list of strings
        File paths of json files with LLM-judged pairwise comparisons.
        Each file contains the same comparisons, but judged by a different LLM.
        json files must include columns 
        "question_id", "winner", "model_a", "model_b"
    judges : list
        List of LLM-judges corresponding to each file in files_llm
    models : list, optional
        List of models being ranked. The default is [], in which case all
        models in the datasets are ranked.
    alpha : float, optional
        Desired error rate. The default is 0.1.
    n : int, optional
        Number of human judgments to use. The default is 0, in which case all 
        the human judgments are used.
    ignore_ties : bool, optional
        The default is False.
        If True, ignore pairwise comparisons that are ties.
    methods : list of strings, optional
        List of types of methods to compute ranksets from.
        The default is ['human only'].
        Possible methods:
            'baseline': ranksets using all the human data
            'human only': ranksets using only the n human data
            'llm': ranksets using all the LLM data
             (different result for each judge)
            'ppr': ranksets combining the LLM data with n human data
             (different result for each judge)
        

    Returns
    -------
    results : dictionary
        Resulting ranksets. Dictionary with each mode as keys, and items the
        ranksets for each mode. Ranksets are a dictionary with each model as
        keys and the resulting ranksets as items in form [low rank, up rank]

    """
    human_data=data_process.input_json(file_human)
    llm_data=[data_process.input_json(files_llm[i]) for i in range(
        len(judges))]

    if models==[]:
        models=data_process.model_list(human_data)
    k=len(models)

    human_common, small_dataset_human = [[] for _ in range(len(judges))], [[] for _ in range(len(judges))]
    llm_common, small_dataset_llm = [[] for _ in range(len(judges))], [[] for _ in range(len(judges))]
    big_dataset_llm=[[] for _ in range(len(judges))]
    for ii in range(len(judges)):
        human_common[ii], llm_common[ii] = data_process.get_common_instances(seed, human_data, llm_data[ii], models)
        human_common[ii] = data_process.filter_data(seed, human_common[ii], models)
        llm_common[ii] = data_process.filter_data(seed, llm_common[ii], models)
        small_dataset_human[ii] = data_process.filter_data(seed, human_common[ii], models, n=n)
        small_dataset_llm[ii] = data_process.filter_data(seed, llm_common[ii], models, n=n)
        big_dataset_llm[ii] = data_process.get_big_dataset_llm(llm_common[ii], small_dataset_llm[ii])
    
    Dn=[data_process.summarize_dataset(
        models, small_dataset_llm[i], small_dataset_human[i]) for i in range(
            len(judges))]
    DN=[data_process.summarize_dataset(
        models, big_dataset_llm[i]) for i in range(len(judges))]

    results=dict()

    if 'baseline' in methods:
        human_data = data_process.filter_data(seed, human_data, models)
        thetahat,sigma=estimate.sample_avg(
            k, data_process.summarize_dataset(models, human_data))
        rank_sets=ranksets.find_ranksets(k,thetahat,sigma,alpha=alpha)
        results['baseline']={models[i]:rank_sets[i] for i in range(k)}
        
    if 'human only' in methods:
        thetahat,sigma=estimate.sample_avg(k, Dn[0],'human')
        rank_sets=ranksets.find_ranksets(k,thetahat,sigma,alpha=alpha)
        results['human only']={models[i]:rank_sets[i] for i in range(k)}
        
    if 'llm' in methods:
        for j in range(len(judges)):
            llm_data[j] = data_process.filter_data(seed, llm_data[j], models)
            thetahat,sigma=estimate.sample_avg(
                k, data_process.summarize_dataset(models, llm_data[j]))
            rank_sets=ranksets.find_ranksets(k,thetahat,sigma,alpha=alpha)
            results['llm ('+judges[j]+')']={
                models[i]:rank_sets[i] for i in range(k)}
    
    if 'ppr' in methods:
        for j in range(len(judges)):
            lhat_opt=estimate.lhat_opt(k,Dn[j],DN[j])
            thetahat,sigma=estimate.estimate(k, Dn[j], DN[j],lhat=lhat_opt)
            rank_sets=ranksets.find_ranksets(k,thetahat,sigma,alpha=alpha)
            results['ppr ('+judges[j]+')']={
                models[i]:rank_sets[i] for i in range(k)}    
    return results

def run_experiments(seeds,
                    output_directory, 
                    human_file, 
                    llm_files,
                    judges,
                    iterations=10, 
                    alpha=0.1, 
                    n=0, 
                    ignore_ties=False,
                    methods=['human only'],
                    models=[]):
    """
    Calculates ranksets for a given list of models from a given dataset of 
    human- and LLM-judged pairwise comparisons and stores results in a folder.
    Does multiple iterations and stores results for each in a file named
    'i.json' inside the output folder, where i is the current iteration number.

    Parameters
    ----------
    seeds : seed for random sampling
    output_directory : string
        path of directory to write output files
    human_file : string
        file path of json file with human-judged pairwise comparisons
        json file must include columns 
        "question_id", "winner", "model_a", "model_b"
    llm_files : list of strings
        File paths of json files with LLM-judged pairwise comparisons.
        Each file contains the same comparisons, but judged by a different LLM.
        json files must include columns 
        "question_id", "winner", "model_a", "model_b"
    judges : list
        List of LLM-judges corresponding to each file in files_llm
    iterations : int, optional
        Number of iterations to run. The default is 10.
    alpha : float, optional
        Desired error rate. The default is 0.1.
    n : int, optional
        Number of human judgments to use. The default is 0, in which case all 
        the human judgments are used.
    ignore_ties : bool, optional
        The default is False.
        If True, ignore pairwise comparisons that are ties.
    methods : list of strings, optional
        List of types of methods to compute ranksets from.
        The default is ['human only'].
        Possible methods:
            'baseline': ranksets using all the human data
            'human only': ranksets using only the n human data
            'llm': ranksets using all the LLM data
             (different result for each judge)
            'ppr': ranksets combining the LLM data with n human data
             (different result for each judge)
    models : list, optional
        List of models being ranked. The default is [], in which case ppr
        models in the datasets are ranked.

    Returns
    -------
    None.

    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for i in range(iterations):
        filepath=output_directory+str(i)+'.json'
        if os.path.exists(filepath):
            os.remove(filepath)
        with open(filepath,'a') as f:
            try:
                result=run_experiment(
                    seeds[i],human_file,llm_files,judges=judges,models=models,
                    alpha=alpha,n=n,ignore_ties=ignore_ties,methods=methods)
                # output={'iteration':i,'result':result}
                output=result
                json.dump(output, f)
            except AssertionError as msg:
                print('i,a,n',i,alpha,n,'\n',msg)
                continue