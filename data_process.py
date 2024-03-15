import json
import random
import numpy as np


def input_json(path, ignore_ties=False):
    """
    Parses a json file of pairwise comparison data

    Parameters
    ----------
    path : string
        file path of json file with data
        json file must include columns "question_id", "winner", "model_a", "model_b"
        
    ignore_ties : bool, optional
        The default is False.
        If True, ignores the pairwise comparisons that are ties
        
    Returns
    -------
    data : list of dictionaries
        Keys: 'question_id', 'model_a', 'model_b', 'winner'
        Each element corresponds to one row of the json dataset
    instances : dictionary
        Keys: every distinct tuple ('question_id', 'model_a', 'model_b')
        Values: list of indices of rows in the dataset that correspond to tuple
    """
    
    data = []
    instances = dict()
    with open(path) as f:
        for line in f:
            l = json.loads(line)
            
            if ignore_ties and (l['winner'] != 'model_a' and l['winner'] != 'model_b'):
                continue
            
            data.append({'question_id':l['question_id'],
                           'model_a':l['model_a'],
                           'model_b':l['model_b'],
                           'winner':l['winner']})
        
            t = (l['question_id'], l['model_a'], l['model_b'])
            if t in instances:
                instances[t].append(len(data)-1)
            else:
                instances[t] = [len(data)-1]
                
    return data, instances


def model_list(data):
    """
    Returns list of unique models in data

    Parameters
    ----------
    data : list of dictionaries
        Keys: must include 'model_a', 'model_b'
        Each element corresponds to one row of the json dataset

    Returns
    -------
    list
        Each element corresponds to a unique model.
    """
    
    models = set()
    for sample in data:
        models.add(sample['model_a'])
        models.add(sample['model_b'])
    return list(models)


def sample_data(human_data, llm_data, human_instances, llm_instances, n=0, N=0):
    """
    Samples from human and llm pairwise comparison datasets

    Parameters
    ----------
    human_data : list of dictionaries
        Human pairwise comparisons as returned by input_json
    llm_data : list of dictionaries
        LLM pairwise comparisons as returned by input_json
    human_instances : dictionary
        Indices of distinct human pairwise comparisons as returned by input_json
    llm_instances : dictionary
        Indices of distinct llm pairwise comparisons as returned by input_json
    n : int, optional
        Number of human samples. The default is 0, in which case all the samples are used.
    N : int, optional
        Number of predicted samples. The default is 0, in which case all the samples are used.

    Returns
    -------
    small_dataset_human : list of dictionaries
        Random samples from human_data
    small_dataset_llm : list of dictionaries
        Samples from llm_data where 'question_id','model_a','model_b' are the same as small_dataset_human
    big_dataset_llm : list of dictionaries
        Samples from llm_data that are not in small_dataset_llm

    """
    
    # find instances of (question_id, model_a, model_b) where there exists both human and llm annotation, and keep their indices
    common_instances = dict()
    for t in human_instances:
        if t in llm_instances:
            l1 = len(human_instances[t])
            l2 = len(llm_instances[t])
            h = random.sample(human_instances[t], min(l1,l2))
            l = random.sample(llm_instances[t], min(l1,l2))
            
            for i in range(len(h)):
                common_instances[(t,i)] = [h[i],l[i]]
                
    if (n == 0) or (n > len(common_instances)):
        n = len(common_instances)
    if (N == 0) or (N > len(llm_data)-n):
        N = len(llm_data) - n
        
    same_samples = random.sample([t for t in common_instances],n)
    small_dataset_human = []
    small_dataset_llm = []
    small_dataset_llm_index = set()
    for t in same_samples:
        small_dataset_human.append(human_data[common_instances[t][0]])
        llm_i = common_instances[t][1]
        small_dataset_llm_index.add(llm_i)
        small_dataset_llm.append(llm_data[llm_i])
    
    big_dataset_llm = [llm_data[i] for i in range(len(llm_data)) if i not in small_dataset_llm_index]
    big_dataset_llm=random.sample(big_dataset_llm,N)
    
    return small_dataset_human, small_dataset_llm, big_dataset_llm


def summarize_dataset(models, llm_dataset, human_dataset=[]):   
    """
    Transforms data into boolean matrix form

    Parameters
    ----------
    models : list
        List of unique models
    llm_dataset : list of dictionaries
        List of llm pairwise comparison samples
    human_dataset : list of dictionaries, optional
        List of human pairwise comparison samples, for the same 'question_id','model_a','model_b' as llm_dataset.
        The default is [].

    Returns
    -------
    summarized : dictionary
        Keys: 'winner_predicted', numpy.ndarray indicating the winner of each pairwise comparison in llm_dataset (1:model_a won, 0:model_b won, 0.5:tie)
              'model_a_matrix', numpy.ndarray where each column is a one-hot vector corresponding to each sample indicating which model was model_a
              'model_b_matrix', numpy.ndarray where each column is a one-hot vector corresponding to each sample indicating which model was model_b
              'winner_human' (optional), numpy.ndarray indicating the winner of each pairwise comparison in human_dataset (1:model_a won, 0:model_b won, 0.5:tie)

    """
    n = len(llm_dataset)
    
    if len(human_dataset) > 0:
        assert len(human_dataset) == len(llm_dataset), 'human_dataset and llm_dataset must have the same length'
    
    winner_human, winner_llm = np.zeros(shape=(n,1)), np.zeros(shape=(n,1))
    model_a_matrix, model_b_matrix = np.zeros(shape=(len(models),n)), np.zeros(shape=(len(models),n))
    
    for i in range(n):
        if len(human_dataset) > 0:
            if human_dataset[i]['winner'] == 'model_a':
                winner_human[i] = 1
            elif human_dataset[i]['winner'] == 'model_b':
                winner_human[i] = 0
            else:
                winner_human[i] = 0.5
        
        if llm_dataset[i]['winner'] == 'model_a':
            winner_llm[i] = 1
        elif llm_dataset[i]['winner'] == 'model_b':
            winner_llm[i] = 0
        else:
            winner_llm[i] = 0.5
        
        model_a_matrix[models.index(llm_dataset[i]['model_a'])][i] = 1
        model_b_matrix[models.index(llm_dataset[i]['model_b'])][i] = 1
    
    summarized = {'winner_predicted':winner_llm, 'model_a_matrix': model_a_matrix, 'model_b_matrix': model_b_matrix}
    if(len(human_dataset) > 0):
        summarized['winner_human']=winner_human
        
    return summarized
    
        
    
    
