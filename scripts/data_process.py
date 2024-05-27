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
        json file must include columns 
        "question_id", "winner", "model_a", "model_b"
        
    ignore_ties : bool, optional
        The default is False.
        If True, ignores the pairwise comparisons that are ties
        
    Returns
    -------
    data : list of dictionaries
        Keys: 'question_id', 'model_a', 'model_b', 'winner'
        Each element corresponds to one row of the json dataset
    """
    
    data = []
    with open(path) as f:
        for line in f:
            l = json.loads(line)
            
            if ignore_ties and (l['winner'] != 'model_a' and \
                                l['winner'] != 'model_b'):
                continue
            
            data.append({'question_id':l['question_id'],
                           'model_a':l['model_a'],
                           'model_b':l['model_b'],
                           'winner':l['winner']})
                
    return data


def model_list(data):
    """
    Returns list of unique models in data

    Parameters
    ----------
    data : list of dictionaries
        Keys: must include 'model_a', 'model_b'
        Each element corresponds to one row of the json dataset, ie
        one pairwise comparison

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

def get_common_instances(seed,human_data,llm_data,models):
    random.seed(seed)
    human_instances = dict()
    llm_instances = dict()

    for i in range(len(human_data)):
        sample=human_data[i]
        # ignore comparisons from other models
        if (sample['model_a'] not in models) or \
            (sample['model_b'] not in models):
            continue

        t=(sample['question_id'],sample['model_a'],sample['model_b'])
        if t not in human_instances:
            human_instances[t]=[i]
        else:
            human_instances[t].append(i)

    for i in range(len(llm_data)):
        sample = llm_data[i]
        # ignore comparisons from other models
        if (sample['model_a'] not in models) or \
                (sample['model_b'] not in models):
            continue

        t = (sample['question_id'], sample['model_a'], sample['model_b'])
        if t not in llm_instances:
            llm_instances[t] = [i]
        else:
            llm_instances[t].append(i)

    common_data_human=[]
    common_data_llm=[]
    for t in human_instances:
        if t in llm_instances:
            subsamples=min(len(human_instances[t]),len(llm_instances[t]))
            subsamples_human=random.sample(human_instances[t],subsamples)
            subsamples_llm=random.sample(llm_instances[t],subsamples)
            for i in subsamples_human:
                common_data_human.append(human_data[i])
            for i in subsamples_llm:
                common_data_llm.append(llm_data[i])
    return common_data_human, common_data_llm



def filter_data(seed, data, models, n=0):
    """
    Subsamples data so that there are the same number of pairwise comparisons
    per pair of models.

    Parameters
    ----------
    seed : seed for random sampling
    data : list of dictionaries
        Keys: must include 'model_a', 'model_b', 'winner'
        Each element corresponds to one row of the json dataset, ie
        one pairwise comparison
    models : list
        List of models. Ignore comparisons from other models
    n : int, optional
        The default is 0. If non-zero, subsample as many comparisons < n as
        possible such that there are the same number of comparisons per pair

    Returns
    -------
    filtered_data : list of dictionaries
        Each element corresponds to a pairwise comparison. There are the same
        number of pairwise comparisons for each pair of models.

    """
    random.seed(seed)
    filtered_data=[]
    
    pair_instances=dict()
    for i in range(len(models)):
        for j in range(i+1,len(models)):
            pair_instances[(models[i],models[j])]=[]
    
    models=set(models)
    
    for sample in data:
        # ignore comparisons from other models
        if (sample['model_a'] not in models) or \
            (sample['model_b'] not in models):
            continue
        
        # count how many pairwise comparisons there are per pair of models
        if (sample['model_a'],sample['model_b']) in pair_instances:
            pair_instances[(sample['model_a'],
                            sample['model_b'])].append(sample)
        elif (sample['model_b'],sample['model_a']) in pair_instances:
            pair_instances[(sample['model_b'],
                            sample['model_a'])].append(sample)
                
    lens=[(p,len(pair_instances[p])) for p in pair_instances]
    subsamples=min(lens, key = lambda t: t[1])[1]
    
    if n!=0:
        assert n>0, 'n must be positive'
        subsamples_ideal=n//len(pair_instances)
        subsamples=min(subsamples,subsamples_ideal)
    
    for p in pair_instances:
        uniform_samples=random.sample(pair_instances[p], subsamples)
        for sample in uniform_samples:
            filtered_data.append(sample)
    
    return filtered_data

def get_big_dataset_llm(llm_data,small_dataset_llm):
    """
    Returns list of samples in llm_data that are not in small_dataset_llm, ie
    returns the comparisons for which we only have llm judgments.
    

    Parameters
    ----------
    llm_data : list of dictionaries
        List of llm-judged pairwise comparison samples
    small_dataset_llm : llm_dataset : list of dictionaries
        List of llm-judged pairwise comparison samples for which we also have
        human judgments

    Returns
    -------
    big_dataset_llm : list of dictionaries
        List of llm-judged pairwise comparison samples for which we do not have
        human judgments

    """
    big_dataset_llm=[]
    llmset=set()
    for j in range(len(small_dataset_llm)):
        llmset.add(json.dumps(small_dataset_llm[j],sort_keys=True))
    for j in range(len(llm_data)):
        if json.dumps(llm_data[j],sort_keys=True) not in llmset:
            big_dataset_llm.append(llm_data[j])
    return big_dataset_llm
  
    
def summarize_dataset(models, llm_dataset, human_dataset=[]):   
    """
    Transforms data into boolean matrix form

    Parameters
    ----------
    models : list
        List of unique models
    llm_dataset : list of dictionaries
        List of llm-judged pairwise comparison samples
    human_dataset : list of dictionaries, optional
        List of human-judged pairwise comparison samples, 
        for the same 'question_id', 'model_a','model_b' as llm_dataset.
        The default is [], in which case only the llm_dataset is summarized.

    Returns
    -------
    summarized : dictionary
        Keys: 'winner1_predicted', numpy.ndarray indicating whether model_a won
               each pairwise comparison in llm_dataset
               (1:model_a won, 0:model_b won or tie)
              'winner2_predicted', numpy.ndarray indicating whether model_b won
               each pairwise comparison in llm_dataset
               (0:model_a won or tie, 1:model_b won)
              'model_a_matrix', numpy.ndarray, each column is a one-hot vector 
               corresponding to each sample indicating which model was model_a
              'model_b_matrix', numpy.ndarray, each column is a one-hot vector 
               corresponding to each sample indicating which model was model_b
              'winner1_human', numpy.ndarray indicating whether model_a won 
               each pairwise comparison in human_dataset 
               (1:model_a won, 0:model_b won or tie)
              'winner2_human', numpy.ndarray indicating whether model_b won
               each pairwise comparison in human_dataset 
               (0:model_a won or tie, 1:model_b won)
    """
    n = len(llm_dataset)
    
    if len(human_dataset) > 0:
        assert len(human_dataset) == len(llm_dataset),\
            'human_dataset and llm_dataset must have the same length'
    
    winner1_human, winner1_llm = np.zeros(shape=(n,1)), np.zeros(shape=(n,1))
    winner2_human, winner2_llm = np.zeros(shape=(n,1)), np.zeros(shape=(n,1))
    model_a_matrix = np.zeros(shape=(len(models),n))
    model_b_matrix = np.zeros(shape=(len(models),n))
    
    for i in range(n):
        if len(human_dataset) > 0:
            if human_dataset[i]['winner'] == 'model_a':
                winner1_human[i] = 1
            elif human_dataset[i]['winner'] == 'model_b':
                winner2_human[i] = 1
        
        if llm_dataset[i]['winner'] == 'model_a':
            winner1_llm[i] = 1
        elif llm_dataset[i]['winner'] == 'model_b':
            winner2_llm[i] = 1
        
        model_a_matrix[models.index(llm_dataset[i]['model_a'])][i] = 1
        model_b_matrix[models.index(llm_dataset[i]['model_b'])][i] = 1
    
    summarized = {'winner1_predicted':winner1_llm,
                  'winner2_predicted':winner2_llm,
                  'model_a_matrix': model_a_matrix,
                  'model_b_matrix': model_b_matrix}
    if(len(human_dataset) > 0):
        summarized['winner1_human']=winner1_human
        summarized['winner2_human']=winner2_human
        
    return summarized
    