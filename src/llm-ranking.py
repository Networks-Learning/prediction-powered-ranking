import json
import random
import os
import sys
from src import run_experiments as run_experiments


def check_input(configs):
    """
    Checks if config file contains valid config parameters

    Parameters
    ----------
    configs : dict (from json)
        config parameters.

    Returns
    -------
    None.

    """
    assert isinstance(configs['iterations'], int), \
        "Invalid input: iterations = " + str(configs['iterations']) + \
        "\niterations must be an integer.\nPlease edit config.json"
    assert configs['iterations'] > 0, \
        "Invalid input: iterations = " + str(configs['iterations']) + \
        "\niterations must be greater than 0.\nPlease edit config.json"
    assert os.path.exists(configs['human_file']), "Invalid input: " + \
                                                  configs['human_file'] + " does not exist"

    for f in configs['llm_files']:
        assert os.path.exists(f), "Invalid input: " + f + " does not exist"

    for nn in configs['n']:
        assert isinstance(nn, int), \
            "Invalid input: n = " + str(nn) + \
            "\nn values must be integers.\nPlease edit config.json"
        assert nn >= 0, "Invalid input: n = " + str(nn) + \
                        "\nn values cannot be negative.\nPlease edit config.json"

    for a in configs['alpha']:
        assert isinstance(a, float), \
            "Invalid input: alpha = " + str(a) + \
            "\nalpha values must be floats.\nPlease edit config.json"
        assert a > 0 and a < 1, "Invalid input: alpha = " + str(a) + \
                                "\nalpha values must be between 0 and 1.\nPlease edit config.json"

    assert configs['ignore_ties'] == 0 or configs['ignore_ties'] == 1, \
        "Invalid input: " + str(configs['ignore_ties']) + \
        "\nignore_ties must be either 0 or 1.\nPlease edit config.json"

    methods = {'baseline', 'human only', 'llm', 'ppr'}
    for m in configs['methods']:
        assert m in methods, \
            "Invalid method: " + m + \
            ".\nValid methods: baseline, human only, llm, ppr." \
            "\nPlease edit config.json"


def main() -> 0:
    """
    Run: 
        python3 llm-ranking.py <config_file>
    
    Params:
        <config_file>: path to config file containing parameters

    """
    assert len(sys.argv) == 2, "Run:\npython3 llm-ranking.py <config_file>"

    if sys.argv[1] == '-help':
        print("Run:\npython3 llm-ranking.py <config_file>")
        return 0

    config_file = sys.argv[1]

    assert (os.path.exists(config_file)), \
        "config_file " + str(config_file) + " does not exist.\n" \
                                            "Please use a valid config_file and run:\n" \
                                            "python3 llm-ranking.py <config_file>"

    with open(config_file, 'r') as f:
        configs = json.load(f)

    check_input(configs)

    seed = configs['seed']
    iterations = configs['iterations']
    human_file = configs['human_file']
    llm_files = configs['llm_files']
    experiment_base_dir = os.getcwd() + configs['experiment_base_dir']
    judges = configs['judges']
    n = configs['n']
    alpha = configs['alpha']
    methods = configs['methods']
    models = configs['models']

    ignore_ties = False
    if configs['ignore_ties'] == 1:
        ignore_ties = True

    random.seed(seed)
    seeds = [random.sample(range(10, 100000000), iterations) for _ in range(
        len(n) * len(alpha))]

    experiment_dirs = []
    i = 0
    for nn in n:
        for a in alpha:
            experiment_dirs.append(
                experiment_base_dir + '/n' + str(nn) + '_a' + str(a).split('.')[1] + '/')
            run_experiments.run_experiments(seeds[i],
                                            experiment_dirs[-1],
                                            human_file,
                                            llm_files,
                                            judges, iterations,
                                            alpha=a, n=nn,
                                            ignore_ties=ignore_ties,
                                            models=models,
                                            methods=methods)


if __name__ == '__main__':
    sys.exit(main())
