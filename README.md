
# Prediction-Powered Ranking

This repository contains the code for the paper [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826).

## Dependencies

All the code is written in Python 3.11.2\
In order to create a virtual environment and install the project dependencies you can run the following commands:

```commandline
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 scripts/llm-ranking.py <file_config>
```
where

- ` file_config ` : json file of configuration parameters.

### Configuration parameters:

- `seed`: Seed used for random sampling.
- `iterations`: number of times each experiment is run
- `human_file`: dataset containing pairwise comparisons by humans
- `llm_files`: list of datasets containing pairwise comparisons by strong LLMs (one for each)
- `experiments_base_dir`: folder where the output will be stored.
- `judges`: list of names of the strong LLMs (same order as their corresponding files in `llm_files`)
- `n` : Number of comparisons to subsample from `human_file`.
- `alpha`: error probability parameter
- `ignore_ties`: Default - 0. If 1, ignore comparisons where the verdict is a tie.
- `methods`: list of methods to construct rank-sets, among `baseline`, `human only`, `llm`, `ppr`.
- `models`: list of models to be ranked. If `[]`, all models in `human_file` are ranked.

## Structure
The file [config.json](config.json) contains the configuration parameters we used for our experimentation.

The folder [data](data/) contains the datasets used for our experimentation:
- [human.json](data/human.json): pairwise comparisons by humans.
- [gpt-4-0125-preview.json](data/gpt-4-0125-preview.json): pairwise comparisons by GPT 4.
- [claude-3-opus-20240229.json](data/claude-3-opus-20240229.json): pairwise comparisons by Claude 3.
- [gpt-3.5-turbo.json](data/gpt-3.5-turbo.json): pairwise comparisons by GPT 3.5.

The folder [scripts](scripts/) contains the code to construct rank-sets and run experiments:
- [llm-ranking.py](scripts/llm-ranking.py): main file
- [data_process.py](scripts/data_process.py): inputs and subsamples from datasets
- [estimate.py](scripts/estimate.py): implements Algorithms 1,3,4 from the paper to compute $\hat{\theta}$ and $\widehat{\Sigma}$
- [ranksets.py](scripts/ranksets.py): implements Algorithm 2 from the paper to construct rank-sets
- [run_experiments.py](scripts/run_experiments.py): runs experiments for all input parameters

The folder [plots](plots/) contains the code to create the plots:
- [create_plots.py](plots/create_plots.py): generates all plots
- [Result.py](plots/Result.py): class that computes metrics for each experiment
- [ExperimentCollection.py](plots/ExperimentCollection.py): class that contains multiple experiments
- [PlotRanksets.py](plots/PlotRanksets.py): code to plot figures 3, 4, 9 and 10
- [PlotIntersectSize.py](plots/PlotIntersectSize.py): code to plot figures 1, 2, 6, 7 and 8


## Output

The results are stored in directory `experiments_base_dir`.
For every combination of n and $\alpha$ values in `n` and `alpha`, a new child folder is created inside
`experiments_base_dir`.
For example, for n=1000 and alpha=0.05, folder `experiments_base_dir/n1000_a05` will be created.

Inside each child folder, multiple json files are created (number equal to number of `iterations`).
Each json file is named `x.json` where `x` the iteration number.
These json files contain the rank-sets of their respective iteration, in json format:
```
{
    method 1:   { model 1: [low rank, up rank],
                  ...
                  model k: [low rank, up rank]
                },
    ...
    method m:   { model 1: [low rank, up rank],
                  ...
                  model k: [low rank, up rank]
                }
 }

```

### Plots
First run the experiments via [llm-ranking.py](scripts/llm-ranking.py) using [config.json](config.json).

Then, install the plot code requirements:
```commandline
pip install -r plots/plot_requirements.txt
```
Then, run:
```commandline
python3 plots/create_plots.py
```

Figures 3, 4, 9 and 10 are stored in folder [plots/ranksets](plots/ranksets).\
Figures 1, 2, 6, 7 and 8 are stored in folder [plots/intersect_size](plots/intersect_size).

## Citation

If you use parts of the code in this repository for your own research purposes, please consider citing:

    @article{chatzi2024predictionpowered,
      title={Prediction-Powered Ranking of Large Language Models},
      author={Ivi Chatzi and Eleni Straitouri and Suhas Thejaswi and Manuel Gomez Rodriguez},
      year={2024},
      journal={arXiv preprint arXiv:2402.17826}
      }
