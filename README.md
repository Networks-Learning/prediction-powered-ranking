# Prediction-Powered Ranking

This repository contains the code for the paper [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826) published in NeurIPS (2024) by Ivi Chatzi, Eleni Straitouri, Suhas Thejaswi and Manuel Gomez-Rodriguez.

__Contents__:
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Repository structure](#repository-structure)
- [Contact & attribution](#contact--attribution)

## Dependencies

All the code is written in Python 3.11.2\
In order to create a virtual environment and install the project dependencies you can run the following commands:

```commandline
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

In addition to the above dependencies, to run the notebooks and produce the figures it is necessary to install the additional requirements from [notebooks/requirements.txt](notebooks/requirements.txt).

## Usage
To reproduce the main experiments of the paper (Section 5 and Appendix D), run:
```
./scripts/llm-ranking.sh
```

To reproduce the synthetic experiments in Appendix E of the paper, run:
```
./scripts/synthetic.sh
```

To create the figures, run the notebooks in [notebooks](notebooks/).

<!-- 

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
Figures 1, 2, 6, 7 and 8 are stored in folder [plots/intersect_size](plots/intersect_size). -->

## Repository structure

```
├── data
│   ├── human.json
│   ├── gpt-4-0125-preview.json
│   ├── claude-3-opus-20240229.json
│   └── gpt-3.5-turbo.json
├── figures
├── notebooks
├── outputs
├── scripts
│   ├── llm-ranking.sh
│   ├── synthetic.sh
│   └── config.json
└── src
    ├── data_process.py
    ├── estimate.py
    ├── llm-ranking.py
    ├── plot_utils.py
    ├── ranksets.py
    ├── run_experiments.py
    └── synthetic.py

```

The folder [data](data/) contains the datasets used for our experimentation:
- [human.json](data/human.json): pairwise comparisons by humans.
- [gpt-4-0125-preview.json](data/gpt-4-0125-preview.json): pairwise comparisons by GPT 4.
- [claude-3-opus-20240229.json](data/claude-3-opus-20240229.json): pairwise comparisons by Claude 3.
- [gpt-3.5-turbo.json](data/gpt-3.5-turbo.json): pairwise comparisons by GPT 3.5.

The folder [figures](figures/) contains all the figures presented in the paper.

The folder [notebooks](notebooks/) contains python notebooks that generate all the figures included in the paper. 

The folder [outputs](outputs/) contains the output files produced by the experiments' scripts.

The folder [scripts](scripts/) contains bash scripts used to run all the experiments presented in the paper:
- [llm-ranking.sh](scripts/llm-ranking.sh): runs the main experiments of the paper, in Section 5 and Appendix D.
- [synthetic.sh](scripts/synthetic.sh): runs the synthetic experiments in Appendix E of the paper.
- [config.json](scripts/config.json): configuration file with parameters for the experiments ran by [llm-ranking.sh](scripts/llm-ranking.sh).

The folder [src](src/) contains all the code necessary to reproduce the results in the paper. Specifically:
- [data_process.py](src/data_process.py): inputs and subsamples from datasets.
- [estimate.py](src/estimate.py): implements Algorithms 1,3,4 from the paper to compute $\hat{\theta}$ and $\widehat{\Sigma}$.
- [llm-ranking.py](src/llm-ranking.py): reads config file and runs the experiments in Section 5 and Appendix D of the paper.
- [plot_utils.py](src/plot_utils.py): contains auxiliary functions for plotting.
- [ranksets.py](src/ranksets.py): implements Algorithm 2 from the paper to construct rank-sets.
- [run_experiments.py](src/run_experiments.py): runs experiments for all input parameters.
- [synthetic.py](src/synthetic.py): generates synthetic data and runs the synthetic experiments in Appendix E of the paper.


## Contact & attribution

In case you have questions about the code, you identify potential bugs or you would like us to include additional functionalities, feel free to open an issue or contact [Ivi Chatzi](mailto:ichatzi@mpi-sws.org).

If you use parts of the code in this repository for your own research purposes, please consider citing:

    @inproceedings{chatzi2024prediction,
      title={Prediction-Powered Ranking of Large Language Models},
      author={Ivi Chatzi and Eleni Straitouri and Suhas Thejaswi and Manuel Gomez Rodriguez},
      year={2024},
      booktitle = {Advances in Neural Information Processing Systems},
      publisher = {Curran Associates, Inc.}
      }
