
# Prediction-Powered Ranking

This repository contains the code for the paper [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826).

### Dependencies

All the code is written in Python 3.\
In order to create a virtual environment and install the project dependencies you can run the following commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
python3 llm-ranking.py <file_human> <file_llm> <alpha> <ignore_ties> <n> <N> 
```
Parameters

- ` file_human ` : json file of human-annotated comparison data.

- ` file_llm ` : json file of LLM-annotated comparison data.

- `alpha` : error probability (real value between 0 and 1).

Optional parameters:

- `ignore_ties`: Default - 0. If 1, ignore comparisons from both datasets where the verdict is a tie.

- `n` : Default - number of comparisons in `file_human`. If specified, number of comparisons to subsample from ` file_human`.

- `N` : Default - number of comparisons in `file_llm`. If specified, number of comparisons to subsample from ` file_llm `.


### Datasets

The files `file_human` and `file_llm` must contain the keys `question_id`, `model_a`, `model_b` and `winner`.

Each model must be part of at least one comparison in both `file_human` and `file_llm`. In this comparison, the values of the keys `question_id`, `model_a` and `model_b` must be the same in both datasets.

The folder [data](data/) contains some sample data.


### Output

Example run:
```bash
python3 llm-ranking.py data/human_data.json data/llm_data.json 0.3
```
Example output:
```bash
     model    rank-set
----------    --------
   model_3      [2, 3]
   model_4      [4, 5]
   model_0      [1, 3]
   model_1      [1, 2]
   model_2      [4, 5]

```

The output is printed to console.\
Each output line contains the name of a model and the rank-set computed for it.


### Citation

If you use parts of the code in this repository for your own research purposes, please consider citing:

    @article{chatzi2024predictionpowered,
      title={Prediction-Powered Ranking of Large Language Models}, 
      author={Ivi Chatzi and Eleni Straitouri and Suhas Thejaswi and Manuel Gomez Rodriguez},
      year={2024},
      journal={arXiv preprint arXiv:2402.17826}
      }
