## Configuration
File [config.json](config.json) lists the configuration parameters used in [llm-ranking.sh](llm-ranking.sh):

- `seed`: Seed used for random sampling.
- `iterations`: number of times each experiment is run
- `human_file`: dataset containing pairwise comparisons by humans
- `llm_files`: list of datasets containing pairwise comparisons by strong LLMs (one for each)
- `experiments_base_dir`: folder where the output will be stored.
- `judges`: list of names of the strong LLMs (same order as their corresponding files in `llm_files`)
- `n` : Number of comparisons to subsample from `human_file`.
- `alpha`: error probability parameter
- `ignore_ties`: Default 0. If 1, ignore comparisons where the verdict is a tie.
- `methods`: list of methods to construct rank-sets, among `baseline`, `human only`, `llm`, `ppr`.
- `models`: list of models to be ranked. If `[]`, all models in `human_file` are ranked.

The configuration parameters for [synthetic.sh](synthetic.sh) are inside the script.