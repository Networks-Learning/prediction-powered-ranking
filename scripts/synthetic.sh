iterations=300   # number of iterations for each experiment
k=8   # number of models
n='[400,1000,5000,10000,20000]'   # number of synthetic pairwise comparisons by humans
N=50000   # number of synthetic pairwise comparisons by a strong LLM
alpha=0.1   # error probability
noises='[0.05, 0.1, 0.3]'   # noise levels for the strong LLMs
seed=12345678  # random seed
output_dir='outputs/synthetic/'   # output directory

python -m src.synthetic --iterations=$iterations --k=$k --n="$n" --N=$N --alpha=$alpha --noises="$noises" --seed=$seed --output_dir="$output_dir"