## Output
The results of [llm-ranking.sh](../scripts/llm-ranking.sh) are stored in directory [chatbot_arena](chatbot_arena/).
For every combination of n and $\alpha$ values in `n` and `alpha` set in [config.json](../scripts/config.json), a new child folder is created inside
[chatbot_arena](chatbot_arena/).
For example, for n=1000 and alpha=0.05, folder `chatbot_arena/n1000_a05` will be created.

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