from Result import Result
from PlotRanksets import PlotRanksets
from ExperimentCollection import ExperimentCollection
from PlotIntersectSize import PlotIntersectSize

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "font.size": 35,
    "figure.figsize":(33,18),
    "lines.markersize": 20
})
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0


label_mapping={'baseline':r'$\textsc{Baseline}$',
               'human only':r'$\textsc{Human Only}$',
               'llm (claude-3-opus)':r'$\textsc{Llm Cl3}$',
               'llm (gpt-4-0125-preview)':r'$\textsc{Llm Gpt4}$',
               'llm (gpt-3.5-turbo)':r'$\textsc{Llm Gpt3.5}$',
               'ppr (claude-3-opus)': r'$\textsc{Ppr Cl3}$',
               'ppr (gpt-4-0125-preview)': r'$\textsc{Ppr Gpt4}$',
               'ppr (gpt-3.5-turbo)': r'$\textsc{Ppr Gpt3.5}$'}
model_mapping={'gpt-4': r'$\textbf{GPT 4}$',
                       'claude-v1': r'$\textbf{Claude 1}$',
                       'claude-instant-v1': r'$\textbf{Claude 1-I}$',
                       'gpt-3.5-turbo': r'$\textbf{GPT 3.5}$',
                       'vicuna-13b': r'$\textbf{Vicuna}$',
                       'palm-2': r'$\textbf{PaLM 2}$',
                       'koala-13b': r'$\textbf{Koala}$',
                       'RWKV-4-Raven-14B': r'$\textbf{RWKV}$',
                       'oasst-pythia-12b': r'$\textbf{Pythia}$',
                       'alpaca-13b': r'$\textbf{Alpaca}$',
                       'chatglm-6b': r'$\textbf{ChatGLM}$',
                       'fastchat-t5-3b': r'$\textbf{FastChat}$'
                       }
markers={'baseline':'*',
         'human only':'o',
         'llm (gpt-4-0125-preview)':'v',
         'llm (claude-3-opus)':'P',
         'llm (gpt-3.5-turbo)':'D',
         'ppr (claude-3-opus)': 'X',
         'ppr (gpt-4-0125-preview)': '^',
         'ppr (gpt-3.5-turbo)': 's'}
linestyles={'baseline':'-',
         'human only':'dashed',
         'llm (gpt-4-0125-preview)':'dashdot',
         'llm (claude-3-opus)':'dashdot',
         'llm (gpt-3.5-turbo)':'dashdot',
         'ppr (claude-3-opus)': 'dashdot',
         'ppr (gpt-4-0125-preview)': 'dotted',
         'ppr (gpt-3.5-turbo)': 'dotted'}

alpha=[0.2,0.01,0.05,0.1,0.025,0.075,0.15,0.3,0.25,0.4]
n=[100,150,200,500,1000,1500,2000,2500,3000]

results = Result('experiments_output/n'+str(1000)+'_a05/')
results.do_analysis()

plotter=PlotRanksets(results,label_mapping)
plotter.set_model_mapping(model_mapping)

# for model in results.models:
#     plotter.plot_ranksets_all(results.methods,model)
model_mapping={'gpt-4': 'GPT 4',
                       'claude-v1': 'Claude 1',
                       'claude-instant-v1': 'Claude 1-I',
                       'gpt-3.5-turbo': 'GPT 3.5',
                       'vicuna-13b': 'Vicuna',
                       'palm-2': 'PaLM 2',
                       'koala-13b': 'Koala',
                       'RWKV-4-Raven-14B': 'RWKV',
                       'oasst-pythia-12b': 'Pythia',
                       'alpaca-13b': 'Alpaca',
                       'chatglm-6b': 'ChatGLM',
                       'fastchat-t5-3b': 'FastChat'
                       }
plotter.set_model_mapping(model_mapping)
plotter.plot_ranksets_main(methods=['baseline','llm (gpt-4-0125-preview)','ppr (gpt-4-0125-preview)'],models=['gpt-4','claude-v1','vicuna-13b','palm-2'])
plotter.plot_ranks_dots6(['baseline','llm (gpt-4-0125-preview)','ppr (gpt-4-0125-preview)'])
plotter.plot_ranks_dots6(results.methods)

experiments=ExperimentCollection('experiments_output/',n=n,alpha=alpha)
experiments.do_analysis()

plotter=PlotIntersectSize(experiments,label_mapping)
plotter.set_markers(markers)
plotter.set_linestyles(linestyles)
plotter.set_model_mapping(model_mapping)

plotter.plot_intersect_size_setmethod_all(['ppr (gpt-4-0125-preview)','ppr (claude-3-opus)','ppr (gpt-3.5-turbo)'],alpha=sorted(alpha))
plotter.plot_intersect_size_setmethod_all(['ppr (gpt-4-0125-preview)','ppr (claude-3-opus)','ppr (gpt-3.5-turbo)'],alpha=sorted(alpha),contain=True)
plotter.plot_intersect_size_setn(1000,alpha=sorted(alpha),methods=['llm (gpt-3.5-turbo)','llm (claude-3-opus)','llm (gpt-4-0125-preview)','human only','ppr (gpt-3.5-turbo)','ppr (claude-3-opus)','ppr (gpt-4-0125-preview)'])
plotter.plot_contain_size_setn(1000,alpha=sorted(alpha),methods=['llm (gpt-3.5-turbo)','llm (claude-3-opus)','llm (gpt-4-0125-preview)','human only','ppr (gpt-3.5-turbo)','ppr (claude-3-opus)','ppr (gpt-4-0125-preview)'])

plotter.plot_intersect_size_setn(1000,alpha=sorted(alpha),methods=['llm (gpt-4-0125-preview)','human only','ppr (gpt-4-0125-preview)'],one=True)
plotter.plot_intersect_size_setn(1000,alpha=sorted(alpha),methods=['llm (gpt-3.5-turbo)','human only','ppr (gpt-3.5-turbo)'],one=True)
plotter.plot_intersect_size_setn(1000,alpha=sorted(alpha),methods=['llm (claude-3-opus)','human only','ppr (claude-3-opus)'],one=True)