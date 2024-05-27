import json
import os
from statistics import mean,stdev
import numpy as np


class Result:
    def __init__(self,input_path,parameters=None):
        if parameters is None:
            params=input_path.split('/')[-2].split('_')
            params[0]=int(params[0][1:])
            params[1]=float('0.'+params[1][1:])
            self.parameters={'n':params[0],'alpha':params[1]}
            exps=dict()
            it = 0
            for file in os.listdir(input_path):
                with open(input_path + file) as f:
                    for line in f:
                        l = json.loads(line)
                        exps[it] = l
                        it+=1

            iterations=len(exps)

            methods = [m for m in exps[0]]
            models = [m for m in exps[0][methods[0]]]

            results={method:{model:[] for model in models} for method in methods}

            for method in methods:
                for model in models:
                    for i in range(iterations):
                        results[method][model].append(exps[i][method][model])

            self.results=results
            self.iterations=iterations
            self.models=models
            self.methods=methods
        else:
            filepath=input_path+'n'+str(parameters['n'])+'_a'+str(parameters['alpha'])+'.json'
            with open(filepath) as f:
                results=json.load(f)
            self.results=results['results']
            self.parameters=parameters
            self.methods=[method for method in self.results]
            self.models=[model for model in self.results[self.methods[0]]]
            self.iterations=len(self.results[self.methods[0]][self.models[0]])

    def store_summary(self,folder):
        path=folder+'n'+str(self.parameters['n'])+'_a'+str(self.parameters['alpha'])+'.json'
        stor={'n':self.parameters['n'],'alpha':self.parameters['alpha'],'results':self.results}
        with open(path,'w') as f:
            json.dump(stor,f)

    def rankset_low_up_size(self):
        self.lows = {method:{model: [] for model in self.models} for method in self.methods}
        self.ups = {method:{model: [] for model in self.models} for method in self.methods}
        self.sizes = {method:{model: [] for model in self.models} for method in self.methods}
        for method in self.methods:
            for model in self.results[method]:
                self.lows[method][model] = [s[0] for s in self.results[method][model]]
                self.ups[method][model] = [s[1] for s in self.results[method][model]]
                self.sizes[method][model] = [s[1] - s[0] + 1 for s in self.results[method][model]]

    def set_model_order(self):
        models=self.models
        avg_lows=[mean(self.lows['baseline'][model]) for model in models]
        sorted_indices = np.argsort(avg_lows)
        self.models = [models[i] for i in sorted_indices]

    def rankset_stats(self):
        self.avg_lows={method:[mean(self.lows[method][model]) for model in self.models] for method in self.methods}
        self.avg_ups = {method: [mean(self.ups[method][model]) for model in self.models] for method in self.methods}
        self.avg_sizes_model = {method: [mean(self.sizes[method][model]) for model in self.models] for method in self.methods}
        self.ci_lows = {method: [stdev(self.lows[method][model])*1.96/np.sqrt(self.iterations) for model in self.models] for method in self.methods}
        self.ci_ups = {method: [stdev(self.ups[method][model])*1.96/np.sqrt(self.iterations) for model in self.models] for method in self.methods}
        self.ci_sizes_model = {method: [stdev(self.sizes[method][model])*1.96/np.sqrt(self.iterations) for model in self.models] for method in self.methods}

        all_sizes={method:[self.sizes[method][model] for model in self.models] for method in self.methods}
        all_sizes={method:[x for xs in [all_sizes[method][m] for m in range(len(self.models))] for x in xs] for method in self.methods}
        self.avg_sizes={method:mean(all_sizes[method]) for method in self.methods}
        self.ci_sizes={method:stdev(all_sizes[method])*1.96/np.sqrt(len(all_sizes[method])) for method in self.methods}

    def find_ranks(self):
        ranks = {method:{model:[0 for _ in range(len(self.models))] for model in self.models} for method in self.methods}
        sizes_stats = {method: {model: [0 for _ in range(len(self.models))] for model in self.models} for method in self.methods}
        for method in self.methods:
            for model in self.models:
                for i in range(self.iterations):
                    for r in range(len(self.models)):
                        if self.lows[method][model][i]<=r+1 and self.ups[method][model][i]>=r+1:
                            ranks[method][model][r]+=1
                        if self.sizes[method][model][i]==r+1:
                            sizes_stats[method][model][r]+=1
                for r in range(len(self.models)):
                    ranks[method][model][r]/=self.iterations
                    sizes_stats[method][model][r] /= self.iterations
        self.ranks=ranks
        self.top_ranks = {method: {model: np.argmax(self.ranks[method][model]) for model in self.models} for method in
                     self.methods}
        self.sizes_stats = sizes_stats


    def all_ranksets(self,method,model):
        ranksets=dict()
        for i in range(self.iterations):
            rankset=(self.lows[method][model][i],self.ups[method][model][i])
            if rankset in ranksets:
                ranksets[rankset]+=1
            else:
                ranksets[rankset]=1
        return ranksets


    def total_solutions(self):
        self.sols = {method: [1 for _ in range(self.iterations)] for method in self.methods}
        for method in self.methods:
            for model in self.models:
                for i in range(self.iterations):
                    self.sols[method][i] *= self.sizes[method][model][i]
        self.avg_sols = {method: mean(self.sols[method]) for method in self.methods}
        self.ci_sols = {method: stdev(self.sols[method])*1.96/np.sqrt(self.iterations) for method in self.methods}

    def find_errors(self,type='intersect'):
        errors_per_model={method:{model:0 for model in self.models} for method in self.methods if method!='baseline'}
        errors_per_iteration={method:[0 for _ in range(self.iterations)] for method in self.methods if method!='baseline'}
        correct_iterations={method:[0 for _ in range(self.iterations)] for method in self.methods if method!='baseline'}
        singletons={model:[] for model in self.models}

        for method in self.methods:
            if method=='baseline':
                continue
            for i in range(self.iterations):
                correct_iteration=True
                for model in self.models:
                    if self.lows['baseline'][model][i]==self.ups['baseline'][model][i]:
                        singletons[model].append(i)
                    if type=='intersect':
                        if self.lows[method][model][i]>self.ups['baseline'][model][i] or self.ups[method][model][i]<self.lows['baseline'][model][i]:
                            errors_per_model[method][model]+=1
                            errors_per_iteration[method][i]+=1
                            correct_iteration=False
                    elif type=='contain':
                        if self.lows[method][model][i]>self.lows['baseline'][model][i] or self.ups[method][model][i]<self.ups['baseline'][model][i]:
                            errors_per_model[method][model]+=1
                            errors_per_iteration[method][i]+=1
                            correct_iteration=False
                if correct_iteration:
                    correct_iterations[method][i]=1
        return singletons, errors_per_model, errors_per_iteration, correct_iterations

    def find_correct(self):
        _,_,_, correct_iterations_intersect=self.find_errors()
        _, _, _, correct_iterations_contain = self.find_errors(type='contain')
        self.correct_intersect={method:sum(correct_iterations_intersect[method]) for method in self.methods if method!='baseline'}
        self.correct_contain = {method: sum(correct_iterations_contain[method]) for method in self.methods if
                                  method != 'baseline'}

    def do_analysis(self):
        self.rankset_low_up_size()
        self.set_model_order()
        self.rankset_stats()
        self.total_solutions()
        self.find_ranks()
        self.find_correct()

