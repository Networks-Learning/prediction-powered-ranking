import json
import os
from statistics import mean,stdev
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib
import os
import matplotlib.cm as cm

BASELINE='baseline'


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
        avg_lows=[mean(self.lows[BASELINE][model]) for model in models]
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
        errors_per_model={method:{model:0 for model in self.models} for method in self.methods if method!=BASELINE}
        errors_per_iteration={method:[0 for _ in range(self.iterations)] for method in self.methods if method!=BASELINE}
        correct_iterations={method:[0 for _ in range(self.iterations)] for method in self.methods if method!=BASELINE}
        singletons={model:[] for model in self.models}

        for method in self.methods:
            if method==BASELINE:
                continue
            for i in range(self.iterations):
                correct_iteration=True
                for model in self.models:
                    if self.lows[BASELINE][model][i]==self.ups[BASELINE][model][i]:
                        singletons[model].append(i)
                    if type=='intersect':
                        if self.lows[method][model][i]>self.ups[BASELINE][model][i] or self.ups[method][model][i]<self.lows[BASELINE][model][i]:
                            errors_per_model[method][model]+=1
                            errors_per_iteration[method][i]+=1
                            correct_iteration=False
                    elif type=='contain':
                        if self.lows[method][model][i]>self.lows[BASELINE][model][i] or self.ups[method][model][i]<self.ups[BASELINE][model][i]:
                            errors_per_model[method][model]+=1
                            errors_per_iteration[method][i]+=1
                            correct_iteration=False
                if correct_iteration:
                    correct_iterations[method][i]=1
        return singletons, errors_per_model, errors_per_iteration, correct_iterations

    def find_correct(self):
        _,_,_, correct_iterations_intersect=self.find_errors()
        _, _, _, correct_iterations_contain = self.find_errors(type='contain')
        self.correct_intersect={method:sum(correct_iterations_intersect[method]) for method in self.methods if method!=BASELINE}
        self.correct_contain = {method: sum(correct_iterations_contain[method]) for method in self.methods if
                                  method != BASELINE}

    def do_analysis(self):
        self.rankset_low_up_size()
        self.set_model_order()
        self.rankset_stats()
        self.total_solutions()
        self.find_ranks()
        self.find_correct()

class ExperimentCollection:
    def __init__(self,input_directory,n,alpha):
        self.experiments=[]
        self.parameter_list=[]
        for nn in n:
            for a in alpha:
                # result=Result(input_directory,{'n':nn,'alpha':a})
                result=Result(input_directory+'n'+str(nn)+'_a'+str(a).split('.')[1]+'/')
                self.experiments.append(result)
                self.parameter_list.append({'n':nn,'alpha':a})

    def do_analysis(self,param_list=[]):
        if param_list==[]:
            param_list=self.parameter_list
        for experiment in self.experiments:
            if experiment.parameters in param_list:
                experiment.do_analysis()

class PlotRanksets:
    def __init__(self,results,mapping):
        self.results=results
        self.parameters=results.parameters
        self.method_mapping=mapping
        self.methods=results.methods
        self.models=results.models
        self.directory='../figures/chatbot_arena_ranksets/'
        # Create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def set_model_mapping(self,model_mapping):
        self.model_mapping=model_mapping
        self.model_labels=[model_mapping[model] for model in self.models]

    # plots figures 3 and 9
    def plot_ranks_dots6(self, methods, save=True):
        data = [[self.results.ranks[method][model] for model in self.models] for method in methods]

        # fig.set_constrained_layout_pads(hspace=0, wspace=2)
        if len(methods)==8:
            fig, axs = plt.subplots(2, 4, sharey=True, figsize=(37,18))
        else:
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=(37,9))

        cmap0 = plt.get_cmap('coolwarm')

        for j in range(len(methods)):
            for m in range(len(self.models)):
                xmin, xmax = -1, -1
                for i in range(len(data[j][m])):
                    if xmin == -1 and data[j][m][i] > 0:
                        xmin = i
                    if data[j][m][i] > 0:
                        xmax = i
                fig.axes[j].plot(np.linspace(xmin, xmax, 2), np.linspace(m, m, 2), color='gainsboro', zorder=0, linewidth=30,
                         solid_capstyle='round')
                x = [i for i in range(12)]
                y = np.linspace(m, m, 12)
                mappable=fig.axes[j].scatter(x,y,edgecolors='black',zorder=1,c=data[j][m],cmap=cmap0,
                               s=[d * 500 for d in data[j][m]],vmin=0, vmax=1)
            ax = fig.axes[j]
            ax.set_xticks(np.arange(len(self.models)), labels=np.arange(1, 13))
            ax.set_yticks(np.arange(len(self.models)), labels=self.model_labels)
            # ax.set_facecolor("whitesmoke")
            ax.set_xlim(-0.5,11.5)
            ax.set_title(self.method_mapping[methods[j]])
            ax.set_ylim(-0.5,11.5)
            if j>=4 or len(methods)<4:
                ax.set_xlabel('Ranking position')
            ax.spines[['right', 'top', 'left']].set_visible(False)
            ax.tick_params(left=False)

        fig.subplots_adjust(wspace=0.12, hspace=0.25)
        cbar = fig.colorbar(mappable, ax=axs.ravel().tolist())
        cbar.ax.set_ylabel('Empirical probability', rotation=-90, va="bottom")


        if save:
            filepath = self.directory + 'ranks_'+str(len(methods))+'_n' + str(
                self.parameters['n']) + '_a' + str(self.parameters['alpha']).split('.')[1]
            plt.savefig(filepath + '.pdf', bbox_inches='tight')
            print('Saved figure at ' + filepath)

    # plots figure 10
    def plot_ranksets_all(self,methods,model,save=True):
        ranksets = [self.results.all_ranksets(method, model) for method in methods]
        fig, axs = plt.subplots(2, 4, sharey=True, figsize=(37,15))
        if len(methods)==3:
            fig, axs = plt.subplots(1,3, sharey=True, figsize=(37,7))

        cmap1 = LinearSegmentedColormap.from_list('', ['lightblue', 'midnightblue'])

        for j in range(len(methods)):
            ax=fig.axes[j]
            yy=1
            for rankset in sorted(ranksets[j]):
                linewidth=9
                if len(methods)<4:
                    linewidth=15
                ax.plot(np.linspace(rankset[0], rankset[1], 2), np.linspace(yy, yy, 2),
                        markersize=ranksets[j][rankset] / 10,
                        c=cmap1(ranksets[j][rankset] / self.results.iterations), linewidth=linewidth,
                        solid_capstyle='round')
                yy+=1
            ax.set_ylim(0, 1+max([len(ranksets[j]) for j in range(len(ranksets))]))
            ax.set_xlim(0.5, 12.5)
            if j>=4 or len(methods)<4:
                ax.set_xlabel('Ranking position')
            ax.set_title(self.method_mapping[methods[j]])
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.set_xticks(np.linspace(1, 12, 12))
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        fig.subplots_adjust(wspace=0.12, hspace=0.3)
        fig.suptitle(self.model_mapping[model])
        if len(methods)<4:
            plt.subplots_adjust(top=0.8)
        sm = plt.cm.ScalarMappable(cmap=cmap1)
        sm.set_clim(vmin=0, vmax=1000)
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist())
        cbar.set_ticks(np.linspace(0,1000,5),labels=np.linspace(0,1,5))
        cbar.ax.set_ylabel('Empirical probability', rotation=-90, va="bottom")

        if save:
            filepath = self.directory + 'all_ranksets_model_' + model + '_n' + str(
                self.parameters['n']) + '_a' + str(self.parameters['alpha']).split('.')[1] + '.pdf'
            if len(methods)<4:
                filepath = self.directory + '3_ranksets_model_' + model + '_n' + str(
                    self.parameters['n']) + '_a' + str(self.parameters['alpha']).split('.')[1] + '.pdf'

            plt.savefig(filepath, bbox_inches='tight')
            print('Saved figure at ' + filepath)

    # plots figure 4
    def plot_ranksets_main(self,methods,models,save=True):
        ranksets = [[self.results.all_ranksets(method, model) for method in methods] for model in models]
        fig, axs = plt.subplots(1,len(models), sharey=True, figsize=(37,7))


        cmap0 = LinearSegmentedColormap.from_list('', ['lightblue', 'midnightblue'])

        for j in range(len(models)):
            ax=fig.axes[j]
            yy=1
            lims = [0, 5]
            if j >= 2:
                lims = [2, 9]
            for i in range(len(methods)):
                if i==1 and j>=2:
                    yy+=1
                if i==1 and j==1:
                    yy+=0.5
                for rankset in sorted(ranksets[j][i]):
                    linewidth=13
                    ax.plot(np.linspace(rankset[0], rankset[1], 2), np.linspace(yy, yy, 2),
                            markersize=ranksets[j][i][rankset] / 10,
                            c=cmap0(ranksets[j][i][rankset] / self.results.iterations), linewidth=linewidth,
                            solid_capstyle='round')
                    yy+=1
                if yy<=4:
                    yy=4
                elif yy<=8:
                    yy=8
                if yy<12:
                    ax.plot(np.linspace(lims[0]+0.7,lims[1]+0.3,2),np.linspace(yy-0.5,yy-0.5,2),color='grey',linestyle='--')
            ax.set_ylim(0,yy)
            ax.set_xlim(lims[0]+0.5,lims[1]+0.5)
            ax.set_title(self.model_mapping[models[j]])
            ax.set_xlabel('Ranking position')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.set_xticks(np.linspace(lims[0]+1, lims[1], lims[1]-lims[0]))
            ax.set_yticks([2,5.5,12],labels=[self.method_mapping[method] for method in methods])
            ax.tick_params(axis='y', which='both', left=False, right=False)

        fig.subplots_adjust(wspace=0.12, hspace=0.3)
        sm = plt.cm.ScalarMappable(cmap=cmap0)
        sm.set_clim(vmin=0, vmax=1000)
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist())
        cbar.set_ticks(np.linspace(0,1000,5),labels=np.linspace(0,1,5))
        cbar.ax.set_ylabel('Empirical probability', rotation=-90, va="bottom")

        if save:
            filepath = self.directory + 'ranksets_main_n' + str(
                self.parameters['n']) + '_a' + str(self.parameters['alpha']).split('.')[1] + '.pdf'

            plt.savefig(filepath, bbox_inches='tight')
            print('Saved figure at ' + filepath)

class PlotIntersectSize:
    def __init__(self,experiment_collection,mapping):
        self.experiment_collection=experiment_collection
        self.parameter_list=experiment_collection.parameter_list
        self.method_mapping=mapping
        self.methods=experiment_collection.experiments[0].methods
        self.models=experiment_collection.experiments[0].models
        self.directory='../figures/chatbot_arena_metrics/'
        # Create directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def set_model_mapping(self,model_mapping):
        self.model_mapping=model_mapping
        self.model_labels=[model_mapping[model] for model in self.models]

    def set_markers(self,markers):
        self.markers=markers
    def set_colors(self,colors):
        self.colors=colors

    def set_linestyles  (self,linestyles):
        self.linestyles=linestyles


    # plots figures 1 and 6
    def plot_intersect_size_setn(self,n,methods=0,alpha=0,save=True, one=False,contain=False):
        n=[n]
        if alpha==0:
            alpha=set()
            for p in self.experiment_collection.parameter_list:
                alpha.add(p['alpha'])
            alpha = sorted(list(alpha))
        if methods==0:
            methods=[method for method in self.methods if method!=BASELINE]
        experiments=[]
        for a in alpha:
            for i in range(len(self.experiment_collection.experiments)):
                if self.experiment_collection.parameter_list[i]['n']==n[0] and self.experiment_collection.parameter_list[i]['alpha']==a:
                    experiments.append(self.experiment_collection.experiments[i])

        x_method={method:[e.correct_intersect[method]/e.iterations for e in experiments] for method in methods}
        if contain:
            x_method={method:[e.correct_contain[method]/e.iterations for e in experiments] for method in methods}
        y_method={method:[e.avg_sizes[method] for e in experiments] for method in methods}
        yerr_method={method:[e.ci_sizes[method] for e in experiments] for method in methods}
        fig, ax = plt.subplots(figsize=(19,11))
        cmap0=plt.get_cmap('coolwarm')


        for method in methods:
            if method !='llm (gpt-3.5-turbo)':
                ax.plot(x_method[method],y_method[method],zorder=0,color='grey',linestyle=self.linestyles[method])
            for i in range(len(alpha)):
                label='_nolabel'
                if i==0:
                    label=self.method_mapping[method]
                    if method=='human only' or 'ppr' in method:
                        label = '_nolabel'
                ax.scatter(x_method[method][i], y_method[method][i],
                           marker=self.markers[method],
                           edgecolor='black',
                           vmin=0.6, vmax=1,
                           c=1 - alpha[i],
                           cmap=cmap0, zorder=2, label=label)

        ax.set_xlabel('Baseline intersection probability')
        if contain:
            ax.set_xlabel('Baseline coverage probability')
        ax.set_ylabel('Average rank-set size')
        if not contain:
            ax.set_xscale('log')
        xticks=ax.get_xticks()
        yticks=ax.get_yticks()
        ax.set_xlim(0.0008)
        if contain:
            ax.set_xlim(-0.01,.3)
        ax.set_ylim(bottom=0.9,top=3.2)
        ax.legend(handletextpad=0.02,frameon=False,loc='upper center')

        ax.set_aspect('auto')

        axins = ax.inset_axes([1.12, 0.2, 0.8, 0.8])
        for method in methods:
            axins.plot(x_method[method], y_method[method], zorder=0, color='grey', linestyle=self.linestyles[method])
            if one:
                axins.fill_between(x_method[method],
                               [y_method[method][i] - yerr_method[method][i] for i in range(len(alpha))],
                               [y_method[method][i] + yerr_method[method][i] for i in range(len(alpha))],
                               color='gainsboro',alpha=0.5, edgecolor='lightgrey')

            for i in range(len(alpha)):
                label = '_nolabel'
                if i == 0 and (method=='human only' or 'ppr' in method):
                    label = self.method_mapping[method]

                axins.scatter(x_method[method][i], y_method[method][i],
                       marker=self.markers[method],
                       edgecolor='black',
                       vmin=0.6, vmax=1,
                       c=1 - alpha[i],
                       cmap=cmap0, zorder=2, label=label)

        x1, x2, y1, y2 = 0.49, 0.81, 1.84, 2.4  # n=1000
        if contain:
            x1, x2, y1, y2 = 0.09, 0.25, 1.84, 2.4  # n=1000
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xlabel('Baseline intersection probability')
        if contain:
            axins.set_xlabel('Baseline coverage probability')
        axins.set_aspect('auto')
        ax.indicate_inset_zoom(axins)
        ax.spines[['right', 'top']].set_visible(False)
        axins.spines[['right', 'top']].set_visible(False)
        fig.subplots_adjust(left=0.05, right=0.8)
        axins.legend(handletextpad=0.02,frameon=False,bbox_to_anchor=(-0.05, 1.1),loc='upper left')

        sm = plt.cm.ScalarMappable(cmap=cmap0)
        sm.set_clim(vmin=0.6, vmax=1)
        cbar=plt.colorbar(sm,cax = fig.add_axes([1.52, 0.1, 0.02, 0.8]))
        cbar.ax.set_xlabel(r'$1-\alpha$',labelpad=20)


        if save:
            filepath = self.directory + 'intersect_size_n' + str(n[0]) + '_all.pdf'
            if one:
                filepath = self.directory + 'intersect_size_n' + str(n[0]) +'_'+methods[0]+ '.pdf'
            plt.savefig(filepath, bbox_inches='tight')
            print('Saved figure at ' + filepath)

    # plots figure 7
    def plot_contain_size_setn(self,n,methods=0,alpha=0,save=True, one=False):
        n=[n]
        if alpha==0:
            alpha=set()
            for p in self.experiment_collection.parameter_list:
                alpha.add(p['alpha'])
            alpha = sorted(list(alpha))
        if methods==0:
            methods=[method for method in self.methods if method!=BASELINE]
        experiments=[]
        for a in alpha:
            for i in range(len(self.experiment_collection.experiments)):
                if self.experiment_collection.parameter_list[i]['n']==n[0] and self.experiment_collection.parameter_list[i]['alpha']==a:
                    experiments.append(self.experiment_collection.experiments[i])

        x_method={method:[e.correct_contain[method]/e.iterations for e in experiments] for method in methods}
        y_method={method:[e.avg_sizes[method] for e in experiments] for method in methods}
        yerr_method={method:[e.ci_sizes[method] for e in experiments] for method in methods}
        fig, ax = plt.subplots(figsize=(19,11))
        cmap0=plt.get_cmap('coolwarm')


        for method in methods:
            if method !='llm (gpt-3.5-turbo)':
                ax.plot(x_method[method],y_method[method],zorder=0,color='grey',linestyle=self.linestyles[method])
            for i in range(len(alpha)):
                label='_nolabel'
                if i==0:
                    label=self.method_mapping[method]
                    if method=='human only' or 'ppr' in method:
                        label = '_nolabel'
                ax.scatter(x_method[method][i], y_method[method][i],
                           marker=self.markers[method],
                           edgecolor='black',
                           vmin=0.6, vmax=1,
                           c=1 - alpha[i],
                           cmap=cmap0, zorder=2, label=label)
                if len(methods)<5:
                    ax.plot(x_method[method][i],y_method[method][i],markeredgewidth=0.5,marker=self.markers[method],markeredgecolor='black',markerfacecolor=cmap0((i)/(len(alpha))),color='grey',zorder=2,linestyle=self.linestyles[method])

        ax.set_xlabel('Baseline coverage probability')
        ax.set_ylabel('Average rank-set size')
        xticks=ax.get_xticks()
        yticks=ax.get_yticks()
        ax.set_xlim(-0.005,0.3)
        ax.set_ylim(bottom=0.9,top=3.2)
        ax.legend(handletextpad=0.02,frameon=False,loc='upper center')

        ax.set_aspect('auto')

        axins = ax.inset_axes([1.12, 0.2, 0.8, 0.8])
        for method in methods:
            axins.plot(x_method[method], y_method[method], zorder=0, color='grey', linestyle=self.linestyles[method])
            if one:
                axins.fill_between(x_method[method],
                               [y_method[method][i] - yerr_method[method][i] for i in range(len(alpha))],
                               [y_method[method][i] + yerr_method[method][i] for i in range(len(alpha))],
                               color='gainsboro',alpha=0.5, edgecolor='lightgrey')

            for i in range(len(alpha)):
                label = '_nolabel'
                if i == 0 and (method=='human only' or 'ppr' in method):
                    label = self.method_mapping[method]
                if not one:
                    axins.scatter(x_method[method][i], y_method[method][i],
                           marker=self.markers[method],
                           edgecolor='black',
                           vmin=0.6, vmax=1,
                           c=1 - alpha[i],
                           cmap=cmap0, zorder=2, label=label)
                if one:
                    axins.plot(x_method[method][i], y_method[method][i], marker=self.markers[method],markeredgewidth=0.5, markeredgecolor='black',
                        markerfacecolor=cmap0((i) / (len(alpha))), color='grey', zorder=2, label=label,
                        linestyle=self.linestyles[method])
        x1, x2, y1, y2 = 0.09, 0.25, 1.84, 2.4  # n=1000
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xlabel('Baseline coverage probability')
        axins.set_aspect('auto')
        ax.indicate_inset_zoom(axins)
        ax.spines[['right', 'top']].set_visible(False)
        axins.spines[['right', 'top']].set_visible(False)
        fig.subplots_adjust(left=0.05, right=0.8)
        axins.legend(handletextpad=0.02,frameon=False,bbox_to_anchor=(-0.05, 1.1),loc='upper left')

        sm = plt.cm.ScalarMappable(cmap=cmap0)
        sm.set_clim(vmin=0.6, vmax=1)
        cbar=plt.colorbar(sm,cax = fig.add_axes([1.52, 0.1, 0.02, 0.8]))
        # cbar = fig.colorbar(mappable)
        cbar.ax.set_xlabel(r'$1-\alpha$',labelpad=20)


        if save:
            filepath = self.directory + 'coverage_size_n' + str(n[0]) + '_all.pdf'
            if one:
                filepath = self.directory + 'coverage_size_n' + str(n[0]) +'_'+methods[0]+ '.pdf'
            plt.savefig(filepath, bbox_inches='tight')
            print('Saved figure at ' + filepath)

    # plots figures 2 and 8
    def plot_intersect_size_setmethod_all(self, methods, n=0, alpha=0, contain=False, save=True):
        if alpha == 0:
            alpha = set()
            for p in self.experiment_collection.parameter_list:
                alpha.add(p['alpha'])
            alpha = sorted(list(alpha))
        if n == 0:
            n = set()
            for p in self.experiment_collection.parameter_list:
                n.add(p['n'])
            n = sorted(list(n))
        experiments = {nn: [] for nn in n}
        for nn in n:
            for a in alpha:
                for i in range(len(self.experiment_collection.experiments)):
                    if self.experiment_collection.parameter_list[i]['n'] == nn and \
                            self.experiment_collection.parameter_list[i]['alpha'] == a:
                        experiments[nn].append(self.experiment_collection.experiments[i])

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(37, 8))
        cmap0 = plt.get_cmap('coolwarm')

        for m in range(3):
            method=methods[m]
            x_n = {nn: [e.correct_intersect[method] / e.iterations for e in experiments[nn]] for nn in n}
            if contain:
                x_n = {nn: [e.correct_contain[method] / e.iterations for e in experiments[nn]] for nn in n}
            y_n = {nn: [e.avg_sizes[method] for e in experiments[nn]] for nn in n}
            yerr_n = {nn: [e.ci_sizes[method] for e in experiments[nn]] for nn in n}
            ax=fig.axes[m]
            for nn in n:
                ax.plot(x_n[nn], y_n[nn], zorder=0, color='grey', linestyle=':')
                for i in range(len(alpha)):
                    label = '_nolabel'
                    if i == 0:
                        label = r'$n=$' + str((nn // 66) * 66)
                    ax.errorbar(x_n[nn][i], y_n[nn][i], yerr_n[nn][i], lw=1, capsize=3, color='black', zorder=2)
                    ax.scatter(x_n[nn][i], y_n[nn][i],
                               marker='.',s=800,
                               edgecolor='black',
                               vmin=50, vmax=3000,
                               c=nn,
                               cmap=cmap0, zorder=2, label=label)

            ax.set_xlabel('Baseline intersection probability')
            if contain:
                ax.set_xlabel('Baseline coverage probability')
            if m==0:
                ax.set_ylabel('Average rank-set size')
            ax.set_ylim(0.9, 5.1)
            ax.set_yticks(np.linspace(1, 5, 5))
            ax.set_title(self.method_mapping[method])
            ax.set_aspect('auto')
            ax.spines[['top', 'right']].set_visible(False)


        sm = plt.cm.ScalarMappable(cmap=cmap0)
        sm.set_clim(vmin=0, vmax=3000)
        cbar = plt.colorbar(sm, ax=axs.ravel().tolist())
        cbar.ax.set_xlabel(r'$n$')
        if save:
            filepath = self.directory + 'n_all.pdf'
            if contain:
                filepath = self.directory + 'n_all_coverage.pdf'
            plt.savefig(filepath, bbox_inches='tight')
            print('Saved figure at ' + filepath)


