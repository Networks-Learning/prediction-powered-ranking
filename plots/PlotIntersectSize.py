import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib
import matplotlib.cm as cm
class PlotIntersectSize:
    def __init__(self,experiment_collection,mapping):
        self.experiment_collection=experiment_collection
        self.parameter_list=experiment_collection.parameter_list
        self.method_mapping=mapping
        self.methods=experiment_collection.experiments[0].methods
        self.models=experiment_collection.experiments[0].models
        self.directory='plots/intersect_size/'

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
            methods=[method for method in self.methods if method!='baseline']
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
            methods=[method for method in self.methods if method!='baseline']
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
