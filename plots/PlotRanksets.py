from Result import Result
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import numpy as np


class PlotRanksets:
    def __init__(self,results,mapping):
        self.results=results
        self.parameters=results.parameters
        self.method_mapping=mapping
        self.methods=results.methods
        self.models=results.models
        self.directory='plots/ranksets/'

    def set_model_mapping(self,model_mapping):
        self.model_mapping=model_mapping
        self.model_labels=[model_mapping[model] for model in self.models]

    # plots figures 3 and 9
    def plot_ranks_dots6(self, methods, save=True):
        data = [[self.results.ranks[method][model] for model in self.models] for method in methods]

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(37,9))
        # fig.set_constrained_layout_pads(hspace=0, wspace=2)
        if len(methods)==8:
            fig, axs = plt.subplots(2, 4, sharey=True, figsize=(37,18))

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

