import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_results(data, save=False, show=False):
    sns.set(rc={'figure.figsize':(20,15)})
    sns.set_style("whitegrid", {'axes.grid' : False})
    df = pd.DataFrame(data=data)
    g = sns.barplot(x='Task', y="Accuracy", hue="Task", data=df, palette='Spectral', dodge=False)
    plt.yticks(np.arange(0,1.4,0.2),fontsize=50)
    plt.xticks([],rotation=0, fontsize=50)
    plt.xlabel('Task', fontsize=60)
    plt.ylabel("Accuracy", fontsize=60)
    plt.legend(fontsize=30)
    plt.tight_layout()
    if save: 
        plt.savefig('baseline_results.png')
    if show: 
        plt.show()

def plot_grouped_lengths(group, save=False, show=False):
    hist = np.histogram([item for task in group for item in group[task]['lengths']], bins=6)
    bin_dict = {(hist[1][i], hist[1][i+1]):{0:0,1:0} for i in range(len(hist[1])-1)}
    for task in group:
        sns.set(rc={'figure.figsize':(20,15), "lines.linewidth": 3})
        sns.set_style("whitegrid", {'axes.grid' : False})
        for l, res in zip(group[task]['lengths'], group[task]['all']):
            for bin in bin_dict:
                if l > bin[0] and l <= bin[1]:
                    bin_dict[bin][res]+=1
        #plot_resultsprint(hist)
        g = sns.barplot(x=[i for i in range(len(bin_dict.keys()))],y=[bin_dict[i][1]/max(bin_dict[i][1]+bin_dict[i][0],1)\
                for i in bin_dict], palette='Spectral')
        xticks = [str(bin)[:-1]+']' for bin in bin_dict.keys()]
        plt.xticks(list(range(len(xticks))),xticks, rotation=0, fontsize=40)
        plt.yticks(np.arange(0,1.25,0.25),rotation=45, fontsize=50)
        plt.xlabel('Lengths', fontsize=60)
        plt.ylabel('Accuracy', fontsize=60)
        plt.title(' '.join(task.upper().split('_')), fontsize=70)
        #plt.legend([],fontsize=50)
        plt.tight_layout()
        if show:
            plt.show()
        if save:
            plt.savefig(task+'_length.png')