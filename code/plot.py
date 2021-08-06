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