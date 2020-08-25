import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import defaultdict
import pickle

matplotlib.use('Agg')  # for plotting without display

font = {'family': 'normal',
        'weight': 'normal',
        'size': 26}

matplotlib.rc('font', **font)
matplotlib.rc('image')

baseline_dict = {'Berlin': 582.2998761858258,
                 'Istanbul': 789.232608468192,
                 'Moscow': 1031.9532104492187
                 }


def inverse_result_dict(resultdict):
    """Orders the given dictionary by city instead of by model"""

    inv_resultdict = defaultdict(dict)

    for model_name, innerdict in resultdict.items():

        for key, value in innerdict.items():
            if key == 'nb_params':
                continue
            inv_resultdict[key][model_name] = [innerdict['nb_params'], value]

    return inv_resultdict


def plot_performance_nbparams(ax, nb_params_dict, norm=False):
    # create plot

    for key, value in nb_params_dict.items():
        if len(value) < 2:
            continue

        if 'MIE-Lab' in key:
            continue

        if 'Graph' in key:
            marker = 'x'

        if 'KipfNet' in key:
            marker = '^'
        elif 'Graph' in key:
            marker = 'v'
        elif 'Skipf' in key:
            marker = '*'
        else:
            marker = 'P'

        if norm:
            ax.scatter(np.log10(value[0]), value[1] / baseline_dict[city], label=key, marker=marker, s=300, cmap='jet')
        else:
            ax.scatter(np.log10(value[0]), value[1], label=key, marker=marker, s=300, cmap='jet')

    return ax


if __name__ == '__main__':
    path_dict = {}
    nb_params_path_dict = {}

    # transform_to_csv(inpath='../runs/PMLR_nets', outpath='../runs/PMLR_nets_csv')

    with  open(os.path.join('.', 'output', 'data_generalization.p'), "rb") as input_file:
        resultdict = pickle.load(input_file)

    inv_resultdict = inverse_result_dict(resultdict)

    #
    nb_params_path_dict['PMLR_nets'] = os.path.join('runs', 'PMLR_nets')

    fig, ax = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    for ix, city in enumerate(['Moscow', 'Berlin', 'Istanbul']):
        ax[ix] = plot_performance_nbparams(ax[ix], inv_resultdict[city], norm=False)
        ax[ix].set_title(city)

        ax[ix].axhline(y=baseline_dict[city], xmin=0, xmax=2, linewidth=3, linestyle='--', color='k')
        ax[ix].axhline(y=baseline_dict[city] * 1.05, xmin=0, xmax=2, linewidth=3, linestyle=':', color='w')

    ax = ax[-1]

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    plt.xlabel('Number of parameters (log)', fontdict=font)
    fig.text(-0.01, 0.5, 'Mean Squared Error', va='center', rotation='vertical')
    #    plt.ylabel('MSE', fontdict=font)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=3, bbox_to_anchor=(0.1, -0.01), ncol=5, frameon=False,
               fontsize=20)
    figbox = matplotlib.transforms.Bbox([[-0.4, -0.75], [20, 11.8]])
    plt.tight_layout()
    plt.savefig(os.path.join('.', 'output', 'performance_nb_params.pdf'), bbox_inches=figbox)
    plt.close()
