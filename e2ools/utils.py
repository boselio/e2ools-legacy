import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from concurrent.futures import as_completed
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from copy import deepcopy
from scipy.special import logit
import matplotlib.backends.backend_pdf
import math


def plot_event_times(interactions, r, ax):
    r_interactions = [[t, len([rec for rec in recs if rec == r])] for [t, recs] in interactions]

    x = []
    y = []
    for t, num_times in r_interactions:
        y.append(np.arange(num_times) * 0.02)
        x.append(np.ones(num_times) * t)
    x = np.concatenate(x)
    y = np.concatenate(y)
    ax.plot(x, y, 'o', color='C2', markersize=4, alpha=0.5, label='data')
    return ax


def plot_teem_debug_plots(interactions, true_probs, means, upper_limits, lower_limits, change_times,
                    save_dir='debug.pdf', r_list='all'):

    sns.set()
    sns.set_context('paper')

    max_time = interactions[-1][0]
    unique_nodes, degrees = np.unique([i for interaction in interactions for i in interaction[1]], return_counts=True)
    if r_list == 'all':
        r_list = unique_nodes
    
    elif r_list[:3] == 'top':
        number_of_plots = int(r_list[3:])
        r_list = unique_nodes[np.argsort(degrees)[::-1][:number_of_plots]]

    num_pages = math.ceil(len(r_list) / 10)

    r_counter = 0

    posterior_color = sns.color_palette("Paired")[1]
    fill_color = sns.color_palette("Paired")[0]

    with matplotlib.backends.backend_pdf.PdfPages(save_dir) as pdf:
        for p in range(num_pages):
            fig, ax = plt.subplots(5,2, figsize=(8.5, 11))
            for k in range(10):
                r = r_list[r_counter]
                i, j = np.unravel_index(k, [5, 2])
                if i == 0 and j == 1:
                    #use these for the legend
                    true_label = 'Oracle'
                    confidence_label = '95% Posterior CI'
                    mean_label = 'Posterior Mean'
                else:
                    true_label = None
                    confidence_label = None
                    mean_label = None

                ax[i, j].plot(*true_probs[r], color='k', linewidth=2, alpha=0.5, label=true_label)
                
                x = np.concatenate([np.repeat(change_times, 2)[1:], [max_time]])

                y_ll = np.repeat(upper_limits[:, r], 2)
                y_ul = np.repeat(lower_limits[:, r], 2)

                ax[i, j].fill_between(x, y_ll, y_ul, color=fill_color, alpha=0.5, label=confidence_label)
                ax[i, j].plot(x, y_ll, color=posterior_color, linewidth=1.5, linestyle='--')
                ax[i, j].plot(x, y_ul, color=posterior_color, linewidth=1.5, linestyle='--')
                

                
                y = np.repeat(means[:, r], 2)

                ax[i, j].plot(x, y, color=posterior_color, linewidth=1.5, linestyle='-', label=mean_label)
                if i == 0 and j == 1:
                    ax[i, j].legend()
                
                
                _ = plot_event_times(interactions, r, ax[i, j])
                ax[i, j].set_title('Receiver {}'.format(r))
                r_counter += 1
                if r_counter == len(r_list):
                    break

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return


def plot_ddcrp_debug_plots(interactions, true_probs, estimated_probs, estimated_times,
                    save_dir='debug.pdf', r_list='all'):

    sns.set()
    sns.set_context('paper')

    max_time = interactions[-1][0]
    unique_nodes, degrees = np.unique([i for interaction in interactions for i in interaction[1]], return_counts=True)
    if r_list == 'all':
        r_list = unique_nodes
    
    elif r_list[:3] == 'top':
        number_of_plots = int(r_list[3:])
        r_list = unique_nodes[np.argsort(degrees)[::-1][:number_of_plots]]

    num_pages = math.ceil(len(r_list) / 10)

    r_counter = 0

    posterior_color = sns.color_palette("Paired")[1]
    fill_color = sns.color_palette("Paired")[0]

    with matplotlib.backends.backend_pdf.PdfPages(save_dir) as pdf:
        for p in range(num_pages):
            fig, ax = plt.subplots(5,2, figsize=(8.5, 11))
            for k in range(10):
                r = r_list[r_counter]
                i, j = np.unravel_index(k, [5, 2])
                if i == 0 and j == 1:
                    #use these for the legend
                    true_label = 'Oracle'
                    prob_label = 'DDCRP Probs.'
                else:
                    true_label = None
                    prob_label = None

                ax[i, j].plot(*true_probs[r], color='k', linewidth=2, alpha=0.5, label=true_label)
                
                #x = np.concatenate([np.repeat(change_times, 2)[1:], [max_time]])

                #y_ll = np.repeat(upper_limits[:, r], 2)
                #y_ul = np.repeat(lower_limits[:, r], 2)

                #ax[i, j].fill_between(x, y_ll, y_ul, color=fill_color, alpha=0.5)
                #ax[i, j].plot(x, y_ll, color=posterior_color, linewidth=1.5, linestyle='--')
                #ax[i, j].plot(x, y_ul, color=posterior_color, linewidth=1.5, linestyle='--')
                
                #y = np.repeat(means[:, r], 2)

                ax[i, j].plot(estimated_times, estimated_probs[:, r], color=posterior_color, linewidth=1.5, linestyle='--', label=prob_label)
                if i == 0 and j == 1:
                    ax[i, j].legend()
                
                
                _ = plot_event_times(interactions, r, ax[i, j])
                ax[i, j].set_title('Receiver {}'.format(r))
                r_counter += 1
                if r_counter == len(r_list):
                    break

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return