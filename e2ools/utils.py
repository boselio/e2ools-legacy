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
from .models import teem
import pickle
from pathlib import Path

def plot_event_times(interactions, r, ax, color='C2'):
    r_interactions = [[t, len([rec for rec in recs if rec == r])] for [t, recs] in interactions]

    x = []
    y = []
    for t, num_times in r_interactions:
        y.append(np.arange(num_times) * 0.005)
        x.append(np.ones(num_times) * t)
    x = np.concatenate(x)
    y = np.concatenate(y)
    ax.plot(x, y, 'o', color=color, markersize=4, alpha=0.5, label='data')
    return ax


def get_cumulative_node_data(interactions, return_total=False):
    interaction_time_by_rec = defaultdict(list)
    all_interaction_times = []
    for t, interaction_recs in interactions:
        for r in interaction_recs:
            interaction_time_by_rec[r].append(t)
            all_interaction_times.append(t)
    final_time = interactions[-1][0]
    cum_dict = {}

    for r, times in interaction_time_by_rec.items():
        x = np.concatenate([[0], np.repeat(times, 2), [final_time]])
        y = np.repeat(np.arange(0, len(times)+1), 2)
        cum_dict[r] = (x, y)

    #all_interaction_times.sort()
    x = np.concatenate([[0], np.repeat(all_interaction_times, 2), [final_time]])
    y = np.repeat(np.arange(0, len(all_interaction_times)+1), 2)

    if return_total:
        return cum_dict, (x, y)
    else:
        return cum_dict


def plot_teem_posterior_plot(mean, upper_limit, lower_limit, change_times, max_time, true_probs=None, plot_events=False, 
                                interactions=None, legend=True, ax=None):
    
    if ax is None:
        fig, ax = plt.subplots()
    posterior_color = sns.color_palette("Paired")[1]
    fill_color = sns.color_palette("Paired")[0]

    x = np.concatenate([np.repeat(change_times, 2)[1:], [max_time]])

    y_ll = np.repeat(upper_limit, 2)
    y_ul = np.repeat(lower_limit, 2)

    ax.fill_between(x, y_ll, y_ul, color=fill_color, alpha=0.5, label='95% Posterior CI')
    ax.plot(x, y_ll, color=posterior_color, linewidth=1.5, linestyle='--')
    ax.plot(x, y_ul, color=posterior_color, linewidth=1.5, linestyle='--')
    ax.set_ylabel('Probability')
    
    y = np.repeat(mean, 2)
    ax.plot(x, y, color=posterior_color, linewidth=1.5, linestyle='-', label='Posterior Mean')
    if legend:
        ax.legend()

    if true_probs is not None:
        ax.plot(true_probs, color='k', linewidth=2, alpha=0.5, label='True Probability')

    if plot_events:
        plot_event_times(interactions, r, ax)


def plot_teem_debug_plots(interactions, means, upper_limits, lower_limits, change_times, true_probs=None, 
                    plot_events=False, true_params=None, gibbs_dir=None, num_chains=None, num_iters_per_chain=None,
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

    num_pages = math.ceil((len(r_list) + 2)/ 10)

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

                if true_probs is not None:
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
                
                
                if plot_events:
                    _ = plot_event_times(interactions, r, ax[i, j])
                    
                ax[i, j].set_title('Receiver {}'.format(r))
                r_counter += 1
                if r_counter == len(r_list):
                    break

            if p == num_pages-1:
                if true_params is not None:
                    page_counter = r_counter % 10
                    if 'alpha' in true_params:
                        i, j = np.unravel_index(page_counter, [5, 2])
                        alphas = teem.get_posterior_alphas(gibbs_dir, num_chains, num_iters_per_chain)
                        ax[i,j].plot(alphas, color=posterior_color, label='Posterior estimates')
                        x = [0, len(alphas)]
                        y = [true_params['alpha'], true_params['alpha']]

                        ax[i, j].plot(x, y, color='k', label='True alpha = {}'.format(true_params['alpha']))
                        ax[i, j].legend()
                        ax[i, j].set_title('Alpha Trace Plot')
                        page_counter += 1
                        
                    if 'theta' in true_params:
                        i, j = np.unravel_index(page_counter, [5, 2])
                        thetas = teem.get_posterior_thetas(gibbs_dir, num_chains, num_iters_per_chain)
                        ax[i,j].plot(thetas, color=posterior_color, label='Posterior estimates')
                        x = [0, len(thetas)]
                        y = [true_params['theta'], true_params['theta']]

                        ax[i, j].plot(x, y, color='k', label='True theta = {}'.format(true_params['theta']))
                        ax[i, j].legend()
                        ax[i, j].set_title('Theta Trace Plot')

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


def create_posterior_predictive_dataset(temporal_probs, interaction_times, num_recs_per_interaction):

    np.random.seed()
    num_recs = len(temporal_probs.created_times)

    stick_list = []
    for r in range(num_recs):        
        sticks_ind = np.digitize(interaction_times, temporal_probs.arrival_times_dict[r], right=False) - 1

        #sticks_ind[sticks_ind == len(tp.stick_dict[r])] = len(tp.stick_dict[r]) - 1
        sticks = np.array(temporal_probs.stick_dict[r])[sticks_ind]
        sticks[interaction_times < temporal_probs.created_times[r]] = 1
        stick_list.append(sticks)

    stick_array = np.array(stick_list)
    probs_array = np.vstack([stick_array, np.ones(stick_array.shape[1])])
    #probs_array = stick_array.copy()
    probs_array[1:, :] = probs_array[1:, :] * np.cumprod(1 - probs_array[:-1, :], axis=0)

    probs_sum = probs_array.sum(axis=0)
    probs_array = probs_array / probs_sum
    pp_interactions = []

    for i, (t, n) in enumerate(zip(interaction_times, num_recs_per_interaction)):
        interaction_recs = np.random.choice(a=num_recs+1, size=n, p=probs_array[:, i])

        pp_interactions.append([t, interaction_recs.tolist()])
    
    return pp_interactions


def create_pp_datasets(save_dir, interaction_times, num_recs, num_chains, num_iters_per_chain, pp_save_dir=None):

    chain_dirs = [save_dir / '{}'.format(c) for c in range(num_chains)]
    save_paths = [cd / '{}.pkl'.format(i) for cd in chain_dirs for i in range(num_iters_per_chain)]

    tp_list = []
    for sd in save_paths:
        with sd.open('rb') as infile:
            tp, params = pickle.load(infile)
        tp_list.append(tp)

    if pp_save_dir is None:
        pp_save_dir = save_dir / 'pp_data'

    pp_chain_dirs = [pp_save_dir / '{}'.format(c) for c in range(num_chains)] 

    for pp_chain_dir in pp_chain_dirs:
        if not pp_chain_dir.is_dir():
            pp_chain_dir.mkdir(parents=True)

    pp_save_paths = [pp_cd / '{}.dat'.format(i) for pp_cd in pp_chain_dirs for i in range(num_iters_per_chain)]

    pp_func = partial(create_posterior_predictive_dataset, interaction_times=interaction_times,
                                        num_recs_per_interaction=num_recs)

    with ProcessPoolExecutor() as executor:
        for pp_save_dir, pp_interactions in zip(pp_save_paths, executor.map(pp_func, tp_list)):
            save_interactions(pp_interactions, pp_save_dir)



def save_interactions(interactions, file_path):

    file_path = file_path
    with file_path.open('w') as outfile:
        for interaction in interactions:
            outline = '{} '.format(interaction[0])
            outline += ' '.join([str(i) for i in interaction[1]])
            outline += '\n'
            outfile.write(outline)

def load_interactions(file_name):

    interactions = []
    with open(file_name, 'r') as infile:
        for interaction_str in infile:
            temp_list = interaction_str.split()
            interactions.append([float(temp_list[0]), [int(i) for i in temp_list[1:]]])

    return interactions