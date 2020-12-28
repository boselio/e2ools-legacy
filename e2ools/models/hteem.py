import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
import numpy as np
import pdb
import scipy.optimize as opt
import pandas as pd
from collections import defaultdict
import random
import time
from functools import partial
import scipy
import scipy.sparse as sps
import scipy.stats as stats
import pickle
import os.path
from copy import deepcopy
from .choice_fns import choice_discrete_unnormalized
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from .generators import SharedHollywoodNetwork, NewmanClausetPower
from concurrent.futures import ProcessPoolExecutor


class HVCM():
    def __init__(self, holdout=100, alpha_hyperparams=(5, 5),
                 theta_hyperparams=(10, 10), lower_alpha_hyperparams=(5,5),
                 lower_theta_hyperparams=(10,10), hier_priors=False):

        if hier_priors:
            raise NotImplementedError('Hierarchical priors are broken. Please come again.')
        self.alpha_hyperparams = alpha_hyperparams
        self.theta_hyperparams = theta_hyperparams
        self.lower_alpha_hyperparams = lower_alpha_hyperparams
        self.lower_theta_hyperparams = lower_theta_hyperparams


    def initialize_state(self, interactions, save_dir):

        #Initialize state variables
        #Number of customers in each restaurant
        self.num_customers_in_s = defaultdict(int)

        ####Don't think I need this
        #Number of total tables across franchise serving each dish
        self.global_table_counts = defaultdict(int)
        ####

        #Number of total tables in the franchise
        self.global_num_tables = 0
        #Number of tables in each restaurant
        self.num_tables_in_s = defaultdict(int)
        #configuration: top level key is s (the restaurant),
        #lower level key is r (type of dish),
        #lowest level value is list of tables with # of customers.
        self.receiver_inds = defaultdict(lambda: np.array([-1]))
        self.table_counts = defaultdict(list)
        self.sticks = defaultdict(lambda: np.array())
        self.probs = defaultdict(lambda: [1])
        self.num_type_tables = 0

        self.store_alphas = []
        self.store_thetas = []
        #Initialize alpha and theta.
        self.alpha = np.random.beta(*self.alpha_hyperparams)
        self.theta = np.random.gamma(*self.theta_hyperparams)

        #Sender set
        self.s_set = set([interaction[1] for interaction in interactions])
        self.s_size = len(self.s_set)

        #Reciever set
        self.r_set = set([r for interaction in interactions for r in interaction[2]])
        self.r_size = len(self.r_set)

        if len(self.lower_theta_hyperparams) != self.s_size:
            self.lower_theta_hyperparams = dict(zip(self.s_set, [self.lower_theta_hyperparams] * self.s_size))

        if len(self.lower_alpha_hyperparams) != self.s_size:
            self.lower_alpha_hyperparams = dict(zip(self.s_set, [self.lower_alpha_hyperparams] * self.s_size))

        self.alpha_s = {}
        self.theta_s = {}
        for s in self.s_set:
            self.alpha_s[s] = np.random.beta(*self.lower_alpha_hyperparams[s])
            self.theta_s[s] = np.random.gamma(*self.lower_theta_hyperparams[s])

        #Priors for alphas and thetas
        #alphas = np.random.beta(self.phi_alpha * self.alpha, self.phi_alpha * (1 - self.alpha), size=len(self.s_set))
        #thetas = np.random.gamma(1/self.phi_theta * self.theta, self.phi_theta, size=len(self.s_set))

        #self.alpha_s = dict(zip(self.s_set, alphas))
        #self.theta_s = dict(zip(self.s_set, thetas))

        self.running_mean_theta_s = defaultdict(float)
        self.running_mean_alpha_s = defaultdict(float)


        self._sample_table_configuration(interactions, initial=True)

        with open(os.path.join(save_dir, 'model_info.pkl'), 'wb') as outfile:
            pickle.dump({'alpha_hyperparams': self.alpha_hyperparams,
                            'theta_hyperparams': self.theta_hyperparams,
                            'theta_s_hyperparams': self.lower_theta_hyperparams,
                            'alpha_s_hyperparams': self.lower_alpha_hyperparams},
                        outfile)

        file_name = os.path.join(save_dir, 'params_{}.pkl'.format(0))
        with open(file_name, 'wb') as outfile:

            pickle.dump({'global_counts': self.global_table_counts,
                            'theta': self.theta,
                            'alpha': self.alpha,
                            'alpha_s': self.alpha_s,
                            'theta_s': self.theta_s
                            },
                        outfile)

        self.store_alphas.append(self.alpha)
        self.store_thetas.append(self.theta)

        for s in self.s_set:
            self.running_mean_theta_s[s] += self.theta_s[s]
            self.running_mean_alpha_s[s] += self.alpha_s[s]


    def fit(self, interactions, initialize=True,
            iters=100, phi_theta=1, phi_alpha=2, save_dir='./'):

        if initialize:
            self.initialize_state(interactions, save_dir)

        for iteration in range(1, iters):
            s_time = time.time()

            print(iteration)
            random.shuffle(interactions)
            self._sample_parameters()
            self._sample_table_configuration(interactions)

            file_name = os.path.join(save_dir, 'params_{}.pkl'.format(iteration))
            with open(file_name, 'wb') as outfile:

                pickle.dump({'global_counts': self.global_table_counts,
                             'theta': self.theta,
                             'alpha': self.alpha,
                             'alpha_s': self.alpha_s,
                             'theta_s': self.theta_s
                             },
                            outfile)

            self.store_alphas.append(self.alpha)
            self.store_thetas.append(self.theta)

            for s in self.s_set:
                self.running_mean_theta_s[s] += self.theta_s[s]
                self.running_mean_alpha_s[s] += self.alpha_s[s]

            e_time = time.time()
            print(e_time - s_time)


    def _sample_table_configuration(self, interactions, initial=False):
        if initial:
            for s, r_list in interactions:
                for r in r_list:
                    self._add_customer(s, r)

        else:
            for s, interaction in interactions:
                for r in interaction:
                    #Remove a customer
                    self._remove_customer(s, r)

                    #Add a customer
                    self._add_customer(s, r)


    def _add_customer(self, s, r, cython_flag=True):
        if self.global_table_counts[r] == 0:
            self.global_table_counts[r] += 1
            self.receiver_inds[s][r].insert(-1, self.num_tables_in_s[s])
            self.global_num_tables += 1
            self.num_customers_in_s[s] += 1
            self.num_tables_in_s[s] += 1
            self.num_type_tables += 1

            self.table_counts[s][self.receiver_inds[s][r][-2]] += 1
            #Draw local stick
            self.sticks[s].append(np.random.beta(1, self.theta_s[s]))
            self.probs[s] = np.concatenate([self.sticks[s], [1]])
            self.probs[s][1:] = self.probs[s][1:] * np.cumprod(1 - self.probs[s][:-1])

            #Draw global stick
            self.global_sticks.append(np.random.beta(1 - self.alpha, 
                            self.theta + len(self.global_sticks) * self.alpha))
            self.global_probs = np.concatenate([self.global_sticks, [1]])
            self.global_probs[1:] = self.global_probs[1:] * np.cumprod(1 - self.global_probs[:-1])
            return

        probs = self.probs[s][self.receiver_inds[s][r]] 
        
        table = choice_discrete_unnormalized(probs, np.random.rand())

        if table == len(probs):
            self.receiver_inds[s][r].insert(-1, self.num_tables_in_s)

            #Draw stick
            self.sticks[s].append(np.random.beta(1, self.theta_s[s]))
            self.probs[s] = np.concatenate([self.sticks[s], [1]])
            self.probs[s][1:] = self.probs[s][1:] * np.cumprod(1 - self.probs[s][:-1])

            self.num_tables_in_s[s] += 1
            self.global_table_counts[r] += 1
            self.global_num_tables += 1

        self.table_counts[s][self.receiver_inds[s][r][table]] += 1
        self.num_customers_in_s[s] += 1


    def _remove_customer(self, s, r, cython_flag=True):
        #Choose uniformly at random a customer to remove.
        self.num_customers_in_s[s] -= 1
        remove_probs = self.table_counts[s][self.receiver_inds[s][r][:-1]]

        table = choice_discrete_unnormalized(remove_probs, np.random.rand())
        
        ind = self.receiver_inds[s][r][table]
        self.table_counts[s][ind] -= 1
        if self.table_counts[s][ind] == 0:
            self.num_tables_in_s[s] -= 1
            self.global_table_counts[r] -= 1
            self.global_num_tables -= 1
            del self.sticks[s][ind]
            self.probs[s] = np.concatenate([self.sticks[s], [1]])
            self.probs[s][1:] = self.probs[s][1:] * np.cumprod(1 - self.probs[s][:-1])

            del self.receiver_inds[s][r][table]


    def _sample_parameters(self):

        #Sample lower sticks
        #Sample higher sticks
        #Sample all the sticks
        x = np.random.beta(self.theta + 1, self.global_num_tables - 1)
        y = np.random.rand(self.r_size - 1) < (self.theta / (self.theta + self.alpha * np.arange(1, self.r_size)))
        temp = 0
        for r in self.r_set:
            #pdb.set_trace()
            if self.global_table_counts[r] <= 1:
                continue
            z_r = np.random.rand(self.global_table_counts[r]-1) < ((np.arange(1, self.global_table_counts[r]) - 1) / (np.arange(1, self.global_table_counts[r]) - self.alpha))
            temp += np.sum(1 - z_r)

        self.theta = np.random.gamma(np.sum(y) + self.theta_hyperparams[0], 1/(1/self.theta_hyperparams[1] - np.log(x)))
        self.alpha = np.random.beta(self.alpha_hyperparams[0] + np.sum(1 - y), self.alpha_hyperparams[1] + temp)

        for s in self.s_set:
            if self.num_customers_in_s[s] <= 1:
                self.theta_s[s] = np.random.gamma(*self.lower_theta_hyperparams[s])
                self.alpha_s[s] = np.random.beta(*self.lower_alpha_hyperparams[s])
                continue

            x_s = np.random.beta(self.theta_s[s] + 1, self.num_customers_in_s[s] - 1)
            y_s = np.random.rand(self.num_tables_in_s[s] - 1) < (self.theta_s[s] / (self.theta_s[s] + self.alpha_s[s] * np.arange(1, self.num_tables_in_s[s])))

            #For every table in the restaurant s...
            temp_s = 0
            for r, r_list in self.configuration[s].items():
                for table_count in r_list:
                    if table_count <= 1:
                        continue

                    temp_s += np.sum(1 - (np.random.rand(table_count-1) <
                            ((np.arange(1, table_count) - 1) / (np.arange(1, table_count) - self.alpha_s[s]))))

            self.theta_s[s] = np.random.gamma(np.sum(y_s) + self.lower_theta_hyperparams[s][0], 1/(1/self.lower_theta_hyperparams[s][1] - np.log(x_s)))
            self.alpha_s[s] = np.random.beta(self.lower_alpha_hyperparams[s][0] + np.sum(1 - y_s), self.lower_alpha_hyperparams[s][1] + temp_s)


    def save(self, fname):
        model = [self.store_thetas, self.store_alphas,
                 self.running_mean_theta_s, self.running_mean_alpha_s]
        with open(fname, 'wb') as outfile:
            pickle.dump(model, outfile)


    def load_model(self, fname):
        with open(fname, 'rb') as model_infile:
            model_dict = pickle.load(model_infile)


    def load_params(self, fname):
        with open(fname, 'rb') as param_infile:
            param_dict = pickle.load(param_infile)
        self.global_table_counts = param_dict['global_counts']
        self.theta = param_dict['theta']
        self.alpha = param_dict['alpha']
        self.alpha_s = param_dict['alpha_s']
        self.theta_s = param_dict['theta_s']


    def write_report(self, param_files, save_dir='./', cutoff=200,
                     local_dists=None, create_ppcs=False, print_priors=True,
                     context=None, ppc_dir=None):
        #if params_range is None:
        #param_files = [os.path.join(params_dir, i) for i in os.listdir(params_dir) if 'params' in i]
        #else:
        #    param_files = [os.path.join(params_dir, i) for i in range(*params_range)]
        #fig, ax = plt.subplots()

        theta_s_list = []
        alpha_s_list = []
        alpha_list = []
        theta_list = []
        for pf in param_files:
            print(pf)
            with open(pf, 'rb') as infile:
                param_dict = pickle.load(infile)

            theta_list.append(param_dict['theta'])
            alpha_list.append(param_dict['alpha'])
            alpha_s_list.append(param_dict['alpha_s'])
            theta_s_list.append(param_dict['theta_s'])

        #plot thetas and alphas on same figure
        global_fig, global_axs = self.create_trace_and_plots([theta_list, alpha_list], ['theta', 'alpha'],
                                               cutoff, [self.theta_hyperparams, self.alpha_hyperparams],
                                                             ['gamma', 'beta'], context=context,
                                                             print_priors=print_priors)

        global_fig.savefig(os.path.join(save_dir, 'global_trace.pdf'), type='pdf', dpi=2000)

        if local_dists is not None:
            theta_s_traces = defaultdict(list)
            alpha_s_traces = defaultdict(list)

            for a_s in alpha_s_list:
                for s, alph in a_s.items():
                    alpha_s_traces[s].append(alph)

            for t_s in theta_s_list:
                for s, thet in t_s.items():
                    theta_s_traces[s].append(thet)

            local_traces = []
            local_names = []
            local_prior_params = []
            local_prior_types = []

            for i in local_dists:
                local_traces.append(theta_s_traces[i])
                local_traces.append(alpha_s_traces[i])
                local_names.append('theta_{}'.format(i))
                local_names.append('alpha_{}'.format(i))
                local_prior_params.append(self.lower_theta_hyperparams[i])
                local_prior_params.append(self.lower_alpha_hyperparams[i])
                local_prior_types.append('gamma')
                local_prior_types.append('beta')

            #theta_traces = [theta_s_traces[i] for i in local_dists]
            #theta_names = ['theta_{}'.format(i) for i in local_dists]
            #theta_prior_params = [self.lower_theta_hyperparams[i] for i in local_dists]
            #theta_prior_types = ['gamma' for i in local_dists]
            #alpha_traces = [alpha_s_traces[i] for i in local_dists]
            #alpha_names = ['alpha_{}'.format(i) for i in local_dists]
            #alpha_prior_params = [self.lower_alpha_hyperparams[i] for i in local_dists]
            #alpha_prior_types = ['beta' for i in local_dists]


            local_figs, local_axs = self.create_trace_and_plots(local_traces, local_names, cutoff,
                                                                local_prior_params, local_prior_types)
            local_figs.savefig(os.path.join(save_dir, 'local_traces.pdf'), type='pdf', dpi=1000)
        else:
            local_axs = None
        return (global_axs, local_axs)

    def create_trace_and_plots(self, param_lists, param_names, cutoff,
                               prior_params_list, prior_type_list, print_priors,
                               context, axs=None):
        #print traces
        num_params = len(param_lists)
        if context is not None:
            sns.set_context(context)
        fig, axs = plt.subplots(num_params, 2, figsize=(8, 2*num_params))
        for ind, (params, name, prior_params, prior_type) in enumerate(zip(param_lists, param_names,
                                                                   prior_params_list, prior_type_list)):
            axs[ind,0].plot(params, label=name)
            axs[ind,0].plot([cutoff, cutoff], [min(params), max(params)], label='cutoff')
            sns.distplot(params[cutoff:], ax=axs[ind,1], norm_hist=True, kde=False, label='posterior')
            ylims = axs[ind,1].get_ylim()
            param_mean = np.mean(params[cutoff:])
            axs[ind,1].plot([param_mean, param_mean], [*ylims], label='mean={:.2f}'.format(param_mean))

            if print_priors:
                if prior_type == 'gamma':
                    prior_fn = stats.gamma
                    prior_params = {'a': prior_params[0],
                                    'scale': prior_params[1]}
                elif prior_type == 'beta':
                    prior_fn = stats.beta
                    prior_params = {'a': prior_params[0],
                                    'b': prior_params[1]}
                else:
                    raise valueerror('prior type not supported.')

                x = np.linspace(prior_fn.ppf(0.001, **prior_params),
                                prior_fn.ppf(0.999, **prior_params), 100)
                axs[ind,1].plot(x, prior_fn.pdf(x, **prior_params), lw=2, label='prior')
            axs[ind,0].legend()
            axs[ind,1].legend()
            axs[ind,0].set_title(name)
            axs[ind,1].set_title(name)
        fig.tight_layout()
        return fig, axs


    def create_ppc_data(self, save_dir, param_files, save_files,
                        subject_list, num_authors, parallel=False, local_ppc=None):

        if local_ppc is None:
            local_ppc = list(set(subject_list))

        ppc_create = partial(self._create_ppc_file, subject_list=subject_list.copy(),
                                         num_authors=num_authors.copy())
        ppc_results = defaultdict(list)
        #with ProcessPoolExecutor() as executor:
        #    for ind, X in enumerate(executor.map(ppc_create,
        #                                        param_files, save_files)):

        #stat_fns = [self._num_global_recievers]
        local_ppc = partial(self._num_local_recievers, inds=local_ppc)
        stat_fns = [self._num_global_recievers, local_ppc]
        senders = list(set(subject_list))

        #local_fns = [partial(self._num_local_reciever, ind=i) for i in senders]
        #stat_fns += local_fns
        ppc_names = ['global_recievers', 'local_recievers']
        #ppc_names += ['local_recievers_{}' for i in senders]

        ppc_dict = defaultdict(list)
        subject_list = np.array(subject_list)
        num_authors = np.array(num_authors)
        if parallel:
            with ProcessPoolExecutor() as executor:
                for ind, X in enumerate(executor.map(ppc_create, param_files, save_files)):
                    ppc_results = self._get_stat(stat_fns, X)
                    for pn, pr in zip(ppc_names, ppc_results):
                        if pn == 'local_recievers':
                            for k, v in pr.items():
                                ppc_dict[pn + str(k)].append(v)
                        else:
                            ppc_dict[pn].append(pr)
                    ppc_dict['index'].append(ind)
        else:
            for ind, (pf, sf) in enumerate(zip(param_files, save_files)):
                order = np.random.permutation(len(subject_list))
                X = self._create_ppc_file(pf, sf, subject_list=subject_list[order], num_authors=num_authors[order])
                ppc_results = self._get_stat(stat_fns, X)
                for pn, pr in zip(ppc_names, ppc_results):
                    if pn == 'local_recievers':
                        for k, v in pr.items():
                            ppc_dict[pn + str(k)].append(v)
                    else:
                        ppc_dict[pn].append(pr)
                ppc_dict['index'].append(ind)

        #import pdb
        #pdb.set_trace()

        ppc_df = pd.DataFrame(ppc_dict)
        ppc_df.to_csv('./ppc_results.csv')
        return ppc_df


    def write_ppc_report(self, ppc_df, params_dir, params_range, real_val, save_dir='./', local_dists=None):
        param_files = [os.path.join(params_dir, 'params_{}.pkl'.format(i)) for i in range(*params_range)]
        #fig, ax = plt.subplots()

        theta_s_list = []
        alpha_s_list = []
        alpha_list = []
        theta_list = []
        for pf in param_files:
            print(pf)
            with open(pf, 'rb') as infile:
                param_dict = pickle.load(infile)

            theta_list.append(param_dict['theta'])
            alpha_list.append(param_dict['alpha'])
            alpha_s_list.append(param_dict['alpha_s'])
            theta_s_list.append(param_dict['theta_s'])

        #plot thetas and alphas on same figure
        import pdb
        pdb.set_trace()
        global_fig, global_axs = self.create_dists_and_ppc(ppc_df, real_val=real_val, upper_level_labels=['global'],
                                                           theta_lists=[theta_list],
                                                           alpha_lists=[alpha_list],
                                                           theta_params_list=[self.theta_hyperparams],
                                                           alpha_params_list=[self.alpha_hyperparams])

        global_fig.savefig(os.path.join(save_dir, 'global_ppc.pdf'), type='pdf', dpi=2000)

        if local_dists is not None:
            theta_s_traces = defaultdict(list)
            alpha_s_traces = defaultdict(list)

            for a_s in alpha_s_list:
                for s, alph in a_s.items():
                    alpha_s_traces[s].append(alph)

            for t_s in theta_s_list:
                for s, thet in t_s.items():
                    theta_s_traces[s].append(thet)

            local_alpha_traces = []
            local_theta_traces = []
            local_alpha_params = []
            local_theta_params = []

            for i in local_dists:
                local_theta_traces.append(theta_s_traces[i])
                local_alpha_traces.append(alpha_s_traces[i])
                local_theta_params.append(self.lower_theta_hyperparams[i])
                local_alpha_params.append(self.lower_alpha_hyperparams[i])

            #theta_traces = [theta_s_traces[i] for i in local_dists]
            #theta_names = ['theta_{}'.format(i) for i in local_dists]
            #theta_prior_params = [self.lower_theta_hyperparams[i] for i in local_dists]
            #theta_prior_types = ['gamma' for i in local_dists]
            #alpha_traces = [alpha_s_traces[i] for i in local_dists]
            #alpha_names = ['alpha_{}'.format(i) for i in local_dists]
            #alpha_prior_params = [self.lower_alpha_hyperparams[i] for i in local_dists]
            #alpha_prior_types = ['beta' for i in local_dists]


            local_figs, local_axs = self.create_dists_and_ppc(ppc_df, real_val=real_val, upper_level_labels=local_dists,
                                                              theta_lists=local_theta_traces,
                                                              alpha_lists=local_alpha_traces,
                                                              theta_params_list=local_theta_params,
                                                              alpha_params_list=local_alpha_params,
                                                              global_params=False)
            local_figs.savefig(os.path.join(save_dir, 'local_ppc.pdf'), type='pdf', dpi=1000)


    def create_dists_and_ppc(self, ppc_df, upper_level_labels, theta_lists, alpha_lists,
                             theta_params_list, alpha_params_list, real_val, axs=None, global_params=True):
        #print traces and ppc
        num_rows = len(theta_params_list)

        fig, axs = plt.subplots(num_rows, 3, figsize=(10,2*num_rows))
        if num_rows == 1:
            axs = axs[np.newaxis,:]
        for ind, (thetas, alphas, name, theta_params, alpha_params) in enumerate(zip(theta_lists, alpha_lists, upper_level_labels,theta_params_list, alpha_params_list)):
            sns.distplot(thetas, ax=axs[ind,0], norm_hist=True, label='posterior')
            ylims = axs[ind,0].get_ylim()
            param_mean = np.mean(thetas)
            axs[ind,0].plot([param_mean, param_mean], [*ylims], label='post. mean')

            theta_params = {'a': theta_params[0],
                            'scale': theta_params[1]}

            x = np.linspace(stats.gamma.ppf(0.001, **theta_params),
                            stats.gamma.ppf(0.999, **theta_params), 100)
            axs[ind,0].plot(x, stats.gamma.pdf(x, **theta_params), lw=2, label='prior')
            axs[ind,0].legend()
            axs[ind,0].set_title('theta_{}'.format(name))

            sns.distplot(alphas, ax=axs[ind,1], norm_hist=True, label='posterior')
            ylims = axs[ind,1].get_ylim()
            param_mean = np.mean(alphas)
            axs[ind,1].plot([param_mean, param_mean], [*ylims], label='post. mean')

            alpha_params = {'a': alpha_params[0],
                            'b': alpha_params[1]}

            x = np.linspace(stats.beta.ppf(0.001, **alpha_params),
                            stats.beta.ppf(0.999, **alpha_params), 100)
            axs[ind,1].plot(x, stats.beta.pdf(x, **alpha_params), lw=2, label='prior')
            axs[ind,1].legend()
            axs[ind,1].set_title('beta_{}'.format(name))

            if global_params:
                sns.distplot(ppc_df['global_recievers'], ax=axs[ind,2], norm_hist=True, label='PPC')
                ylims = axs[ind,2].get_ylim()
                axs[ind,2].plot([real_val[-1], real_val[-1]], [*ylims], label='Data')
            else:
                sns.distplot(ppc_df['local_recievers{}'.format(name)], ax=axs[ind,2], norm_hist=True, label='PPC')
                #import pdb
                # fspdb.set_trace()
                ylims = axs[ind,2].get_ylim()
                axs[ind,2].plot([real_val[name], real_val[name]], [*ylims], label='Data')

            axs[ind,2].set_title('PPC_{}'.format(name))
        fig.tight_layout()
        return fig, axs


    def _create_ppc_file(self, file_name, save_file, lower_level_only=True,
                         subject_list=None, num_authors=None):

        if not lower_level_only:
            raise NotImplementedError("We haven't coded the fully randomized PPC :(")
        np.random.seed()
        with open(file_name, 'rb') as param_infile:
            param_dict = pickle.load(param_infile)
        theta = param_dict['theta']
        alpha = param_dict['alpha']
        alpha_s = param_dict['alpha_s']
        theta_s = param_dict['theta_s']

        temp = list(zip(subject_list, num_authors))
        random.shuffle(temp)
        subject_list, num_authors = zip(*temp)
        ppc_create = SharedHollywoodNetwork(label_dist=iter(subject_list),
                                            v=iter(num_authors),
                                            alpha_0=alpha,
                                            theta_0=theta,
                                            alphas=alpha_s,
                                            thetas=theta_s)
        start_time = time.time()
        #import pdb
        #pdb.set_trace()
        X, dc, tc = ppc_create.generate_data(num_edges=len(num_authors))
        end_time = time.time()

        print('Took ' + str((end_time - start_time) / 60) + ' minutes.')

        with open(save_file, 'w') as outfile:
            for (sender, reciever) in X:
                subject_str = '{}'.format(sender)
                reciever_str = ' '.join([str(r) for r in reciever])
                outfile.write(subject_str + ' | ' + reciever_str + '\n')
        return X


    def _get_stat(self, stat_fns, X):
        ppc_stats = []
        for fn in stat_fns:
            ppc_stats.append(fn(X))

        return ppc_stats


    def _num_global_recievers(self, X):
        all_recs = [r for (senders, recievers) in X for r in recievers]
        return len(set(all_recs))


    def _global_power_nmc(self, X):
        tpl = NewmanClausetPower(discrete=True)
        all_recs = [r for (senders, recievers) in X for r in recievers]
        recs, degs = np.unique(all_recs, return_counts=True)
        tpl.fit(degs)
        return tpl.alpha


    def _local_power(self, X):
        X_by_subject = defaultdict(list)
        for (s, r) in X:
            X_by_subject[s].extend(r)

        powers = {}
        for s in X_by_subject.keys():
            tpl = NewmanClausetPower(discrete=True)
            recs, degs = np.unique(X_by_subject[s], return_counts=True)
            tpl.fit(degs)
            powers[s] = tpl.alpha
        return powers


    def _num_local_reciever(self, X, ind):
        local_X = []

        for (s, r) in X:
            if s == ind:
                local_X.extend(r)

        return len(set(local_X))


    def _num_local_recievers(self, X, inds):
        X_by_subject = defaultdict(list)
        for (s, r) in X:
            if s in inds:
                X_by_subject[s].extend(r)

        num_verts = {k: len(set(v)) for (k, v) in X_by_subject.items()}
        return num_verts


