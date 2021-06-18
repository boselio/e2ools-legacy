import numpy as np
import dill as pickle
from bisect import bisect_left, bisect_right
import bisect
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
#import pickle
import os.path
from copy import deepcopy
from ..choice_fns import choice_discrete_unnormalized
from .teem import TemporalProbabilities
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.special import betaln, logsumexp
import pathlib
from concurrent.futures import ProcessPoolExecutor
from .teem import sample_alpha_hmc, sample_theta_hmc
from .teem import run_chain as run_chain_teem
from scipy.stats import expon
from .hollywood import E2Estimator

def dd_list():
    return defaultdict(list)

def df_rec_inds1():
    return np.array([-1], dtype='int')

def df_rec_inds2():
    return defaultdict(df_rec_inds1)


class HTEEM(): 
    def __init__(self, nu, sigma_at=None, alpha=None, theta=None, theta_s=None, 
                    num_chains=4, num_iters_per_chain=500, 
                    holdout=100, alpha_hyperparams=(5, 5),
                    theta_hyperparams=(10, 10),
                    lower_theta_hyperparams=None):

        self.alpha_hyperparams = alpha_hyperparams
        self.theta_hyperparams = theta_hyperparams
        self.lower_theta_hyperparams = lower_theta_hyperparams

        self.alpha = alpha
        self.theta = theta
        self.theta_s = theta_s
        self.nu = nu
        if sigma_at is None:
            self.sigma_at = self.nu / 10
        else:
            self.sigma_at = sigma_at
        
    def initialize_state(self, interactions, smart_initialize, change_times=None):

        #Initialize state variables
        ####Don't think I need this
        #Number of total tables across franchise serving each dish
        self.max_time = interactions[-1][0]
        self.global_table_counts = np.array([])
        ####
        
        if not smart_initialize:
            #Sampled (or given) change times
            self.change_times = change_times
        
            #The locations that are sampled for each change time.
            self.change_locations = [(-1, -1) for _ in self.change_times]
        
        #indices of changes, per sender and per table
        self.table_change_inds = defaultdict(dd_list)
        self.rec_change_inds = defaultdict(dd_list)

        #Number of tables in each restaurant
        self.num_tables_in_s = defaultdict(int)
        
        #The inds that are for a particular receiver, in addition to the new table probability.
        self.receiver_inds = defaultdict(df_rec_inds2)

        #For temporal version, table counts now must be list of lists (or arrays)
        #with each entry of the lower list corresponding to the tables counts at
        #a particular jump point.
        self.table_counts = defaultdict(list)
        
        self.sticks = defaultdict(list)
        #Created inds of each table, according to the change times. 
        #This is a new thing; used to be created_times.
        
        #self.created_inds = defaultdict(list)

        #Global info
        self.global_sticks = np.array([])
        self.global_probs = np.array([1])

        #Sender set
        self.s_set = set([interaction[1] for interaction in interactions])
        self.s_size = len(self.s_set)

        #Reciever set
        self.r_set = set([r for interaction in interactions for r in interaction[2]])
        self.r_size = len(self.r_set)

        self.created_sender_times = {}
        for (t, s, receivers) in interactions:
            if s not in self.created_sender_times:
                self.created_sender_times[s] = t

        if self.alpha is None:
            self.alpha = np.random.beta(*self.alpha_hyperparams)

        if self.theta is None: 
            self.theta = np.random.gamma(*self.theta_hyperparams)

        if self.theta_s is None:
            if self.lower_theta_hyperparams is None:
                self.lower_theta_hyperparams = dict(zip(self.s_set, 
                                [(10, 10)] * self.s_size))

            self.theta_s = {s: np.random.gamma(*self.lower_theta_hyperparams[s]) 
                                                            for s in self.s_set}
        else:
            try: 
                len(self.theta_s)
            except TypeError:
                self.theta_s = {s: self.theta_s for s in self.s_set}

        if smart_initialize:
            self._smart_initialize(interactions)

        self._sample_table_configuration(interactions, initial=True)


    def _smart_initialize(self, interactions, cutoff=1000):

        #Sort interactions by sender
        interactions_by_sender = defaultdict(list)
        for t, s, recs in interactions:
            interactions_by_sender[s].append([t, recs])

        #For every sender, run TEEM to get a reasonable starting place.

        init_dir = self.master_save_dir / 'init'
        if not init_dir.is_dir():
            init_dir.mkdir()

        change_point_list = []
        for s in self.s_set:
            print(s)
            if len(interactions_by_sender[s]) < cutoff:
                continue

            #Normalize the interactions_by_sender.
            renamed_interactions = []
            original_to_inference_labels = {}

            t_min = interactions_by_sender[s][0][0]
            t_max = interactions_by_sender[s][-1][0]
            counter = 0

            for (t, recs) in interactions_by_sender[s]:
                new_recs = []
                for r in recs:
                    if r not in original_to_inference_labels:
                        original_to_inference_labels[r] = counter
                        counter += 1
                    new_recs.append(original_to_inference_labels[r])
                new_t = (t - t_min) / (t_max - t_min) * 100
                renamed_interactions.append([new_t, new_recs])

            inference_to_original_labels = {v: k for (k, v) in 
                                        original_to_inference_labels.items()}

            #We need to choose nu. 
            #nu should be dependent on amount of data.
            nu = max(0, 10 * np.log(0.1 * len(renamed_interactions))) / 100

            #Run E2 to get a better prior for theta, alpha.
            e2_est = E2Estimator()
            e2_est.fit([recs for (t, recs) in renamed_interactions])

            #NEED to add protections for a <= 0.
            alpha_mean = max(e2_est.alphas[-1][0], 0.1)

            alpha_priors = (alpha_mean * 5, (1 - alpha_mean) * 5)
            print(alpha_priors)

            theta_mean = e2_est.thetas[-1][0]
            theta_priors = (theta_mean**(0.5), theta_mean**(0.5))
            print(theta_priors)

            run_chain_teem(init_dir, 100, renamed_interactions, nu, 
                alpha_priors=alpha_priors, theta_priors=theta_priors, 
                update_alpha=True, update_theta=True,
                update_interarrival_times=True, sigma_it = 1 / nu)


            with (init_dir / '49.pkl').open('rb') as infile:
                tp, params = pickle.load(infile)
                for node, at_list in tp.arrival_times_dict.items():
                    if node == -1:
                        continue
                    for at in at_list[1:]:
                        new_at = at * (t_max - t_min) / 100 + t_min
                        change_point_list.append(
                                            (new_at,
                                            s,
                                            inference_to_original_labels[node])
                                            )
            print("Non-dust change_times: {}".format(len(change_point_list)))
        change_point_list.sort()
        self.change_times = []
        self.change_locations = []

        #import pdb
        #pdb.set_trace()
        for ind, (t, s, r) in enumerate(change_point_list):
            self.change_times.append(t)
            self.change_locations.append((s, r))
            self.rec_change_inds[s][r].append(ind)


    def run_chain(self, save_dir, num_times, interactions, change_times=None, smart_initialize=True,
                    sample_parameters=True, update_global_alpha=True, update_global_theta=True,
                    update_local_thetas=True, global_alpha_priors=(1,1), global_theta_priors=(10,10),
                    local_theta_priors=(2,5), update_interarrival_times=False, num_iter_itime=10, seed=None, 
                    debug_fixed_loc=False, print_iter=50):
        
        np.random.seed(seed)
        max_time = interactions[-1][0]

        if change_times is None and not smart_initialize:
            change_times = [np.random.exponential(1 / self.nu)]
            while True:
                itime = np.random.exponential(1 / self.nu)
                if change_times[-1] + itime > max_time:
                    break
                else:
                    change_times.append(change_times[-1] + itime)

        self.initialize_state(interactions, smart_initialize, change_times)

        #Get sender probabilities
        print('Evaluating Sender Probabilities as Normal E2')
        sender_data = [interaction[1] for interaction in interactions]
        unique_senders, sender_degrees = np.unique(sender_data, return_counts=True)

        sender_e2 = E2Estimator()
        sender_e2.fit(sender_data, flat_list=True)
        self.sender_alpha = sender_e2.alphas[-1][0]
        self.sender_theta = sender_e2.thetas[-1][0]

        if self.sender_alpha < 0 or self.sender_theta > 1e6:
            self.sender_probabilities = sender_degrees / sender_degrees.sum()
        else:
            self.sender_probabilities = (sender_degrees - self.sender_alpha) / (sender_degrees.sum() + self.sender_theta)
            self.sender_probabilities = self.sender_probabilities / self.sender_probabilities.sum()
        
        s_time = time.time()

        for t in range(num_times):
            
            if t % print_iter == 0:
                e_time = time.time()
                print('Iteration {}, took {} seconds.'.format(t, e_time - s_time))
                s_time = time.time()

            self._sample_table_configuration(interactions, sample_order=True)
            self._sample_jump_locations_all_rec(interactions, debug_fixed_locations=debug_fixed_loc)

            if update_global_alpha or update_global_theta or update_local_thetas:
                if update_global_alpha:
                    self.alpha, accepted = sample_alpha_hmc(self.alpha, self.theta, self.global_sticks, 
                                    np.arange(len(self.global_sticks)),
                                    alpha_prior=global_alpha_priors[0], 
                                    beta_prior=global_alpha_priors[1])
                
                if update_global_theta:
                    self.theta, accepted = sample_theta_hmc(self.theta, self.alpha, self.global_sticks, 
                                    np.arange(len(self.global_sticks)),
                                    k_prior=global_theta_priors[0], 
                                    theta_prior=global_theta_priors[1])
                
                if update_local_thetas:
                    for s in range(self.s_size):
                        sticks = [s[0] for s in self.sticks[s]]
                        recs = [i for i in range(len(sticks))]
                        for rec, inds in self.rec_change_inds[s].items():
                            for i in inds:
                                for table in self.receiver_inds[s][rec][:-1]:
                                    sticks.append(self.sticks[s][table][i])
                                    recs.append(rec)

                        theta_s, accepted = sample_theta_hmc(self.theta_s[s], 0, np.array(sticks), 
                                    np.array(recs),
                                    k_prior=local_theta_priors[0], 
                                    theta_prior=local_theta_priors[1])
                        self.theta_s[s] = theta_s


            if update_interarrival_times and t % num_iter_itime == 0:
                self.sample_interarrival_times(interactions)
                
            params = {'alpha': self.alpha, 'theta': self.theta,
                        'theta_s': self.theta_s,
                        'receiver_inds': self.receiver_inds,
                        'global_sticks': self.global_sticks,
                        'sticks': self.sticks,
                        'change_times': self.change_times,
                        'table_counts': self.table_counts,
                        'receiver_change_inds': self.rec_change_inds,
                        'change_locations': self.change_locations
                        }
            if t >= num_times / 2:
                file_dir = save_dir / '{}.pkl'.format(t - int(num_times / 2))
                with file_dir.open('wb') as outfile:
                    pickle.dump(params, outfile)


    def infer(self, master_save_dir, interactions, num_chains=4, num_iters_per_chain=500, 
                    update_global_alpha=True, update_global_theta=True, update_local_thetas=True, 
                    change_times=None, update_interarrival_times=False, global_theta_priors=(10,10),
                    global_alpha_priors=(1,1), local_theta_priors=(2,5), num_iter_itime=10, print_iter=50):   


        self.master_save_dir = master_save_dir
        
        rc_func = partial(self.run_chain, num_times=num_iters_per_chain, 
                        interactions=interactions, global_alpha_priors=global_alpha_priors,
                        update_local_thetas=update_local_thetas, update_global_alpha=update_global_alpha, 
                        update_global_theta=update_global_theta, global_theta_priors=global_theta_priors,
                        local_theta_priors=local_theta_priors, num_iter_itime=num_iter_itime,
                        print_iter=print_iter, change_times=change_times, 
                        update_interarrival_times=update_interarrival_times)

        if not pathlib.Path(master_save_dir).is_dir():
            pathlib.Path(master_save_dir).mkdir(parents=True)


        if change_times is not None:
            with (pathlib.Path(master_save_dir) / 'initial_change_times.pkl').open('wb') as outfile:
                pickle.dump(change_times, outfile)
            
        save_dirs = [pathlib.Path(master_save_dir) / '{}'.format(i) 
                     for i in range(num_chains)]

        for sd in save_dirs:
            if not sd.is_dir():
                sd.mkdir(parents=True)

        start_time = time.time()
        print('Beginning Inference:')
        tp_lists = []

        with ProcessPoolExecutor() as executor:
            for _ in executor.map(rc_func, save_dirs):
                continue
        end_time = time.time()

        print('Took {} minutes.'.format((end_time - start_time) / 60))


        #print('Calculating posterior estimates:')
        #start_time = time.time()

        #((upper_limits, lower_limits, means),
        #(probs_ul, probs_ll, probs)) = get_limits_and_means_different_times(save_dir, num_chains, num_iters_per_chain)
        #end_time = time.time()

        #print('Took {} minutes.'.format((end_time - start_time) / 60))

        return


    def _sample_table_configuration(self, interactions, sample_order=False, initial=False):
        
        if initial:
            for t, s, receivers in interactions:
                #import pdb
                #pdb.set_trace()
                for r in receivers:
                    self._add_customer(t, s, r)

            self._sample_sticks(interactions)
            #Now, update all sticks accordingly, WITHOUt updating jump points.

            #degree_mats = {}
            #s_mats = {}

            #beta_mat = np.zeros((num_tables, len(change_times) + 1))
            #table_inds = {}
            #counter = 0
            #import pdb
            #pdb.set_trace()

            #self.sample_ordering()
            #num_senders = len(self.created_sender_times)

            #for s in range(len(self.table_counts)):
            #    degree_mats[s] =  np.array(self.table_counts[s])
            #    s_mats[s] = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mats[s]), axis=0))[1:, :], 
                                                        #np.zeros((1, len(self.change_times) + 1))])

                #for table in range(len(self.table_counts[s])):
                #    try:
                #        degree_mats[s][table, self.created_inds[s][table]] -= 1
                #    except IndexError:
                #        import pdb
                #        pdb.set_trace()

            #for s in range(len(self.table_counts)):
            #    for table in range(len(self.table_counts[s])):
                    #draw beta
            #        begin_ind = 0

            #        new_stick = self.draw_local_beta(degree_mats[s][table,:].sum(),
            #                            s_mats[s][table,:].sum(), self.theta_s[s])
            #        self.sticks[s][table][begin_ind:] = new_stick

        else:
            interaction_inds = np.random.permutation(len(interactions))
            #for t, s, interaction in interactions:
            for i in interaction_inds:
                t, s, receivers = interactions[i]
                for r in receivers:
                    #Remove a customer
                    self._remove_customer(t, s, r)
                    self._add_customer(t, s, r)

            self.remove_empty_tables()

            if sample_order:
                accepted, log_prob = self.sample_ordering(interactions)
                #print(accepted, log_prob)
            
        #Update global sticks
        reverse_counts = np.cumsum(self.global_table_counts[::-1])[::-1]
        reverse_counts = np.concatenate([reverse_counts[1:], [0]])
        #minus 1 because all the global "start" in the same global interval.
        a = 1 - self.alpha + self.global_table_counts - 1
        b = reverse_counts + self.theta + np.arange(1, len(self.global_table_counts)+ 1) * self.alpha
        self.global_sticks = np.random.beta(a, b)
        self.global_probs = np.concatenate([self.global_sticks, [1]])
        self.global_probs[1:] = self.global_probs[1:] * np.cumprod(1 - self.global_probs[:-1])


    def sample_ordering(self, interactions):
        #Need to resample the ordering
        try:
            self.check_change_locations()
        except AssertionError:
            import pdb
            pdb.set_trace()

        rev_placement_list = []

        #Calculate the likelihood of particular people jumping (old).
        old_change_probs = np.zeros(len(self.change_locations))

        for i, (s, r) in enumerate(self.change_locations):
            if s != -1:
                table_sticks = [sticks[i] for sticks in self.sticks[s]]

                probs = np.concatenate([table_sticks, [1]])
                probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1], axis=0)
                change_prob = probs[self.receiver_inds[s][r][:-1]].sum()

            #else:
            #    change_prob = 0
            #    for s_prime, table_sticks in self.sticks.items():
            #        table_sticks = [sticks[i] for sticks in self.sticks[s_prime]]
            #        #Need to calculate the average probability over all time.
            #        probs = np.concatenate([table_sticks, [1]])
            #        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1], axis=0)
            #        change_prob += probs[-1]

                old_change_probs[i] = change_prob

        accepted_list = []
        log_prob_list = []

        for s, table_sticks in self.sticks.items():
            new_table_change_inds = dd_list()
            new_sticks = [np.zeros(len(self.change_times)+1) for t in range(len(table_sticks))]
            new_receiver_inds = df_rec_inds2()
            new_table_counts = []

            log_accept = 0
            #Need to calculate the average probability over all time.
            probs = np.array(table_sticks)
            probs[1:, :] = probs[1:, :] * np.cumprod(1 - probs[:-1, :], axis=0)

            if len(self.change_times) == 0:
                ct_diff = self.max_time
            else:
                ct_diff = np.diff(np.concatenate([[0], self.change_times, [self.max_time]]))

            scores = (probs * ct_diff).sum(axis=1) / self.max_time 
            scores = scores / scores.sum()
            #import pdb
            #pdb.set_trace()
            try:
                new_ordering = np.random.choice(len(scores), size=len(scores), replace=False, p=scores)
            except ValueError:
                import pdb
                pdb.set_trace()

            scores_ordered = scores[new_ordering]
            scores_ordered[1:] = scores_ordered[1:] / (1 - np.cumsum(scores_ordered[:-1]))
            temp = scores_ordered.copy()
            #Transition prob from old to new
            log_accept = -np.log(scores_ordered).sum()

            reverse_new_placements = {v: k for k, v in enumerate(new_ordering)}
            reverse_new_placements[-1] = -1
            
            for k, v in self.table_change_inds[s].items():
                new_table_change_inds[reverse_new_placements[k]] = v


            for r in self.receiver_inds[s].keys():
                new_receiver_inds[r] = np.array([reverse_new_placements[t] for t in self.receiver_inds[s][r]])
                new_receiver_inds[r][:-1] = np.sort(new_receiver_inds[r][:-1])
                
            new_table_counts = [self.table_counts[s][i] for i in new_ordering]

            degree_mat =  np.array(new_table_counts)
            s_mat = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mat), axis=0))[1:, :], 
                                                    np.zeros((1, len(self.change_times) + 1))])

            rev_placement_list.append(reverse_new_placements)

            for table in range(len(new_table_counts)):
                #draw beta
                begin_end = 0
                end_ind = self.get_next_switch(s, table, -1)
                new_stick = self.draw_local_beta(degree_mat[table,:end_ind].sum(),
                                        s_mat[table,:end_ind].sum(), self.theta_s[s])
                new_sticks[table][:end_ind] = new_stick

                while end_ind is not None:
                    begin_ind = end_ind
                    end_ind = self.get_next_switch(s, table, begin_ind)
                    new_stick = self.draw_local_beta(degree_mat[table,begin_ind:end_ind].sum(),
                                        s_mat[table,begin_ind:end_ind].sum(), self.theta_s[s])
                    new_sticks[table][begin_ind:end_ind] = new_stick

            probs = np.array(new_sticks)
            probs[1:, :] = probs[1:, :] * np.cumprod(1 - probs[:-1, :], axis=0)

            if len(self.change_times) == 0:
                ct_diff = self.max_time
            else:
                ct_diff = np.diff(np.concatenate([[0], self.change_times, [self.max_time]]))

            scores = (probs * ct_diff).sum(axis=1) / self.max_time 
            scores = scores / scores.sum()
            scores_ordered = np.array([scores[reverse_new_placements[i]] for i in range(len(scores))])
            scores_ordered[1:] = scores_ordered[1:] / (1 - np.cumsum(scores_ordered[:-1]))
            
            log_accept = log_accept + np.log(scores_ordered).sum()

            for change_r, l in self.rec_change_inds[s].items():
                for i in l:
                    table_sticks = [sticks[i] for sticks in new_sticks]

                    probs = np.concatenate([table_sticks, [1]])
                    probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1], axis=0)
                    change_prob = probs[new_receiver_inds[change_r][:-1]].sum()

                    try:
                        log_accept += np.log(change_prob)
                        log_accept -= np.log(old_change_probs[i])
                    except FloatingPointError:
                        import pdb
                        pdb.set_trace()
            #import pdb
            #pdb.set_trace()
            if np.random.rand() < np.exp(log_accept):
                #import pdb
                #pdb.set_trace()
                self.table_change_inds[s] = new_table_change_inds
                self.sticks[s] = new_sticks
                self.receiver_inds[s] = new_receiver_inds
                self.table_counts[s] = new_table_counts
                accepted = True


            else:
                accepted = False

            accepted_list.append(accepted)
            log_prob_list.append(log_accept)

        return accepted_list, log_prob_list


    def insert_table(self, t, s, r):
        time_bin = bisect_right(self.change_times, t)
        #Randomize?
        #insert_left_point = bisect_left(self.created_inds[s], time_bin)
        #insert_right_point = bisect_right(self.created_inds[s], time_bin)
        #insert_point = np.random.choice(np.arange(insert_left_point, insert_right_point+1))

        #insert_point = insert_right_point
        #for r_prime in self.receiver_inds[s].keys():
        #    ii = self.receiver_inds[s][r_prime] >= insert_point
        #    self.receiver_inds[s][r_prime][ii] = self.receiver_inds[s][r_prime][ii] + 1
        insert_point = len(self.table_counts[s])

        self.receiver_inds[s][r] = np.insert(self.receiver_inds[s][r], -1, insert_point)
        self.num_tables_in_s[s] += 1
        self.table_counts[s].append(np.zeros(len(self.change_times) + 1))
        self.table_counts[s][insert_point][time_bin] += 1

        #self.created_inds[s].insert(insert_point, time_bin)
        #The below line is for onle tables changing (commented out, 1 line)
        #self.sticks[s].append(np.ones(len(self.change_times) + 1) * np.random.beta(1, self.theta_s[s]))
        
        #Need to match the table structure with the receiver, draw the relevant sticks, and set table_change_inds.
        change_inds = self.rec_change_inds[s][r].copy()
        change_inds.insert(0, 0)
        change_inds.append(None)

        sticks = np.ones(len(self.change_times) + 1)
        
        for begin_ind, end_ind in zip(change_inds[:-1], change_inds[1:]):
            sticks[begin_ind:end_ind] = np.random.beta(1, self.theta_s[s])
        
        self.sticks[s].append(sticks)
        
        self.table_change_inds[s][insert_point] = change_inds[1:-1]

        self.global_table_counts[r] += 1

        #for s_temp in range(len(self.receiver_inds)):
        #    temp = np.sort(np.concatenate([l[:-1] for l in self.receiver_inds[s].values()]))
        #    assert (temp == np.arange(len(temp))).all()
        #    assert len(temp) == len(self.table_counts[s])
        #try:
        #    assert np.all(np.diff(self.receiver_inds[s][r][:-1]) >= 0)
        #except AssertionError:
        #    import pdb
        #    pdb.set_trace()


    def _add_customer(self, t, s, r, cython_flag=True):
        if len(self.global_table_counts) == r:
            assert r == len(self.global_sticks)
            #self.global_table_counts gets updated in insert_table
            self.global_table_counts = np.append(self.global_table_counts, [0])
            #self.global_table_counts[r] += 1
            #This is now wrong. Need to insert %after
            self.insert_table(t, s, r)

            #Draw global stick
            self.global_sticks = np.append(self.global_sticks, [np.random.beta(1 - self.alpha, 
                            self.theta + (len(self.global_sticks) + 1) * self.alpha)])
            self.global_probs = np.concatenate([self.global_sticks, [1]])
            self.global_probs[1:] = self.global_probs[1:] * np.cumprod(1 - self.global_probs[:-1])
            return

        probs, table_inds = self.get_unnormalized_probabilities(t, s, r)
        choice = choice_discrete_unnormalized(probs, np.random.rand())
        
        if choice == len(probs)-1:
            self.insert_table(t, s, r)
        else:
            try:
                table = table_inds[choice]
            except IndexError:
                import pdb
                pdb.set_trace()
            time_ind = bisect_right(self.change_times, t)
            self.table_counts[s][table][time_ind] += 1


    def get_unnormalized_probabilities(self, t, s, r):
        time_bin = bisect_right(self.change_times, t)
        #max_point = bisect_right(self.created_inds[s], time_bin)
        #if max_point == 0:
            #No tables have been created at this time.
        #    rec_probs = [1.0]
        #    rec_inds = []
        #    return  rec_probs, rec_inds
        
        sticks = [stick[time_bin] for stick in self.sticks[s]]
        probs = np.concatenate([sticks, [1]])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])
        rec_probs = probs[self.receiver_inds[s][r]]
        
        rec_probs[-1] = rec_probs[-1] * self.global_probs[r]
    
        return rec_probs.tolist(), self.receiver_inds[s][r][:-1]



    def get_table_counts(self, s, table, t):
        #change_ind = bisect_right(self.change_times, t)
        #before_ind = self.get_last_switch(s, i, change_ind)
        #after_ind = self.get_next_switch(s, i, before_ind)
        time_bin = bisect_right(self.change_times, t)
        counts = self.table_counts[s][table][time_bin]

        return counts


    def remove_empty_tables(self):
        try:
            self.check_change_locations()
        except AssertionError:
            import pdb
            pdb.set_trace()
        for s in self.table_counts.keys():
            #Find all the empty tables
            table_counts = np.array(self.table_counts[s]).sum(axis=1)
            empty_tables = np.where(table_counts == 0)[0]
            for e_t in empty_tables:
                for r, inds in self.receiver_inds[s].items():
                        inds = np.flatnonzero(inds == e_t)
                        if len(inds) > 0:
                            ind = inds[0]
                            break
                self.delete_table(s, r, e_t, ind)
                empty_tables[empty_tables > e_t] -= 1

            try:
                assert (np.array(self.table_counts[s]).sum(axis=1) != 0).all()
            except AssertionError:
                import pdb
                pdb.set_trace()
        try:
            self.check_change_locations()
        except AssertionError:
            import pdb
            pdb.set_trace()


    def delete_table(self, s, r, table, ind):
        self.num_tables_in_s[s] -= 1
        self.global_table_counts[r] -= 1
        self.sticks[s] = self.sticks[s][:table] + self.sticks[s][table+1:]
        self.table_counts[s] = self.table_counts[s][:table] +  self.table_counts[s][table+1:]

        self.receiver_inds[s][r] = np.concatenate([self.receiver_inds[s][r][:ind],
                                               self.receiver_inds[s][r][ind+1:]])
        
        #Removed the ind table - so all tables greater than ind+1 -> ind
        for r in self.receiver_inds[s].keys():
            change = self.receiver_inds[s][r] > table
            self.receiver_inds[s][r][change] = self.receiver_inds[s][r][change] - 1

        tables = list(self.table_change_inds[s].keys())
        tables.sort()
        if table in tables:
            del self.table_change_inds[s][table]

        for change_table in tables:
            if change_table > table:
                self.table_change_inds[s][change_table-1] = self.table_change_inds[s][change_table]
                del self.table_change_inds[s][change_table]

        #for i, (change_s, change_table) in enumerate(self.change_locations):
        #    if change_s == s:
        #        if change_table == table:
        #            self.change_locations[i] = (-1, -1)
        #        elif change_table > table:
        #            self.change_locations[i] = (change_s, change_table - 1)

    def _remove_customer(self, t, s, r, cython_flag=True):
        #Choose uniformly at random a customer to remove.
        remove_probs = [self.get_table_counts(s, table, t) for table in self.receiver_inds[s][r][:-1]]
        #remove_probs = [1 if rp >= 1 else 0 for rp in remove_probs]
        ind = choice_discrete_unnormalized(remove_probs, np.random.rand())
        
        table = self.receiver_inds[s][r][ind]

        time_ind = bisect.bisect_right(self.change_times, t)
        self.table_counts[s][table][time_ind] -= 1
        #import pdb
        #pdb.set_trace()
        try:
            assert self.table_counts[s][table][time_ind] >= 0
        except AssertionError:
            import pdb
            pdb.set_trace()


    def _sample_jump_locations_all_rec(self, interactions, debug_fixed_locations=False):

        #try:
        #    self.check_change_locations()
        #except AssertionError:
        #    import pdb
        #    pdb.set_trace()

        #try:
        #    self.check_degrees(interactions)
        #except AssertionError:
        #    import pdb
        #    pdb.set_trace()

        num_tables = sum([len(v) for k, v in self.table_counts.items()])

        change_times = np.array(self.change_times)
        old_locs = self.change_locations
        sorted_inds = change_times.argsort()
        change_times = change_times[sorted_inds]
        old_locs = np.array(old_locs)[sorted_inds]

        interaction_times = np.array([interaction[0] for interaction in interactions])
        max_time = interactions[-1][0]
        #created_set = set()

        permuted_inds = np.random.permutation(len(change_times))
        
        # calculate all degrees between change times for all receivers
        degree_mats = {}
        s_mats = {}

        #beta_mat = np.zeros((num_tables, len(change_times) + 1))
        #table_inds = {}
        #counter = 0

        num_senders = len(self.created_sender_times)

        for s in range(len(self.table_counts)):
            degree_mats[s] =  np.array(self.table_counts[s])
            s_mats[s] = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mats[s]), axis=0))[1:, :], 
                                                    np.zeros((1, len(self.change_times) + 1))])


        #for s in range(num_senders):
            #assert (degree_mats[s] >= 0).all()
            #assert (s_mats[s] >= 0).all()
            #for i in range(s_mats[s].shape[1]):
            #    try:
            #        assert (np.diff(s_mats[s][:, i]) <= 0).all()
            #    except AssertionError:
            #        import pdb
            #        pdb.set_trace()

        for ind in permuted_inds:
            change_s, change_r = self.change_locations[ind]
        #Need to calculate, the likelihood of each stick if that receiver
        #was not chosen.
            #if ind == 13:
            #    import pdb
            #    pdb.set_trace()

            ct = self.change_times[ind]
            if ind > 0:
                begin_time = self.change_times[ind-1]
            else:
                begin_time = 0
            try:
                end_time = self.change_times[ind+1]
            except IndexError:
                end_time = interaction_times[-1] + 1

            num_created_tables = {}
            rec_probs = {}
            log_rec_probs = {}
            non_zero_recs = {}

            before_rec_likelihood_components = {}
            after_rec_likelihood_components = {}
            after_rec_likelihood_potential = {}
            integrated_rec_counts = {}
            beta_temp = {}
            #Calculate likelihood of each jumps
            created_senders = [s for (s, t) in self.created_sender_times.items() if t < ct]

            num_recs = len(self.global_sticks)

            #if ind == 5:
            #    import pdb
            #    pdb.set_trace()
            total_likelihood_components = 0
            for s in created_senders:
                #Calculate log probs for all potential jumps, at the period BEFORE the jump
                table_labels, table_inds = zip(*[(k, i) for (k, v) in self.receiver_inds[s].items() for i in v[:-1]])
                table_labels = np.array(table_labels)
                sorted_inds = np.argsort(table_inds)
                table_labels = table_labels[sorted_inds]
                
                num_tables = len(self.table_counts[s])
                table_sticks = np.array([self.sticks[s][table][ind] for table in range(num_tables)])
                table_probs = np.concatenate([table_sticks,  [1]])
                table_probs[1:] = table_probs[1:] * np.cumprod(1 - table_probs[:-1])

                #rec_probs[s] = np.concatenate([np.array([table_probs[self.receiver_inds[s][r][:-1]].sum(axis=0) 
                #                for r in range(num_recs)]), [table_probs[-1]]])
                #Second way:
                rec_probs[s] = np.bincount(table_labels, table_probs[:-1], minlength=num_recs + 1)
                rec_probs[s][-1] = table_probs[-1]

                #temp = np.bincount(table_labels, table_probs[:-1], minlength=num_recs + 1)
                #temp[-1] = table_probs[-1]

                #assert (temp == rec_probs[s]).all()
                #assert np.isclose(1, rec_probs[s].sum())

                non_zero_recs[s] = np.where(rec_probs[s] > 0)[0]
                log_rec_probs[s] = np.log(rec_probs[s][non_zero_recs[s]])

                before_inds = np.array([self.get_last_switch(s, t, ind) for t in range(num_tables)])
                after_inds = np.array([self.get_next_switch(s, t, ind) for t in range(num_tables)])

                #Now, need to add all other likelihood components, i.e. all degrees for
                #which the receiver did not jump. s, i, ind

                degrees_before = np.array([degree_mats[s][r, before_inds[r]:ind+1].sum() for r in range(num_tables)])
                s_before = np.array([s_mats[s][r, before_inds[r]:ind+1].sum() for r in range(num_tables)])

                before_table_likelihood_components = degrees_before * np.log(np.array(self.sticks[s])[:, ind])
                before_table_likelihood_components += s_before * np.log(1 - np.array(self.sticks[s])[:, ind])
                #before_rec_likelihood_components[s] = np.array([before_table_likelihood_components[self.receiver_inds[s][r][:-1]].sum(axis=0) 
                #                for r in range(num_recs)])

                before_rec_likelihood_components[s] = np.bincount(table_labels, before_table_likelihood_components, minlength=num_recs)
                before_rec_likelihood_components[s] =  before_rec_likelihood_components[s][non_zero_recs[s][:-1]]

                #before_rec_likelihood_components[s] = before_rec_likelihood_components[s][non_zero_recs[s][:-1]]
                #assert (temp == before_rec_likelihood_components[s]).all()

                degrees_after = np.array([degree_mats[s][r, ind+1:after_inds[r]].sum() for r in range(num_tables)])
                s_after = np.array([s_mats[s][r, ind+1:after_inds[r]].sum() for r in range(num_tables)])
                
                #For all after ones, draw new betas
                beta_temp[s] = self.draw_local_beta(degrees_after, s_after, self.theta_s[s])
                after_table_likelihood_potential = degrees_after * np.log(beta_temp[s])
                after_table_likelihood_potential += s_after * np.log(1 - beta_temp[s])
                after_rec_likelihood_potential[s] = np.bincount(table_labels, after_table_likelihood_potential, minlength=num_recs)
                after_rec_likelihood_potential[s] = after_rec_likelihood_potential[s][non_zero_recs[s][:-1]]

                if change_s == s:
                    fix_tables = self.receiver_inds[s][change_r][:-1]
                    beta_fixed = self.draw_local_beta(degrees_after[fix_tables], s_after[fix_tables], self.theta_s[s])
                    degrees = degrees_before[fix_tables] + degrees_after[fix_tables]
                    s_temp = s_before[fix_tables] + s_after[fix_tables]
                    no_change_table_likelihoods = degrees * np.log(beta_fixed)
                    no_change_table_likelihoods += s_temp * np.log(1 - beta_fixed)

                    no_change_ll = no_change_table_likelihoods.sum()

                #Add integrated new beta using future table counts.
                #integrated_table_counts = betaln(1 + degrees_after, self.theta_s[s] + s_after)
                #integrated_table_counts = integrated_table_counts - betaln(1, self.theta_s[s])
                #integrated_rec_counts = np.array([integrated_table_counts[self.receiver_inds[s][r][:-1]].sum(axis=0) 
                #                for r in range(num_recs)])

                #integrated_rec_counts[s] = np.bincount(table_labels, integrated_table_counts, minlength=num_recs)

                #log_rec_probs[s][:-1] += integrated_rec_counts[s][non_zero_recs[s][:-1]]

                #Use ind and not ind+1 because this is the component of the likelihood when the stick does not change.
                after_table_likelihood_components = degrees_after * np.log(np.array(self.sticks[s])[:, ind])
                after_table_likelihood_components += s_after * np.log(1 - np.array(self.sticks[s])[:, ind])

                #after_rec_likelihood_components[s] = np.array([after_table_likelihood_components[self.receiver_inds[s][r][:-1]].sum(axis=0) 
                #                for r in range(num_recs)])

                after_rec_likelihood_components[s] = np.bincount(table_labels, after_table_likelihood_components, minlength=num_recs)
                after_rec_likelihood_components[s] = after_rec_likelihood_components[s][non_zero_recs[s][:-1]]

                total_likelihood_components += after_rec_likelihood_components[s].sum() + before_rec_likelihood_components[s].sum()
                


                #after_rec_likelihood_components[s] = after_rec_likelihood_components[s][non_zero_recs[:-1]]
                #assert (temp == after_rec_likelihood_components[s]).all()

                #if ind == 13:
                #    import pdb
                #    pdb.set_trace()
            #for s in created_senders:
                #for ss in created_senders:
                #    log_rec_probs[s] += np.sum(before_rec_likelihood_components[ss])
                #    log_rec_probs[s] += np.sum(after_rec_likelihood_components[ss])
            #    log_rec_probs[s][:-1] -= after_rec_likelihood_components[s]
            #    log_rec_probs[s] += total_likelihood_components
            #    log_rec_probs[s] += np.log(self.sender_probabilities[s])

            if change_s != -1:
                try:
                    change_ind = np.where(non_zero_recs[change_s] == change_r)[0][0]
                except KeyError:
                    import pdb
                    pdb.set_trace()

            for s in created_senders:
                log_rec_probs[s] += total_likelihood_components

                #remove old change ll
                if change_s != -1:
                    log_rec_probs[s] -= after_rec_likelihood_components[change_s][change_ind]
                    log_rec_probs[s] -= before_rec_likelihood_components[change_s][change_ind]
                    #add in new non-change ll
                    log_rec_probs[s] += no_change_ll

                #remove old after ll
                log_rec_probs[s][:-1] -= after_rec_likelihood_components[s]


                #replace with new potential ll
                log_rec_probs[s][:-1] += after_rec_likelihood_potential[s]

                #Add sender prior probs
                log_rec_probs[s] += np.log(self.sender_probabilities[s])

            if change_s != -1:
                log_rec_probs[change_s][change_ind] -= no_change_ll
                log_rec_probs[change_s][change_ind] += after_rec_likelihood_components[change_s][change_ind]
                log_rec_probs[change_s][change_ind] += before_rec_likelihood_components[change_s][change_ind]

            #import pdb
            #pdb.set_trace()
            #First, choose sender:
            integrated_sender_log_probs = [logsumexp(log_rec_probs[s]) for s in created_senders]
            integrated_sender_probs = np.exp(integrated_sender_log_probs - logsumexp(integrated_sender_log_probs))
            new_s = np.random.choice(created_senders, p=integrated_sender_probs)

            #log_prob = np.concatenate([log_probs[s] for s in range(num_senders)])
            #probs = np.exp(log_prob - logsumexp(log_prob))
            probs = np.exp(log_rec_probs[new_s] - logsumexp(log_rec_probs[new_s]))

            #num_total_tables = sum(num_created_tables.values())
            new_ind = np.random.choice(len(non_zero_recs[new_s]), p=probs)
            new_r = non_zero_recs[new_s][new_ind]

            #if new_r != len(self.r_set):
                #try:
                #    assert new_r in self.r_set_by_sender[new_s]
                #except AssertionError:
                #    import pdb
                #    pdb.set_trace()
            #temp = np.cumsum([num_created_tables[i] + 1 for i in range(num_senders)])
            #new_s = bisect_right(temp, new_ind)
            #if new_s > 0:
            #    new_t = new_ind - temp[new_s - 1]

            new_choice = (new_s, new_r)

            #Delete this next section of code for production:
            if debug_fixed_locations:
                new_choice = old_locs[ind]

            if (new_choice[0] == old_locs[ind][0]) and (new_choice[1] == old_locs[ind][1]):
                if new_choice[1] == len(self.table_counts[s]):
                    #Do nothing, it stayed in the tail
                    continue
                else:
                    #Draw the beta s, i, ind
                    for table in self.receiver_inds[new_s][new_r][:-1]:
                        begin_ind = self.get_last_switch(new_s, table, ind)
                        end_ind = self.get_next_switch(new_s, table, ind)
                        
                        new_stick = self.draw_local_beta(degree_mats[new_s][table, ind+1:end_ind].sum(), 
                                                    s_mats[new_s][table, ind+1:end_ind].sum(), self.theta_s[new_s])
                        self.sticks[new_s][table][ind+1:end_ind] = new_stick

                        new_stick = self.draw_local_beta(degree_mats[new_s][table, begin_ind:ind+1].sum(), 
                                                    s_mats[new_s][table, begin_ind:ind+1].sum(), self.theta_s[new_s])
                        self.sticks[new_s][table][begin_ind:ind+1] = new_stick


            else:
                old_loc = old_locs[ind]
                s_del = old_loc[0]
                r_del = old_loc[1]

                if s_del != -1:
                    self.rec_change_inds[s_del][r_del].remove(ind)
                    
                    for table in self.receiver_inds[s_del][r_del][:-1]:
                        try:
                            self.table_change_inds[s_del][table].remove(ind)
                        except ValueError:
                            import pdb
                            pdb.set_trace()

                        # redraw the beta that we had deleted.
                        begin_ind = self.get_last_switch(s_del, table, ind)
                        end_ind = self.get_next_switch(s_del, table, ind)

                        new_stick = self.draw_local_beta(degree_mats[s_del][table, begin_ind:end_ind].sum(), 
                                                s_mats[s_del][table, begin_ind:end_ind].sum(), self.theta_s[s_del])

                        self.sticks[s_del][table][begin_ind:end_ind] = new_stick


                if new_r == len(self.r_set):
                    #import pdb
                    #pdb.set_trace()
                    self.change_locations[ind] = (-1, -1)

                else:
                    self.change_locations[ind] = (new_s, new_r)

                    insert_ind = bisect_right(self.rec_change_inds[new_s][new_r], ind)
                    self.rec_change_inds[new_s][new_r].insert(insert_ind, ind)

                    for table in self.receiver_inds[new_s][new_r][:-1]:
                        insert_ind = bisect_right(self.table_change_inds[new_s][table], ind)
                        self.table_change_inds[new_s][table].insert(insert_ind, ind)
                    # Draw the beta backward
                        begin_ind = self.get_last_switch(new_s, table, ind)
                        end_ind = self.get_next_switch(new_s, table, ind)
                    
                        new_stick = self.draw_local_beta(degree_mats[new_s][table, ind+1:end_ind].sum(), 
                                                s_mats[new_s][table, ind+1:end_ind].sum(), self.theta_s[new_s])

                        self.sticks[new_s][table][ind+1:end_ind] = new_stick

                        new_stick = self.draw_local_beta(degree_mats[new_s][table, begin_ind:ind+1].sum(), 
                                                s_mats[new_s][table, begin_ind:ind+1].sum(), self.theta_s[new_s])
                        self.sticks[new_s][table][begin_ind:ind+1] = new_stick

        for s in range(num_senders):
            for table in range(len(self.table_counts[s])):
                #draw beta
                end_ind = self.get_next_switch(s, table, -1)
                new_stick = self.draw_local_beta(degree_mats[s][table,:end_ind].sum(),
                                        s_mats[s][table,:end_ind].sum(), self.theta_s[s])
                self.sticks[s][table][:end_ind] = new_stick

        try:
            self.check_change_locations()
        except AssertionError:
            import pdb
            pdb.set_trace()
        return 



    def _sample_sticks(self, interactions, debug_fixed_locations=False):

        permuted_inds = np.random.permutation(len(self.change_times))
        
        # calculate all degrees between change times for all receivers
        degree_mats = {}
        s_mats = {}

        num_senders = len(self.created_sender_times)

        for s in range(len(self.table_counts)):
            degree_mats[s] =  np.array(self.table_counts[s])
            s_mats[s] = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mats[s]), axis=0))[1:, :], 
                                                    np.zeros((1, len(self.change_times) + 1))])

        for ind in permuted_inds:
            s, r = self.change_locations[ind]
            for table in self.receiver_inds[s][r][:-1]:
                begin_ind = self.get_last_switch(s, table, ind)
                end_ind = self.get_next_switch(s, table, ind)
                
                new_stick = self.draw_local_beta(degree_mats[s][table, ind+1:end_ind].sum(), 
                                            s_mats[s][table, ind+1:end_ind].sum(), self.theta_s[s])
                self.sticks[s][table][ind+1:end_ind] = new_stick

                new_stick = self.draw_local_beta(degree_mats[s][table, begin_ind:ind+1].sum(), 
                                            s_mats[s][table, begin_ind:ind+1].sum(), self.theta_s[s])
                self.sticks[s][table][begin_ind:ind+1] = new_stick

        for s in range(num_senders):
            for table in range(len(self.table_counts[s])):
                #draw beta
                end_ind = self.get_next_switch(s, table, -1)
                new_stick = self.draw_local_beta(degree_mats[s][table,:end_ind].sum(),
                                        s_mats[s][table,:end_ind].sum(), self.theta_s[s])
                self.sticks[s][table][:end_ind] = new_stick

        try:
            self.check_change_locations()
        except AssertionError:
            import pdb
            pdb.set_trace()
        return 


    def check_change_locations(self):
        num_changes = len(self.change_times)

        for i in range(num_changes):
            s, r = self.change_locations[i]
            if s == -1:
                continue
            for table in self.receiver_inds[s][r][:-1]:
                assert i in self.table_change_inds[s][table], "{}, {}, {}, {}".format(s, r, table, i)


    def check_degrees(self, interactions):
        change_times = np.concatenate([[0], self.change_times, [interactions[-1][0] + 1]])
        interaction_times = [interaction[0] for interaction in interactions]
        bins = np.digitize(interaction_times, change_times) - 1
        true_count_dict = {s: np.zeros((self.r_size, len(self.change_times) + 1)) for s in range(len(self.sticks))}

        for time_bin, interaction in zip(bins, interactions):
            s = interaction[1]
            for r in interaction[-1]:
                true_count_dict[s][r, time_bin] += 1

        test_count_dict = {s: np.zeros((self.r_size, len(self.change_times) + 1)) for s in range(len(self.sticks))}
        for s in range(len(self.sticks)):
            for r in range(self.r_size):
                rec_table_counts = np.array([self.table_counts[s][table] for table in self.receiver_inds[s][r][:-1]]).sum(axis=0)
                test_count_dict[s][r] = rec_table_counts

                try:
                    assert (true_count_dict[s][r] == test_count_dict[s][r]).all()
                except AssertionError:
                    print(s,r)

        

    def get_next_switch(self, s, i, ind):
        after_ind = bisect_right(self.table_change_inds[s][i], ind)
        if after_ind == len(self.table_change_inds[s][i]):
            return None
        return self.table_change_inds[s][i][after_ind] + 1
        

    def get_last_switch(self, s, i, ind):
        before_ind = bisect_left(self.table_change_inds[s][i], ind) - 1
        if before_ind == -1:
            return 0

        return self.table_change_inds[s][i][before_ind] 
            

    def draw_local_beta(self, d, s, theta):
        return np.random.beta(d + 1, s + theta)


    def evaluate_itime_likelihood(self, interactions, s_change, r_change, begin_time, end_time, change_time, change_ind, return_degrees=False):
        ll = 0
        interaction_times = [interaction[0] for interaction in interactions]

        begin_ind = bisect_left(interaction_times, begin_time)
        proposal_ind = bisect_right(interaction_times, change_time)
        end_ind = bisect_right(interaction_times, end_time)
        #sample the table distribution
        #table_probs = np.array([stick[change_ind] for stick in self.sticks[s]])
        #table_probs[1:] = table_probs[1:] * np.cumprod(1 - table_probs[:-1])

        new_table_counts = np.zeros((len(self.table_counts[s]), 2))

        for interaction in interactions[begin_ind:proposal_ind]:
            if interaction[1] != s_change:
                continue
            t = interaction[0]
            for r in interaction[-1]:
                probs, table_inds = self.get_unnormalized_probabilities(t, s_change, r)
                choice = choice_discrete_unnormalized(probs[:-1], np.random.rand())
                try:
                    table = table_inds[choice]
                except IndexError:
                    import pdb
                    pdb.set_trace()
                
                new_table_counts[table][0] += 1

        for interaction in interactions[proposal_ind:end_ind]:
            if interaction[1] != s_change:
                continue
            t = interaction[0]
            for r in interaction[-1]:
                probs, table_inds = self.get_unnormalized_probabilities(t, s, r)
                choice = choice_discrete_unnormalized(probs[:-1], np.random.rand())
                table = table_inds[choice]

                new_table_counts[table][1] += 1

        s_counts = np.vstack([np.flipud(np.cumsum(np.flipud(new_table_counts), axis=0))[1:, :], 
                                                    np.zeros((1, 2))])

        try:
            sticks = np.array(self.sticks[s])[:, change_ind:change_ind+2]
        except IndexError:
            import pdb
            pdb.set_trace()

        ll = (new_table_counts * np.log(sticks) + s_counts * np.log(1 - sticks)).sum()

        ll += expon.logpdf(change_time - begin_time, scale=1/self.nu)
        ll += expon.logpdf(end_time - change_time, scale=1/self.nu)

        if return_degrees:
            return ll, new_table_counts
        else:
            return ll


    def sample_interarrival_times(self, interactions):
        
        accepted = []
        log_acceptance_probs = []
        for i, at in enumerate(self.change_times):
            s_change, r_change = self.change_locations[i]

            if s_change == -1:
                accepted.append(False)
                log_acceptance_probs.append(-1)
                continue

            if i == 0:
                begin_time = 0
            else:
                begin_time = self.change_times[i-1]
            if i == len(self.change_times) - 1:
                end_time = interactions[-1][0]
            else:
                end_time = self.change_times[i+1]

            ll_old = self.evaluate_itime_likelihood(interactions, s_change, r_change, 
                                                            begin_time, end_time, at, i)

            at_new = at + self.sigma_at * np.random.randn()

            if (at_new < begin_time) or (at_new > end_time):
                accepted.append(False)
                log_acceptance_probs.append(-1)
                continue

            ll_new, new_table_counts = self.evaluate_itime_likelihood(interactions, s_change, r_change,
                                                            begin_time, end_time, at_new, i,
                                                            return_degrees=True)
            
            log_acceptance_probs.append(ll_new - ll_old)
            print(log_acceptance_probs)
            if np.log(np.random.rand()) < ll_new - ll_old:
                print(at_new)
                accepted.append(True)
                self.change_times[i] = at_new
                for table in range(len(self.table_counts[s_change])):
                    self.table_counts[s_change][table][i:i+2] = new_table_counts[table, :]

            else:
                accepted.append(False)

        return accepted, log_acceptance_probs
        

def get_limits_and_means_different_times(gibbs_dir, num_chains, num_iters_per_chain, 
                                         prob_name='prob_avgs.pkl'):

    times = []

    save_dirs = [os.path.join(gibbs_dir, '{}'.format(i)) for i in range(num_chains)]
    param_master_list = []
    for save_dir in save_dirs:
        for i in range(int(num_iters_per_chain / 2)):
            save_path = os.path.join(save_dir, '{}.pkl'.format(i))
            with open(save_path, 'rb') as infile:
                params = pickle.load(infile)
                times.append(params.change_times)
                param_master_list.append(params)
    
    times = np.concatenate(times)
    times = np.unique(times)
    times.concatenate([times, [0]])
    times.sort()
    
    num_times = len(times)
    num_recs = len(params['global_sticks'])
    num_senders = len(params['table_counts'])
    mean_dict = {}
    median_dict = {}
    ul_dict = {}
    ll_dict = {}


    for s in range(num_senders):
        rec_prob_list = []
        for params in param_list:
            receiver_inds = params['receiver_inds']
            table_labels, table_inds = zip(*[(k, i) for (k, v) in receiver_inds[s].items() for i in v[:-1]])
            table_labels = np.array(table_labels)
            sorted_inds = np.argsort(table_inds)
            table_labels = table_labels[sorted_inds]

            table_probs = np.array(params['sticks'][s])
            table_probs[1:, :] = table_probs[1:, :] * np.cumprod(1 - table_probs[:-1, :], axis=0)
        
            rec_probs = np.zeros((num_recs, num_times))
            for i in range(num_times):
                rec_probs[:, i] = np.bincount(table_labels, table_probs[:, i], minlength=num_recs)

            master_inds = np.digitize(times, params['change_times'], right=False) - 1

            rec_prob_list.append(rec_probs[:, master_inds])
            
        rec_prob_array = np.array(rec_prob_list)
        del rec_prob_list

        upper_limits = np.percentile(rec_prob_array, 97.5, axis=0)
        ul_dict[s] = upper_limits

        lower_limits = np.percentile(rec_prob_array, 2.5, axis=0)
        ll_dict[s] = lower_limits

        means = rec_prob_array.mean(axis=0)
        mean_dict[s] = means

        medians = np.median(rec_prob_array, axis=0)
        median_dict[s] = medians



    with open(os.path.join(gibbs_dir, prob_name), 'wb') as outfile:
        pickle.dump({'means': mean_dict,
                     'upper_limits': ul_dict,
                     'lower_limits': ll_dict,
                     'medians': median_dict}, 
                     outfile)

    return (ul_dict, ll_dict, mean_dict)


def read_files(save_dir=None, num_chains=4, num_iters_per_chain=500):

    save_dirs = [save_dir /  '{}'.format(i) for i in range(num_chains)]
    param_dicts = []
    for d in save_dirs:
        for i in range(int(num_iters_per_chain / 2)):
            save_path = d / '{}.pkl'.format(i)
            with save_path.open('rb') as infile:
                param_dicts.append(pickle.load(infile))

    return param_dicts


def instantiate_and_run(save_dir, interactions, nu=None, alpha=None, theta=None, theta_local=None,
                        num_iters=500, update_alpha=True, update_theta=True,
                        update_theta_local=True, update_interarrival_times=True,
                        change_times=None):
    np.random.seed()
    he2 = HTEEM(nu, alpha=alpha, theta=theta, theta_s=theta_local)
    he2.run_chain(save_dir, num_iters, interactions, change_times,
                    sample_parameters=False, update_alpha=False,
                    update_theta=False, update_interarrival_times=False,
                    seed=None)
    
    return


def infer(master_save_dir, interactions, nu=None, num_chains=4, num_iters_per_chain=500, 
                alpha=None, theta=None, theta_local=None, update_alpha=True, 
                update_theta=True, update_theta_local=True, change_times=None, 
                update_interarrival_times=True):   


    rc_func = partial(instantiate_and_run, nu=nu, num_iters=num_iters_per_chain, 
                    interactions=interactions, alpha=alpha, theta=theta, theta_local=theta_local,
                    update_theta=update_theta, update_alpha=update_alpha, update_theta_local=update_theta_local,
                  change_times=change_times, update_interarrival_times=update_interarrival_times)

    if not pathlib.Path(master_save_dir).is_dir():
        pathlib.Path(master_save_dir).mkdir(parents=True)


    if change_times is not None:
        with (pathlib.Path(master_save_dir) / 'initial_change_times.pkl').open('wb') as outfile:
            pickle.dump(change_times, outfile)
        
    save_dirs = [pathlib.Path(master_save_dir) / '{}'.format(i) 
                 for i in range(num_chains)]

    for sd in save_dirs:
        if not sd.is_dir():
            sd.mkdir(parents=True)

    start_time = time.time()
    print('Beginning Inference:')
    tp_lists = []

    with ProcessPoolExecutor() as executor:
        for _ in executor.map(rc_func, save_dirs):
            continue
    end_time = time.time()

    print('Took {} minutes.'.format((end_time - start_time) / 60))


    #print('Calculating posterior estimates:')
    #start_time = time.time()

    #((upper_limits, lower_limits, means),
    #(probs_ul, probs_ll, probs)) = get_limits_and_means_different_times(save_dir, num_chains, num_iters_per_chain)
    #end_time = time.time()

    #print('Took {} minutes.'.format((end_time - start_time) / 60))

    return