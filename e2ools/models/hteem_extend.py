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

def dd_list():
    return defaultdict(list)

def df_rec_inds1():
    return np.array([-1], dtype='int')

def df_rec_inds2():
    return defaultdict(df_rec_inds1)


class HTEEM(): 
    def __init__(self, nu, alpha=None, theta=None, theta_s=None, 
                    num_chains=4, num_iters_per_chain=500, 
                    holdout=100, alpha_hyperparams=(5, 5),
                    theta_hyperparams=(10, 10), lower_alpha_hyperparams=(5,5),
                    lower_theta_hyperparams=(10,10)):

        self.alpha_hyperparams = alpha_hyperparams
        self.theta_hyperparams = theta_hyperparams
        self.lower_theta_hyperparams = lower_theta_hyperparams

        self.alpha = alpha
        self.theta = theta
        self.theta_s = theta_s

        
    def initialize_state(self, interactions, change_times):

        #Initialize state variables
        ####Don't think I need this
        #Number of total tables across franchise serving each dish
        self.max_time = interactions[-1][0]
        self.global_table_counts = np.array([])
        ####
        
        #Sampled (or given) change times
        self.change_times = change_times
        
        #The locations that are sampled for each change time.
        self.change_locations = [(-1, -1) for _ in self.change_times]
        
        #indices of changes, per sender and per table
        self.table_change_inds = defaultdict(dd_list)
        
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
            if len(self.lower_theta_hyperparams) != self.s_size:
                self.lower_theta_hyperparams = dict(zip(self.s_set, 
                                [self.lower_theta_hyperparams] * self.s_size))

            self.theta_s = {s: np.random.gamma(*self.lower_theta_hyperparams[s]) 
                                                            for s in self.s_set}
        else:
            try: 
                len(self.theta_s)
            except TypeError:
                self.theta_s = {s: self.theta_s for s in self.s_set}

        self._sample_table_configuration(interactions, initial=True)


    def run_chain(self, save_dir, num_times, interactions, change_times=None,
                    sample_parameters=True, update_global_alpha=False, update_global_theta=False,
                    update_local_thetas=False, global_alpha_priors=(1,1), globasl_theta_priors=(10,10),
                    local_theta_priors=(2,5), update_interarrival_times=False, seed=None, 
                    debug_fixed_loc=False):
        
        np.random.seed(seed)
        max_time = interactions[-1][0]

        if change_times is None:
            change_times = [np.random.exponential(1 / nu)]
            while True:
                itime = np.random.exponential(1 / nu)
                if change_times[-1] + itime > max_time:
                    break
                else:
                    change_times.append(change_times[-1] + itime)

        self.initialize_state(interactions, change_times)

        s_time = time.time()
        for t in range(num_times):
            
            if t % 50 == 0:
                e_time = time.time()
                print('Iteration {}, took {} seconds.'.format(t, e_time - s_time))
                s_time = time.time()

            self._sample_table_configuration(interactions)
            self._sample_jump_locations(interactions, debug_fixed_locations=debug_fixed_loc)


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
                        sticks = np.concatenate(self.sticks[s])
                        rs = np.array([i for (i, a) in sticks[s] for _ in a])
                        theta_s, accepted = sample_theta_hmc(self.theta_s[s], 0, sticks, rs,
                                    k_prior=local_theta_priors[0], 
                                    theta_prior=local_theta_priors[1])
                        self.theta_s[s] = theta_s


            if update_interarrival_times:
                self.sample_interarrival_times()
                
            params = {'alpha': self.alpha, 'theta': self.theta,
                        'theta_s': self.theta_s,
                        'receiver_inds': self.receiver_inds,
                        'global_sticks': self.global_sticks,
                        'sticks': self.sticks,
                        'change_times': change_times,
                        'table_counts': self.table_counts,
                        }
            if t >= num_times / 2:
                file_dir = save_dir / '{}.pkl'.format(t - int(num_times / 2))
                with file_dir.open('wb') as outfile:
                    pickle.dump(params, outfile)


    def _sample_table_configuration(self, interactions, initial=False):

        #import pdb
        #pdb.set_trace()
        if initial:
            for t, s, receivers in interactions:
                #import pdb
                #pdb.set_trace()
                for r in receivers:
                    self._add_customer(t, s, r)

            degree_mats = {}
            s_mats = {}

            #beta_mat = np.zeros((num_tables, len(change_times) + 1))
            #table_inds = {}
            #counter = 0
            #import pdb
            #pdb.set_trace()

            #self.sample_ordering()
            num_senders = len(self.created_sender_times)

            for s in range(len(self.table_counts)):
                degree_mats[s] =  np.array(self.table_counts[s])
                s_mats[s] = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mats[s]), axis=0))[1:, :], 
                                                        np.zeros((1, len(self.change_times) + 1))])

                #for table in range(len(self.table_counts[s])):
                #    try:
                #        degree_mats[s][table, self.created_inds[s][table]] -= 1
                #    except IndexError:
                #        import pdb
                #        pdb.set_trace()

            for s in range(len(self.table_counts)):
                for table in range(len(self.table_counts[s])):
                    #draw beta
                    begin_ind = 0

                    new_stick = self.draw_local_beta(degree_mats[s][table,:].sum(),
                                        s_mats[s][table,:].sum(), self.theta_s[s])
                    self.sticks[s][table][begin_ind:] = new_stick

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
            #self.sample_ordering()
            #sample_ordering()

        #Update global sticks
        reverse_counts = np.cumsum(self.global_table_counts[::-1])[::-1]
        reverse_counts = np.concatenate([reverse_counts[1:], [0]])
        #minus 1 because all the global "start" in the same global interval.
        a = 1 - self.alpha + self.global_table_counts - 1
        b = reverse_counts + self.theta + np.arange(1, len(self.global_table_counts)+ 1) * self.alpha
        self.global_sticks = np.random.beta(a, b)
        self.global_probs = np.concatenate([self.global_sticks, [1]])
        self.global_probs[1:] = self.global_probs[1:] * np.cumprod(1 - self.global_probs[:-1])


    def sample_ordering(self):
        #Need to resample the ordering
        
        for s, table_sticks in self.sticks.items():
            #Need to calculate the average probability over all time.
            ct_diff = np.diff(np.concatenate([self.change_times, [self.max_time]]))
            probs = np.array(table_sticks)
            probs[1:, :] = probs[1:, :] * np.cumprod(1 - probs[:-1, :], axis=0)

            scores = (probs * ct_diff).sum(axis=1) / self.max_time 
            scores = scores / scores.sum()
            new_ordering = np.random.choice(len(scores), size=len(scores), replace=False, p=scores)

            reverse_new_placements = {v: k for k, v in enumerate(new_ordering)}
            reverse_new_placements[-1] = [-1]
            self.sticks[s] = [self.sticks[s][i] for i in new_ordering]
            for r in self.receiver_inds[s].keys():
                self.receiver_inds[s][r] = np.array([reverse_new_placements[t] for t in self.receiver_inds[s][r]])
                self.receiver_inds[s][r][:-1] = np.sort(self.receiver_inds[s][r][:-1])

            self.table_counts[s] = [self.table_counts[s][i] for i in new_ordering]


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
        self.sticks[s].append(np.ones(len(self.change_times) + 1) * np.random.beta(1, self.theta_s[s]))
        #self.sticks[s][insert_point][:time_bin] = 1
        self.global_table_counts[r] += 1

        for s_temp in range(len(self.receiver_inds)):
            temp = np.sort(np.concatenate([l[:-1] for l in self.receiver_inds[s].values()]))
            assert (temp == np.arange(len(temp))).all()
            assert len(temp) == len(self.table_counts[s])
        try:
            assert np.all(np.diff(self.receiver_inds[s][r][:-1]) >= 0)
        except AssertionError:
            import pdb
            pdb.set_trace()


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


    def get_stick(self, s, table, t):
        time_bin = bisect_right(self.change_times, t)
        stick = self.sticks[s][table][time_bin]
        
        return stick


    def get_table_counts(self, s, table, t):
        #change_ind = bisect_right(self.change_times, t)
        #before_ind = self.get_last_switch(s, i, change_ind)
        #after_ind = self.get_next_switch(s, i, before_ind)
        time_bin = bisect_right(self.change_times, t)
        counts = self.table_counts[s][table][time_bin]

        return counts


    def remove_empty_tables(self):

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


    def _sample_jump_locations(self, interactions, debug_fixed_locations=False):

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
            try:
                s_mats[s] = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mats[s]), axis=0))[1:, :], 
                                                    np.zeros((1, len(self.change_times) + 1))])
            except ValueError:
                import pdb
                pdb.set_trace()

            #for i in range(len(self.table_counts[s])):
                #begin_ind = self.created_inds[s][i]
                #degree_mats[s][i, begin_ind] -= 1

        for s in range(num_senders):
            try:
                assert (degree_mats[s] >= 0).all()
            except AssertionError:
                import pdb
                pdb.set_trace()
            assert (s_mats[s] >= 0).all()

        for ind in permuted_inds:
        #Need to calculate, the likelihood of each stick if that receiver
        #was not chosen.

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
            probs = {}
            log_probs = {}

            before_likelihood_components = {}
            after_likelihood_components = {}
            #Calculate likelihood of each jumps
            created_senders = [s for (s, t) in self.created_sender_times.items() if t < ct]

            for s in created_senders:
                #Calculate log probs for all potential jumps, at the period BEFORE the jump
                num_tables = len(self.table_counts[s])
                probs[s] = np.array([self.get_stick(s, i, ct - 1e-8) for i in range(num_tables)] + [1])
                probs[s][1:] = probs[s][1:] * np.cumprod(1 - probs[s][:-1])
                log_probs[s] = np.log(probs[s])

                #Add integrated new beta using future table counts.
                log_probs[s][:-1] += betaln(1 + degree_mats[s][:, ind+1], 
                                    self.theta_s[s] + s_mats[s][:, ind+1])
            
                #Now, need to add all other likelihood components, i.e. all degrees for
                #which the receiver did not jump. s, i, ind
                before_inds = np.array([self.get_last_switch(s, t, ind) for t in range(num_tables)])
                after_inds = np.array([self.get_next_switch(s, t, ind) for t in range(num_tables)])

                degrees_before = np.array([degree_mats[s][r, before_inds[r]:ind+1].sum() for r in range(num_tables)])
                s_before = np.array([s_mats[s][r, before_inds[r]:ind+1].sum() for r in range(num_tables)])

                before_likelihood_components[s] = degrees_before * np.log(np.array(self.sticks[s])[:, ind])
                before_likelihood_components[s] += s_before * np.log(1 - np.array(self.sticks[s])[:, ind])

                degrees_after = np.array([degree_mats[s][r, ind+1:after_inds[r]].sum() for r in range(num_tables)])
                s_after = np.array([s_mats[s][r, ind+1:after_inds[r]].sum() for r in range(num_tables)])
                
                after_likelihood_components[s] = degrees_after * np.log(np.array(self.sticks[s])[:, ind+1])
                after_likelihood_components[s] += s_after * np.log(1 - np.array(self.sticks[s])[:, ind+1])

            for s in created_senders:
                for ss in created_senders:
                    log_probs[s] += np.sum(before_likelihood_components[ss])
                    log_probs[s] += np.sum(after_likelihood_components[ss])
                log_probs[s][:-1] -= after_likelihood_components[s]

            #First, choose sender:
            integrated_sender_log_probs = [logsumexp(log_probs[s]) for s in created_senders]
            integrated_sender_probs = np.exp(integrated_sender_log_probs - logsumexp(integrated_sender_log_probs))
            new_s = np.random.choice(created_senders, p=integrated_sender_probs)

            #log_prob = np.concatenate([log_probs[s] for s in range(num_senders)])
            #probs = np.exp(log_prob - logsumexp(log_prob))
            probs = np.exp(log_probs[new_s] - logsumexp(log_probs[new_s]))
            #num_total_tables = sum(num_created_tables.values())
            new_t = np.random.choice(len(self.table_counts[new_s]) + 1, p=probs)

            #temp = np.cumsum([num_created_tables[i] + 1 for i in range(num_senders)])
            #new_s = bisect_right(temp, new_ind)
            #if new_s > 0:
            #    new_t = new_ind - temp[new_s - 1]

            new_choice = (new_s, new_t)

            #Delete this next section of code for production:
            if debug_fixed_locations:
                new_choice = old_locs[ind]

            if (new_choice[0] == old_locs[ind][0]) and (new_choice[1] == old_locs[ind][1]):
                if new_choice[1] == len(self.table_counts[s]):
                    #Do nothing, it stayed in the tail
                    continue
                else:
                    #Draw the beta s, i, ind
                    end_ind = self.get_next_switch(new_s, new_t, ind)
                    if end_ind == -1:
                        end_time = max_time
                    begin_ind = self.get_last_switch(new_s, new_t, ind)
                    end_ind = bisect_right(interaction_times, end_time)

                    new_stick = self.draw_local_beta(degree_mats[new_s][new_t, ind+1:end_ind].sum(), 
                                                s_mats[new_s][new_t, ind+1:end_ind].sum(), self.theta_s[new_s])
                    self.sticks[new_s][new_t][ind+1:end_ind] = new_stick

                    new_stick = self.draw_local_beta(degree_mats[new_s][new_t, begin_ind:ind+1].sum(), 
                                                s_mats[new_s][new_t, begin_ind:ind+1].sum(), self.theta_s[new_s])
                    self.sticks[new_s][new_t][begin_ind:ind+1] = new_stick


            else:
                old_loc = old_locs[ind]
                s_del = old_loc[0]
                t_del = old_loc[1]
                if s_del != -1:
                    self.table_change_inds[s_del][t_del].remove(ind)
                    # redraw the beta that we had deleted.
                    begin_ind = self.get_last_switch(s_del, t_del, ind)
                    end_ind = self.get_next_switch(s_del, t_del, ind)
                    if end_time == -1:
                        end_time = max_time

                    try:
                        new_stick = self.draw_local_beta(degree_mats[s_del][t_del, begin_ind:end_ind].sum(), 
                                                s_mats[s_del][t_del, begin_ind:end_ind].sum(), self.theta_s[s_del])
                    except IndexError:
                        import pdb
                        pdb.set_trace()
                    self.sticks[s_del][t_del][begin_ind:end_ind] = new_stick


                if new_t == len(self.table_counts[s]):
                    #import pdb
                    #pdb.set_trace()
                    self.change_locations[ind] = (-1, -1)

                else:
                    self.change_locations[ind] = (new_s, new_t)
                    insert_ind = bisect_right(self.table_change_inds[new_s][new_t], ind)
                    self.table_change_inds[new_s][new_t].insert(insert_ind, ind)
                    # Draw the beta backward
                    begin_ind = self.get_last_switch(new_s, new_t, ind)
                    end_ind = self.get_next_switch(new_s, new_t, ind)
                    
                    try:
                        new_stick = self.draw_local_beta(degree_mats[new_s][new_t, ind+1:end_ind].sum(), 
                                                s_mats[new_s][new_t, ind+1:end_ind].sum(), self.theta_s[new_s])
                    except IndexError:
                        import pdb
                        pdb.set_trace()

                    self.sticks[new_s][new_t][ind+1:end_ind] = new_stick

                    new_stick = self.draw_local_beta(degree_mats[new_s][new_t, begin_ind:ind+1].sum(), 
                                                s_mats[new_s][new_t, begin_ind:ind+1].sum(), self.theta_s[new_s])
                    self.sticks[new_s][new_t][begin_ind:ind+1] = new_stick

        for s in range(num_senders):
            for table in range(len(self.table_counts[s])):
                #draw beta
                end_ind = self.get_next_switch(s, table, 0)

                new_stick = self.draw_local_beta(degree_mats[s][table,:end_ind].sum(),
                                        s_mats[s][table,:end_ind].sum(), self.theta_s[s])
                self.sticks[s][table][:end_ind] = new_stick

        return 


    def get_next_switch(self, s, i, ind):
        after_ind = bisect_right(self.table_change_inds[s][i], ind)
        if after_ind == len(self.table_change_inds[s][i]):
            return None
        return self.table_change_inds[s][i][after_ind]
        

    def get_last_switch(self, s, i, ind):
        before_ind = bisect_left(self.table_change_inds[s][i], ind)
        if before_ind == 0:
            return 0
        elif before_ind == len(self.table_change_inds[s][i]):
            before_ind = -1

        return self.table_change_inds[s][i][before_ind] + 1
            

    def draw_local_beta(self, d, s, theta):
        return np.random.beta(d + 1, s + theta)


    def sample_interarrival_times(self):
        pass


def get_limits_and_means_different_times(gibbs_dir, num_chains, num_iters_per_chain, 
                                        stick_name='stick_avgs.pkl', prob_name='prob_avgs.pkl'):

    times = []

    save_dirs = [os.path.join(gibbs_dir, '{}'.format(i)) for i in range(num_chains)]
    tp_master_list = []
    for save_dir in save_dirs:
        for i in range(int(num_iters_per_chain / 2)):
            save_path = os.path.join(save_dir, '{}.pkl'.format(i))
            with open(save_path, 'rb') as infile:
                params = pickle.load(infile)
                times.append(params.change_times)
                param_master_list.append(params)
    
    times.append(params.created_rec_times)
    times = np.concatenate(times)
    times = np.unique(times)
    times.sort()
    
    num_times = len(times)
    num_recs = len(params['global_sticks'])
    num_senders = len(params['table_counts'])
    means = {s: np.zeros((num_times, num_recs)) for s in range(num_senders)}
    medians = {s: np.zeros((num_times, num_recs)) for s in range(num_senders)}
    upper_limits = {s: np.zeros((num_times, num_recs)) for s in range(num_senders)}
    lower_limits = {s: np.zeros((num_times, num_recs)) for s in range(num_senders)}


    for s in range(num_senders):
        for params in param_list:
            print(s)
            table_probs = np.array(params['sticks'][s])
            table_probs[1:, :] = table_probs[1:, :] * np.cumprod(1 - table_probs[:-1, :], axis=0)
            rec_probs = np.array([table_probs[receiver_inds[s][r][:-1]].sum(axis=0) 
                                    for r in range(num_recs)])

            master_inds = np.digitize(times, params['change_times'], right=False) - 1

            rec_prob_list.append(rec_probs[:, master_inds])

        rec_prob_array = np.array(rec_prob_list)
        del rec_prob_list

        upper_limits = np.percentile(rec_prob_array, 97.5, axis=-1)
        lower_limits = np.percentile(rec_prob_array, 2.5, axis=-1)
        means = stick_array.mean(axis=-1)
        medians = np.median(stick_array, axis=-1)


        with open(os.path.join(gibbs_dir, stick_name), 'wb') as outfile:
            pickle.dump({'means': means,
                         'upper_limits': upper_limits,
                         'lower_limits': lower_limits,
                         'medians': medians}, outfile)


        probs = np.ones((means.shape[0], means.shape[1] + 1))
        probs[:, :-1] = means
        probs[:, 1:] = probs[:, 1:] * (np.cumprod(1 - means, axis=1))


        probs_ll = np.ones((upper_limits.shape[0], upper_limits.shape[1] + 1))
        probs_ul = np.ones((upper_limits.shape[0], upper_limits.shape[1] + 1))

        probs_ll[:, :-1] = lower_limits
        probs_ll[:, 1:] = probs_ll[:, 1:] * (np.cumprod(1 - upper_limits, axis=1))

        probs_ul[:, :-1] = upper_limits
        probs_ul[:, 1:] = probs_ul[:, 1:] * (np.cumprod(1 - lower_limits, axis=1))

        with open(os.path.join(gibbs_dir, prob_name), 'wb') as outfile:
            pickle.dump({'times': times,
                         'means': probs,
                         'upper_limits': probs_ul,
                         'lower_limits': probs_ll,
                         'medians': medians}, outfile)

    return (upper_limits, lower_limits, means), (probs_ul, probs_ll, probs)

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