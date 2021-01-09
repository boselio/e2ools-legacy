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

from concurrent.futures import ProcessPoolExecutor


class HTEEM(): 
    def __init__(self, save_dir, nu, alpha=None, theta=None, theta_s=None, 
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
        self.global_table_counts = np.array([])
        ####
        self.change_times = change_times
        #Number of tables in each restaurant
        self.num_tables_in_s = defaultdict(int)
        #configuration: top level key is s (the restaurant),
        #lower level key is r (type of dish),
        #lowest level value is list of tables with # of customers.
        self.receiver_inds = defaultdict(lambda: defaultdict(lambda: np.array([-1], dtype='int')))

        #For temporal version, table counts now must be list of lists (or arrays)
        #with each entry of the lower list corresponding to the tables counts at
        #a particular jump point.
        self.table_counts = defaultdict(list)
        self.sticks = defaultdict(list)
        self.arrival_times = defaultdict(list)
        self.created_times = defaultdict(list)
        #Commenting this out beacause probs need to be generated on the fly
        #self.probs = defaultdict(lambda: np.array([1]))

        self.global_sticks = np.array([])
        self.global_probs = np.array([1])

        #Sender set
        self.s_set = set([interaction[1] for interaction in interactions])
        self.s_size = len(self.s_set)

        #Reciever set
        self.r_set = set([r for interaction in interactions for r in interaction[2]])
        self.r_size = len(self.r_set)

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
        self.missed_arrival_times = []


    def run_chain(self, save_dir, num_times, interactions, change_times=None,
                    sample_parameters=True):

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
        for ct in change_time:
            self.missed_arrival_times.append(ct)

        s_time = time.time()
        for t in range(num_times):
            if t % 100 == 0:
                print(t)

            self._sample_jump_locations(interactions)
            self._sample_table_configuration(interactions)

            params = {'alpha': self.alpha, 'theta': self.theta,
                        'theta_s': self.theta_s,
                        'global_counts': self.global_table_counts,
                        'sticks': self.sticks,
                        'receiver_inds': self.receiver_inds,
                        'global_sticks': self.global_sticks}


            if t >= num_times / 2:
                file_dir = save_dir / '{}.pkl'.format(t - int(num_times / 2))
                with file_dir.open('wb') as outfile:
                    pickle.dump(params, outfile)

        e_time = time.time()
        print(e_time - s_time)


    def _sample_table_configuration(self, interactions, initial=False):

        if initial:
            for t, s, interaction in interactions:
                for r in interaction:
                    self._add_customer(t, s, r)

        else:
            for t, s, interaction in interactions:
                for r in interaction:
                    #Remove a customer
                    self._remove_customer(s, r)

                    #Add a customer
                    self._add_customer(s, r)

            #Update local sticks
            for s in self.sticks.keys():
                reverse_counts = np.cumsum(self.table_counts[s][::-1])[::-1]
                reverse_counts = np.concatenate([reverse_counts[1:], [0]])
                a = 1 + self.table_counts[s]
                b = reverse_counts + self.theta_s[s]

                self.sticks[s] = np.random.beta(a, b)
                self.probs[s] = np.concatenate([self.sticks[s], [1]])
                self.probs[s][1:] = self.probs[s][1:] * np.cumprod(1 - self.probs[s][:-1])

            #Update global sticks
            reverse_counts = np.cumsum(self.global_table_counts[::-1])[::-1]
            reverse_counts = np.concatenate([reverse_counts[1:], [0]])
            a = 1 -self.alpha + self.global_table_counts
            b = reverse_counts + self.theta + np.arange(1, len(self.global_table_counts)+ 1) * self.alpha
            self.global_sticks = np.random.beta(a, b)
            self.global_probs = np.concatenate([self.global_sticks, [1]])
            self.global_probs[1:] = self.global_probs[1:] * np.cumprod(1 - self.global_probs[:-1])


    def insert_table(self, t, s, r):
        insert_point = bisect_right(self.created_times[s], t)
        for r in self.receiver_inds[s].keys():
            ii = self.receiver_inds[s][r] >= insert_point
            self.receiver_inds[s][r][ii] = self.receiver_inds[s][r][ii] + 1

        rec_insert_point = bisect_right([self.created_times[s][i] for i in self.receiver_inds[s][r][:-1]], t)
        self.receiver_inds[s][r] = np.insert(self.receiver_inds[s][r], rec_insert_point, insert_point)
        self.num_tables_in_s[s] += 1
        self.table_counts[s].insert(insert_point, np.zeros(len(self.change_times)))
        self.created_times[s].insert(insert_point, t)
        self.sticks[s].insert(insert_point, np.ones(len(self.change_times)) * np.random.beta(1, self.theta_s[s]))
        self.global_table_counts[r] += 1


    def _add_customer(self, t, s, r, cython_flag=True):
        if len(self.global_table_counts) == r:
            assert r == len(self.global_sticks)
            self.global_table_counts = np.append(self.global_table_counts, [1])
            #self.global_table_counts[r] += 1
            #This is noow wrong. Need to insert %after
            self.insert_table(t, s, r)

            #Draw global stick
            self.global_sticks = np.append(self.global_sticks, [np.random.beta(1 - self.alpha, 
                            self.theta + (len(self.global_sticks) + 1) * self.alpha)])
            self.global_probs = np.concatenate([self.global_sticks, [1]])
            self.global_probs[1:] = self.global_probs[1:] * np.cumprod(1 - self.global_probs[:-1])
            return

        probs, table_inds = self.get_unnormalized_probabilities(t, s, r)
        table = choice_discrete_unnormalized(probs, np.random.rand())

        if table == len(probs)-1:
            self.insert_table(t, s, r)
            
        table_ind = bisect_right(self.change_times, t) - 1
        self.table_counts[s][self.receiver_inds[s][r][table]][table_ind] += 1


    def get_unnormalized_probabilities(self, t, s, r):
        max_point = bisect_right(self.created_times[s], t)
        if max_point == 0:
            rec_probs = [1.0]
            rec_inds = []
            return  rec_probs, rec_inds
        
        sticks, inds = zip(*[self.get_stick(s, i, t, return_index=True) for i in range(max_point)])
        probs = np.concatenate([sticks, [1]])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])
        max_rec_point = bisect_right([self.created_times[s][i] for i in self.receiver_inds[s][r][:-1]], t)
        rec_probs = np.concatenate([probs[self.receiver_inds[s][r][:max_rec_point]], probs[-1:]])
        rec_probs[-1] = rec_probs[-1] * self.global_probs[r]
        rec_inds = [inds[i] for i in self.receiver_inds[s][r][:max_rec_point]]
    
        return rec_probs.tolist(), rec_inds


    def get_stick(self, s, i, t, return_index=False):
        if t < self.created_times[s][i]:
            index = -1
            s = 1
        else:
            index = bisect.bisect_right(self.change_times, t) - 1
            s = self.sticks[s][i][index]

        if return_index:
            return s, index
        else:
            return s


    def get_table_counts(self, s, i, t, return_index=False):
        if t < self.created_times[s][i]:
            index = -1
            s = 0
        else:
            index = bisect.bisect_right(self.change_times, t) - 1
            s = self.table_counts[s][i][index]

        if return_index:
            return s, index
        else:
            return s


    def _remove_customer(self, t, s, r, cython_flag=True):
        #Choose uniformly at random a customer to remove.
        remove_probs, remove_inds = zip(*[self.get_table_counts(s, i, t, return_index=True) 
                                            for i in self.receiver_inds[s][r][:-1]])

        table = choice_discrete_unnormalized(remove_probs, np.random.rand())
        
        ind = self.receiver_inds[s][r][table]
        self.table_counts[s][ind][remove_inds[table]] -= 1
        if sum(self.table_counts[s][ind]) == 0:
            self.num_tables_in_s[s] -= 1
            self.global_table_counts[r] -= 1
            self.sticks[s] = self.sticks[s][:ind] + self.sticks[s][ind+1:]
            self.table_counts[s] = self.table_counts[s][:ind] +  self.table_counts[s][ind+1:]
            self.receiver_inds[s][r] = np.concatenate([self.receiver_inds[s][r][:table],
                                                       self.receiver_inds[s][r][table+1:]])
            #Removed the ind table - so all tables greater than ind+1 -> ind
            for r in self.receiver_inds[s].keys():
                self.receiver_inds[s][r][self.receiver_inds[s][r] > ind] = self.receiver_inds[s][r][self.receiver_inds[s][r] > ind] - 1


    def _sample_jump_locations(self, interactions):

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

        for s in range(len(self.table_counts)):
            degree_mats[s] =  np.array(self.table_counts[s])
            s_mats[s] = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mats[s]), axis=0))[1:, :], 
                                                np.zeros((1, len(self.change_times)+1))])
            for i in range(len(self.table_counts[s])):
                begin_ind = bisect_right(self.change_times, self.created_times[s][i]) - 1
                degree_mats[s][i, begin_ind] -= 1




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

            likelihood_components = {}
            #Calculate likelihood of each jumps
            for s in range(num_senders):
                #Calculate log probs for all potential jumps, at the period BEFORE the jump
                num_created_tables[s] = len([i for i in self.created_times[s] if i < ct])
                probs[s] = np.array([self.get_stick(s, i, ct - 1e-8) for i in range(num_created_tables[s])] + [1])
                probs[s][1:] = probs[s][1:] * np.cumprod(1 - probs[s][:-1])
                log_probs[s] = np.log(probs[s])

                #Add integrated new beta using future table counts.
                log_probs[s][:-1] += betaln(1 + degree_mats[s][:num_created_recs, ind+1], 
                                    self.theta_s[s] + s_mats[s][:num_created_recs, ind+1])
            
                #Now, need to add all other likelihood components, i.e. all degrees for
                #which the receiver did not jump.
                likelihood_components[s] = degree_mats[s][:num_created_recs, ind+1] * np.log(self.sticks[s][:num_created_recs, ind+1])
                likelihood_components[s] += s_mats[s][:num_created_recs, ind+1] * np.log(1 - self.sticks[s][:num_created_recs, ind+1])

                likelihood_components[s] = degree_mats[s][:num_created_recs, ind+1] * np.log(self.sticks[s][:num_created_recs, ind])
                likelihood_components[s] += s_mats[s][:num_created_recs, ind+1] * np.log(1 - self.sticks[s][:num_created_recs, ind])

                log_probs[s][:-1] += np.sum(likelihood_components) - likelihood_components
                log_probs[s][-1] += np.sum(likelihood_components)

            log_prob = np.concatenate([log_probs[s] for s in range(num_senders)])
            probs = np.exp(log_prob - logsumexp(log_prob))

            new_choice = np.random.choice(num_created_tables+num_senders, p=probs)

            temp = np.cumsum([num_created_tables[i] + 1 for i in range(num_tables)])
            new_s = bisect_right(temp, new_choice)
            new_t = new_choice - temp[new_s]
            new_choice = (new_s, new_t)
            if new_choice == old_locs[ind]:
                if new_choice[1] == num_created_tables[new_s]:
                    #Do nothing, it stayed in the tail
                    continue
                else:
                    #Draw the beta
                    end_ind = self.get_next_switch(new_s, new_t, ct)
                    if end_ind == -1:
                        end_time = max_time
                    begin_ind = self.get_last_switch(new_s, new_t, ct)
                    end_ind = bisect_right(interaction_times, end_time)

                    new_stick = self.draw_beta(degrees_mats[new_s][new_t, ind+1:end_ind], 
                                                s_mats[new_s][new_t, ind+1:end_ind], self.alpha_s[new_s], 
                                                self.theta_s[new_s])
                    self.sticks[new_s][new_t][ind+1:end_ind] = new_stick

                    new_stick = self.draw_beta(degrees_mats[new_s][new_t, begin_ind:ind+1], 
                                                s_mats[new_s][new_t begin_ind:ind+1], self.alpha_s[new_s], 
                                                self.theta_s[new_s])
                    self.sticks[new_s][new_t][begin_ind:ind+1] = new_stick


            else:
                old_loc = old_locs[ind]
                s_del = old_loc[0]
                t_del = old_loc[1]
                if r_delete != -1:
                    # redraw the beta that we had deleted.
                    begin_ind = self.get_last_switch(s_del, t_del, ct)
                    end_ind = self.get_next_switch(s_del, t_del, ct)
                    if end_time == -1:
                        end_time = max_time

                    new_stick = self.draw_beta(degrees_mats[s_del][t_del, begin_ind:end_ind], 
                                                s_mats[s_del][t_del, begin_ind:end_ind], self.alpha_s[s_del], 
                                                self.theta_s[s_del])
                    self.sticks[s_del][t_del][begin_ind:end_ind] = new_stick


                if new_t == num_created_tables[s]:
                    self.change_locations[ind] = (new_s, -1)

                else:
                    # Draw the beta backward
                    begin_ind = self.get_last_switch(new_s, new_t, ct)
                    end_ind = self.get_next_switch(new_s, new_t, ct)
                    if end_ind == -1:
                        end_time = max_time
                    
                    new_stick = self.draw_beta(degrees_mats[new_s][new_t, ind+1:end_ind], 
                                                s_mats[new_s][new_t, ind+1:end_ind], self.alpha_s[new_s], 
                                                self.theta_s[new_s])
                    self.sticks[new_s][new_t][ind+1:end_ind] = new_stick

                    new_stick = self.draw_beta(degrees_mats[new_s][new_t, begin_ind:ind+1], 
                                                s_mats[new_s][new_t begin_ind:ind+1], self.alpha_s[new_s], 
                                                self.theta_s[new_s])
                    self.sticks[new_s][new_t][begin_ind:ind+1] = new_stick

        return 


    def get_next_switck(self, s, i, t):
        pass


    def get_last_switch(self, s, i, t):
        pass