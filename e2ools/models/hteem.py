import numpy as np
import dill as pickle

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


class HierarchicalTemporalProbabilities():
    def __init__(self, sticks, global_sticks, receiver_inds, change_times, locations,
                    created_sticks):
        self.stick_dict = sticks
        self.global_sticks = global_sticks
        self.arrival_times_dict = defaultdict(list)
        self.created_times = np.array(created_times)
        for r, (ct, s) in enumerate(zip(created_times, created_sticks)):
            self.arrival_times_dict[r].append(ct)
            self.stick_dict[r].append(s)

        for s, r, ct in zip(sticks, receivers, change_times):
            self.arrival_times_dict[r].append(ct)
            self.stick_dict[r].append(s)

    def get_receiver_stick_trace(self, r, upper_limit):
        x = np.array(self.arrival_times_dict[r])
        y = np.array(self.stick_dict[r])
        y = y[x <= upper_limit]
        x = x[x <= upper_limit]
        x = np.repeat(x, 2)[1:]
        y = np.repeat(y, 2)
        x = np.concatenate([x, [upper_limit]])

        return x, y

    def get_receiver_probability_trace(self, r, upper_limit):
        print("This is probably broken. Don't Use!")
        # Times to test
        x = []
        for i in range(r + 1):
            x.extend(self.arrival_times_dict[i])
        x.sort()
        y = np.array([self.get_probability(r, t) for t in x])


        x = np.repeat(x, 2)[1:]
        x = np.concatenate([x, [upper_limit]])
        y = np.repeat(y, 2)

        return x, y

    def get_probability_traces(self):
        times = [t for v in self.arrival_times_dict.values() for t in v]
        times = np.array(times)
        times.sort()

        num_times = len(times)
        num_recs = len(self.created_times)
        true_stick_array = np.zeros((num_times, num_recs))

        for r in range(num_recs):
            stick_list = []
            sticks_ind = np.digitize(times, self.arrival_times_dict[r], right=False) - 1
            sticks = np.array(self.stick_dict[r])[sticks_ind]
            sticks[times < self.created_times[r]] = 0
            true_stick_array[:, r] = sticks

        true_prob_array = true_stick_array.copy()
        true_prob_array[:, 1:] = true_prob_array[:, 1:] * np.cumprod(1 - true_prob_array[:, :-1], axis=1)

        return times, true_prob_array


    def get_stick(self, r, t, return_index=False):
        if t < self.created_times[r]:
            index = -1
            s = 1
        else:
            index = bisect.bisect_left(self.arrival_times_dict[r], t) - 1
            s = self.stick_dict[r][index]

        if return_index:
            return s, index
        else:
            return s

    def get_probability(self, r, t):
        if r != -1:
            prob = self.get_stick(r, t)
            for j in range(r):
                prob = prob * (1 - self.get_stick(j, t))
        else:
            num_recs = (self.created_times <= t).sum() 
            prob = np.prod([1 - self.get_stick(j, t) for j in range(num_recs)])
            
        return prob

    def insert_change(self, r, t, s):
        insert_index = bisect_right(self.arrival_times_dict[r], t)
        self.arrival_times_dict[r].insert(insert_index, t)
        self.stick_dict[r].insert(insert_index, s)
        return

    def delete_change(self, r, t):
        delete_index = self.arrival_times_dict[r].index(t)
        del self.arrival_times_dict[r][delete_index]
        del self.stick_dict[r][delete_index]

    def move_change(self, r, t, t_new):
        change_index = self.arrival_times_dict[r].index(t)
        self.arrival_times_dict[r][change_index] = t_new

    def get_last_switch(self, r, t, return_index=False):
        index = bisect_left(self.arrival_times_dict[r], t) - 1
        if return_index:
            return self.arrival_times_dict[r][index], index
        else:
            return self.arrival_times_dict[r][index]

    def get_next_switch(self, r, t, return_index=False):
        index = bisect_right(self.arrival_times_dict[r], t)
        if index == len(self.arrival_times_dict[r]):
            index == -1
            switch_time = -1
        else:
            switch_time = self.arrival_times_dict[r][index]
        if return_index:
            return switch_time, index
        else:
            return switch_time

    def get_arrival_times(self):
        num_recs = len(self.created_times)

        nodes, arrival_times = zip(*[(k, t) for (k, v) in self.arrival_times_dict.items()
                                for t in v[1:]])
        nodes = list(nodes)
        arrival_times = list(arrival_times)

        if len(self.arrival_times_dict[-1]) > 0:
            arrival_times.append(self.arrival_times_dict[-1][0])
            nodes.append(-1)

        arrival_times = np.array(arrival_times)
        sorted_inds = arrival_times.argsort()
        arrival_times = arrival_times[sorted_inds]
        nodes = np.array(nodes)[sorted_inds]

        return arrival_times, nodes

    def generate_trace_pdf(self, save_dir, interactions=None, r_list='all'):
        if interactions is not None:
            max_time = interactions[-1][0]
            unique_nodes, degrees = np.unique([i for interaction in 
                                               interactions for i in interaction[1]],
                                              return_counts=True)
        
        if r_list == 'all':
            r_list = np.arange(len(self.created_times))
        elif r_list[:3] == 'top':
            number_of_plots = int(r_list[3:])
            r_list = unique_nodes[np.argsort(degrees)[::-1][:number_of_plots]]

        num_pages = math.ceil(len(r_list) / 10)

        r_counter = 0

        data_color = 'C2'

        times, probabilities = self.get_probability_traces()
        with matplotlib.backends.backend_pdf.PdfPages(save_dir) as pdf:
            for p in range(num_pages):
                fig, axs = plt.subplots(5,2, figsize=(8.5, 11))
                for k in range(10):
                    r = r_list[r_counter]
                    i, j = np.unravel_index(k, [5, 2])
                    if i == 0 and j == 1:
                        #use these for the legend
                        true_label = 'Probabilities'
                        data_label = 'Data'
                    else:
                        true_label = None
                        confidence_label = None
                        mean_label = None

                    axs[i, j].plot(times, probabilities[:, r], color='k', 
                        linewidth=2, label=true_label)
                    
                    if interactions is not None:
                        plot_event_times(interactions, r, axs[i, j], color=data_color)

                    if i == 0 and j == 1:
                        axs[i, j].legend()
                    
                    axs[i, j].set_title('Receiver {}'.format(r))
                    r_counter += 1
                    if r_counter == len(r_list):
                        break

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        return


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

        

    def initialize_state(self, interactions, save_dir):

        #Initialize state variables
        ####Don't think I need this
        #Number of total tables across franchise serving each dish
        self.global_table_counts = np.array([])
        ####

        #Number of tables in each restaurant
        self.num_tables_in_s = defaultdict(int)
        #configuration: top level key is s (the restaurant),
        #lower level key is r (type of dish),
        #lowest level value is list of tables with # of customers.
        self.receiver_inds = defaultdict(lambda: defaultdict(lambda: np.array([-1], dtype='int')))

        #For temporal version, table counts now must be list of lists (or arrays)
        #with each entry of the lower list corresponding to the tables counts at
        #a particular jump point.
        self.table_counts = defaultdict(lambda: defaultdict(list))
        self.sticks = defaultdict(lambda: defaultdict(defaultdict(list)))

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


    def run_chain(self, save_dir, num_times, interactions, sample_parameters=True):

        self.initialize_state(interactions, save_dir)

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


    def _sample_jump_locations(self, tp_initial, interactions, alpha, theta):

        num_recs = len(set([r for t, s, recs in interactions for r in recs]))
        recs_initial, change_times = zip(*[(r, t) for (r, v) in tp_initial.arrival_times_dict.items() 
                                            for t in v[1:]])
        change_times = list(change_times)
        recs_initial = list(recs_initial)


        if len(tp_initial.arrival_times_dict[-1]) > 0:
            change_times.append(tp_initial.arrival_times_dict[-1][0])
            recs_initial.append(-1)

        change_times = np.array(change_times)
        sorted_inds = change_times.argsort()
        change_times = change_times[sorted_inds]
        recs_initial = np.array(recs_initial)[sorted_inds]

        rec_choice = np.zeros_like(change_times)
        stick_choice = np.zeros_like(change_times)
        interaction_times = np.array([interaction[0] for interaction in interactions])
        max_time = interactions[-1][0]
        #created_set = set()

        permuted_inds = np.random.permutation(len(change_times))
        
        # calculate all degrees between change times for all receivers
        degree_mat = np.zeros((num_recs, len(change_times) + 1))
        beta_mat = np.zeros((num_recs, len(change_times) + 1))

        for i, (begin_time, end_time) in enumerate(zip(np.concatenate([[0], change_times]), 
                                                np.concatenate([change_times, [interaction_times[-1] + 1]]))):

            begin_ind = bisect_left(interaction_times, begin_time)
            end_ind = bisect_right(interaction_times, end_time)
            if begin_ind == end_ind:
                continue
                
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                  return_counts=True)

            for r in recs:
                if begin_time >= tp_initial.created_times[r] and end_time <= tp_initial.created_times[r]:
                    degrees[recs == r] -= 1

            try:
                degree_mat[recs, i] = degrees
            except IndexError:
                import pdb
                pdb.set_trace()

            for r in range(num_recs):
                beta_mat[r, i] = tp_initial.get_stick(r, end_time)

        s_mat = np.vstack([np.flipud(np.cumsum(np.flipud(degree_mat), axis=0))[1:, :], 
               np.zeros((1, len(change_times)+1))])

        for ind in permuted_inds:
        #Need to calculate, the likelihood of each stick if that receiver
        #was not chosen.

            ct = change_times[ind]
            try:
                end_time = change_times[ind+1]
            except: end_time = interaction_times[-1] + 1

            for r in range(num_recs):
                beta_mat[r, ind+1] = tp_initial.get_stick(r, end_time)

            num_created_recs = len(tp_initial.created_times[tp_initial.created_times < ct])
            probs = np.array([tp_initial.get_stick(r, ct) for r in range(num_created_recs)] + [1])
            probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])

            log_probs = np.log(probs)
            #Calculate likelihood of each jump
            #First step, add integrated new beta
            log_probs[:-1] += betaln(1 - alpha + degree_mat[:num_created_recs, ind+1], 
                                theta + np.arange(1, num_created_recs+1) * alpha + s_mat[:num_created_recs, ind+1])
            
            #I think this next line is wrong.
            #log_probs[-1] += betaln(1 - alpha, 
            #                    theta + num_created_recs+1 * alpha)

            #Now, need to add all other likelihood components, i.e. all degrees for
            #which the receiver did not jump.
            likelihood_components = degree_mat[:num_created_recs, ind+1] * np.log(beta_mat[:num_created_recs, ind+1])
            likelihood_components += s_mat[:num_created_recs, ind+1] * np.log(1 - beta_mat[:num_created_recs, ind+1])

            log_probs[:-1] += np.sum(likelihood_components) - likelihood_components
            log_probs[-1] += np.sum(likelihood_components)

            probs = np.exp(log_probs - logsumexp(log_probs))

            new_choice = np.random.choice(num_created_recs+1, p=probs)
            rec_choice[ind] = new_choice
            if new_choice == recs_initial[ind]:
                if new_choice == num_created_recs:
                    #Do nothing, it stayed in the tail
                    continue
                else:
                    #Draw the beta
                    end_time = tp_initial.get_next_switch(new_choice, ct)
                    if end_time == -1:
                        end_time = max_time
                    begin_ind = bisect_left(interaction_times, ct)
                    end_ind = bisect_right(interaction_times, end_time)

                    new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, ct, alpha, theta, new_choice)

                    change_index = tp_initial.arrival_times_dict[new_choice].index(ct)
                    tp_initial.stick_dict[new_choice][change_index] = new_stick

            else:
                r_delete = int(recs_initial[ind])
                tp_initial.delete_change(r_delete, ct)
            
                if r_delete != -1:
                    # redraw the beta that we had deleted.
                    begin_time, change_ind = tp_initial.get_last_switch(r_delete, ct, return_index=True)
                    end_time = tp_initial.get_next_switch(r_delete, ct)
                    if end_time == -1:
                        end_time = max_time

                    begin_ind = bisect_left(interaction_times, begin_time)
                    end_ind = bisect_right(interaction_times, end_time)

                    new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, begin_time, alpha, theta, r_delete)

                    tp_initial.stick_dict[r_delete][change_ind] = new_stick

                if new_choice == num_created_recs:
                    rec_choice[ind] = -1
                    stick_choice[ind] = -1
                    tp_initial.insert_change(-1, ct, -1.0)

                else:
                    # Draw the beta backward
                    begin_time, change_ind = tp_initial.get_last_switch(new_choice, ct, return_index=True)
                    begin_ind = bisect_left(interaction_times, begin_time)
                    end_ind = bisect_right(interaction_times, ct)
                    
                    new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, begin_time, alpha, theta, new_choice)

                    tp_initial.stick_dict[new_choice][change_ind] = new_stick

                    #Draw the beta forward
                    end_time = tp_initial.get_next_switch(new_choice, ct)
                    if end_time == -1:
                        end_time = max_time
                    begin_ind = bisect_left(interaction_times, ct)
                    end_ind = bisect_right(interaction_times, end_time)

                    new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, ct, alpha, theta, new_choice)

                    tp_initial.insert_change(new_choice, ct, new_stick)

        # Reupdate all the initial sticks, in case they did not get updated.
        for r in range(num_recs):
                #draw beta
            end_time = tp_initial.get_next_switch(r, tp_initial.created_times[r])
            if end_time == -1:
                end_time = max_time

            begin_ind = bisect_left(interaction_times, tp_initial.created_times[r])
            end_ind = bisect_right(interaction_times, end_time)

            new_stick = draw_beta(interactions[begin_ind:end_ind], tp_initial, tp_initial.created_times[r], alpha, theta, r)

            tp_initial.stick_dict[r][0] = new_stick

        return tp_initial, rec_choice, stick_choice


    def _sample_table_configuration(self, interactions, initial=False):

        if initial:
            for t, s, interaction in interactions:
                for r in interaction:
                    self._add_customer(s, r)

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


    def _add_customer(self, s, r, cython_flag=True):
        if len(self.global_table_counts) == r:
            assert r == len(self.global_sticks)
            self.global_table_counts = np.append(self.global_table_counts, [1])
            #self.global_table_counts[r] += 1
            self.receiver_inds[s][r] = np.insert(self.receiver_inds[s][r], -1, self.num_tables_in_s[s])
            self.num_tables_in_s[s] += 1

            self.table_counts[s] = np.append(self.table_counts[s], [1])
            #Draw local stick
            self.sticks[s] = np.concatenate([self.sticks[s], [np.random.beta(1, self.theta_s[s])]])
            self.probs[s] = np.concatenate([self.sticks[s], [1]])
            self.probs[s][1:] = self.probs[s][1:] * np.cumprod(1 - self.probs[s][:-1])

            #Draw global stick
            self.global_sticks = np.append(self.global_sticks, [np.random.beta(1 - self.alpha, 
                            self.theta + (len(self.global_sticks) + 1) * self.alpha)])
            self.global_probs = np.concatenate([self.global_sticks, [1]])
            self.global_probs[1:] = self.global_probs[1:] * np.cumprod(1 - self.global_probs[:-1])
            return

        probs = self.probs[s][self.receiver_inds[s][r]]
        probs[-1] = probs[-1] * self.global_probs[r]
        probs = probs.tolist()

        table = choice_discrete_unnormalized(probs, np.random.rand())

        if table == len(probs)-1:
            self.receiver_inds[s][r]= np.insert(self.receiver_inds[s][r], -1, self.num_tables_in_s[s])

            #Draw stick
            self.sticks[s] = np.concatenate([self.sticks[s], [np.random.beta(1, self.theta_s[s])]])
            self.probs[s] = np.concatenate([self.sticks[s], [1]])
            self.probs[s][1:] = self.probs[s][1:] * np.cumprod(1 - self.probs[s][:-1])

            self.num_tables_in_s[s] += 1
            self.global_table_counts[r] += 1
            self.table_counts[s] = np.append(self.table_counts[s], [0])

        self.table_counts[s][self.receiver_inds[s][r][table]] += 1

    def _remove_customer(self, s, r, cython_flag=True):
        #Choose uniformly at random a customer to remove.
        try:
            remove_probs = self.table_counts[s][self.receiver_inds[s][r][:-1]].tolist()
        except IndexError:
            import pdb
            pdb.set_trace()

        table = choice_discrete_unnormalized(remove_probs, np.random.rand())
        
        ind = self.receiver_inds[s][r][table]
        self.table_counts[s][ind] -= 1
        if self.table_counts[s][ind] == 0:
            self.num_tables_in_s[s] -= 1
            self.global_table_counts[r] -= 1
            self.sticks[s] = np.concatenate([self.sticks[s][:ind], self.sticks[s][ind+1:]])
            self.probs[s] = np.concatenate([self.sticks[s], [1]])
            self.probs[s][1:] = self.probs[s][1:] * np.cumprod(1 - self.probs[s][:-1])
            self.table_counts[s] = np.concatenate([self.table_counts[s][:ind], self.table_counts[s][ind+1:]])
            self.receiver_inds[s][r] = np.concatenate([self.receiver_inds[s][r][:table],
                                                       self.receiver_inds[s][r][table+1:]])
            #Removed the ind table - so all tables greater than ind+1 -> ind
            for r in self.receiver_inds[s].keys():
                self.receiver_inds[s][r][self.receiver_inds[s][r] > ind] = self.receiver_inds[s][r][self.receiver_inds[s][r] > ind] - 1


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


