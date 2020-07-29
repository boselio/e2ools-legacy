from scipy.special import logit, logsumexp, expit, beta
import numpy as np
from collections import defaultdict, Counter
import bisect
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import pickle
from copy import deepcopy
from bisect import bisect_right, bisect_left


class TemporalProbabilities:
    def __init__(self, sticks, interarrival_times, receiver_inds, created_times):
        self.stick_dict = defaultdict(list)
        self.arrival_times_dict = defaultdict(list)
        self.itimes_dict = defaultdict(list)
        self.created_times = np.array(created_times)
        for r, ct in enumerate(created_times):
            self.arrival_times_dict[r].append(ct)

        for s, it, r in zip(sticks, interarrival_times, receiver_inds):
            self.arrival_times_dict[r].append(self.arrival_times_dict[r][-1] + it)
            self.stick_dict[r].append(s)
            self.itimes_dict[r].append(it)

    def get_stick(self, r, t, return_index=False):
        index = bisect.bisect_right(self.arrival_times_dict[r], t) - 1
        if index == len(self.arrival_times_dict[r]) - 1:
            index = index - 1
        if return_index:
            return self.stick_dict[r][index], index
        else:
            return self.stick_dict[r][index]

    def get_probability(self, r, t):
        prob = self.get_stick(r, t)
        for j in range(r):
            prob = prob * (1 - self.get_stick(j, t))
        return prob

    def change_itime(self, r, i, itime):
        self.arrival_times_dict[r][i:] = [at - self.itimes_dict[r][i] + itime
                                          for at in self.arrival_times_dict[r][i:]]
        self.itimes_dict[r][i] = itime

    def change_stick(self, r, i, stick):
        self.stick_dict[r][i] = stick

    def get_created_recs(self, t1, t2):
        return list(np.nonzero(np.logical_and(self.created_times >= t1, self.created_times <= t2))[0])

    def update(self, r, sticks, itimes):
        created_time = self.arrival_times_dict[r][0]
        del self.arrival_times_dict[r]
        del self.itimes_dict[r]
        del self.stick_dict[r]

        self.arrival_times_dict[r].append(created_time)
        for s, it, in zip(sticks, itimes):
            self.arrival_times_dict[r].append(self.arrival_times_dict[r][-1] + it)
            self.stick_dict[r].append(s)
            self.itimes_dict[r].append(it)

    def get_receiver_stick_trace(self, r, upper_limit=None):
        x = np.repeat(self.arrival_times_dict[r], 2)[1:-1]
        y = np.repeat(self.stick_dict[r], 2)

        if upper_limit is not None:
            x[x > upper_limit] = upper_limit

        return x, y

    def plot_receiver_stick_trace(self, r, upper_limit=None, ax=None, plot_args={}):
        if ax is None:
            _, ax = plt.subplots()

        x, y = self.get_receiver_stick_trace(r, upper_limit=upper_limit)

        ax.plot(x, y, **plot_args)
        ax.set_title('Receiver Sticks')
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')

        return ax

    def get_receiver_probability_trace(self, r, upper_limit=None):
        # Times to test
        x = []
        for i in range(r + 1):
            x.extend(self.arrival_times_dict[i])
        x.sort()
        y = np.array([self.get_probability(r, t) for t in x])

        x = np.repeat(x, 2)[1:]
        y = np.repeat(y, 2)[:-1]

        if upper_limit is not None:
            x[x > upper_limit] = upper_limit

        return x, y

    def plot_receiver_probability_trace(self, r, ax=None, upper_limit=None, plot_args={}):

        if ax is None:
            _, ax = plt.subplots()

        x, y = self.get_receiver_probability_trace(r, upper_limit=upper_limit)

        ax.plot(x, y, **plot_args)
        ax.set_title('Receiver Probabilities')
        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')

        return ax

    def save(self, save_file):
        with open(save_file, 'wb') as outfile:
            pickle.dump(self, outfile)

    def vars_to_flattened_array(self):
        num_receivers = len(self.stick_dict)
        flattened_array = []
        for r in range(num_receivers):
            flattened_array.extend(self.stick_dict[r])
            flattened_array.extend(self.itimes_dict[r])

        return np.array(flattened_array)


class TemporalProbabilitiesV2(TemporalProbabilities):
    def __init__(self, sticks, receivers, created_times, created_sticks, change_times):
        self.stick_dict = defaultdict(list)
        self.arrival_times_dict = defaultdict(list)
        self.created_times = np.array(created_times)
        for r, (ct, s) in enumerate(zip(created_times, created_sticks)):
            self.arrival_times_dict[r].append(ct)
            self.stick_dict[r].append(s)

        for s, r, ct in zip(sticks, receivers, change_times):
            if r == -1:
                continue
            self.arrival_times_dict[r].append(ct)
            self.stick_dict[r].append(s)

    def get_receiver_stick_trace(self, r, upper_limit):
        x = np.repeat(self.arrival_times_dict[r], 2)[1:]
        x = np.concatenate([x, [upper_limit]])
        y = np.repeat(self.stick_dict[r], 2)

        return x, y

    def get_receiver_probability_trace(self, r, upper_limit):
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

    def get_stick(self, r, t, return_index=False):
        if t < self.created_times[r]:
            index = -1
            s = 0
        else:
            index = bisect.bisect_right(self.arrival_times_dict[r], t) - 1
            s = self.stick_dict[r][index]

        if return_index:
            return s, index
        else:
            return s

    def insert_change(self, r, t, s):
        insert_index = bisect_right(self.arrival_times_dict[r], t)
        self.arrival_times_dict[r].insert(insert_index, t)
        self.stick_dict[r].insert(insert_index, s)
        return

    def delete_change(self, r, t):
        delete_index = self.arrival_times_dict[r].index(t)
        del self.arrival_times_dict[r][delete_index]
        del self.stick_dict[r][delete_index]

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

def smart_split(r, temporal_probs, interactions, alpha, theta, nu):
    sticks = temporal_probs.stick_dict[r]
    itimes = temporal_probs.itimes_dict[r]
    # Evaluate per-draw likelihood of each interval.
    ll_array = get_per_interval_ll(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu)

    # Choose according to inverse of likelihood.
    #TODO: use likelihood per number of draws?
    probs = np.exp(ll_array - logsumexp(ll_array))
    split_pt = np.random.choice(probs.shape[0], p=probs)

    u1 = np.random.rand()
    u2 = np.random.rand()

    itimes_proposal = itimes.copy()
    sticks_proposal = sticks.copy()


    # Add correction factor for the transformation (log det jacobian)

    itimes_proposal.insert(split_pt + 1, u1 * itimes_proposal[split_pt])
    itimes_proposal[split_pt] = (1 - u1) * itimes_proposal[split_pt]
    sticks_proposal.insert(split_pt + 1, min(2 * u2 * sticks_proposal[split_pt], 0.99))
    sticks_proposal[split_pt] = min(2 * (1 - u2) * sticks_proposal[split_pt], 0.99)

    #likelihood calculations
    log_accept = log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)
    log_accept -= log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)

    #dumb merge on top
    log_accept += np.log(1 / ll_array.shape[0])

    #smart split on bottom
    log_accept -= np.log(probs[split_pt])

    #Jacobian
    log_accept +=  np.log(4 * itimes[split_pt] * sticks[split_pt])

    log_accept = min(log_accept, 0)

    if np.random.rand() < min(1, np.exp(log_accept)):
        temporal_probs.update(r, sticks_proposal, itimes_proposal)
        return np.exp(log_accept), True, temporal_probs
    else:
        return np.exp(log_accept), False, temporal_probs


def calculate_smart_split_probs(r, sticks, itimes, temporal_probs, interactions, alpha, theta, nu):
    ll_array = get_per_interval_ll(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu)

    # Choose according to inverse of likelihood.
    # TODO: use likelihood per number of draws?
    return np.exp(ll_array - logsumexp(ll_array))


def calculate_smart_merge_probs(r, sticks, itimes, temporal_probs, interactions, alpha, theta, nu):
    num_breaks = len(itimes)
    ll_array = np.zeros(num_breaks - 1)


    for i in range(num_breaks - 1):
        itimes_proposal = itimes.copy()
        sticks_proposal = sticks.copy()

        itimes_proposal[i] += itimes_proposal[i + 1]
        sticks_proposal[i] = (sticks_proposal[i] + sticks_proposal[i + 1]) / 2
        del itimes_proposal[i + 1]
        del sticks_proposal[i + 1]

        ll = log_likelihood_per_r(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu)

        ll_array[i] = ll

    # Choose proportional to the resulting likelihoods
    return np.exp(ll_array - logsumexp(ll_array))


def smart_merge(r, temporal_probs, interactions, alpha, theta, nu):
    sticks = temporal_probs.stick_dict[r]
    itimes = temporal_probs.itimes_dict[r]

    # Choose proportional to the resulting likelihoods
    num_breaks = len(itimes) - 1
    probs = calculate_smart_merge_probs(r, sticks, itimes, temporal_probs, interactions, alpha, theta, nu)
    merge_pt = np.random.choice(num_breaks, p=probs)

    itimes_proposal = itimes.copy()
    sticks_proposal = sticks.copy()
    itimes_proposal[merge_pt] += itimes_proposal[merge_pt + 1]
    sticks_proposal[merge_pt] = (sticks_proposal[merge_pt] + sticks_proposal[merge_pt + 1]) / 2
    del itimes_proposal[merge_pt + 1]
    del sticks_proposal[merge_pt + 1]

    #likelihood calculations
    log_accept = log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)
    log_accept -= log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)

    #dumb split on top
    log_accept += np.log(1 / num_breaks)

    #smart merge on bottom
    log_accept -= np.log(probs[merge_pt])

    #Jacobian of transformation
    log_accept +=  np.log(0.5 / (itimes[merge_pt] + itimes[merge_pt + 1]) / (sticks[merge_pt] + sticks[merge_pt + 1]))

    log_accept = min(log_accept, 0)

    if np.random.rand() < min(1, np.exp(log_accept)):
        temporal_probs.update(r, sticks_proposal, itimes_proposal)
        return np.exp(log_accept), True, temporal_probs
    else:
        return np.exp(log_accept), False, temporal_probs


def dumb_split(r, temporal_probs, interactions, alpha, theta, nu):
    sticks = temporal_probs.stick_dict[r]
    itimes = temporal_probs.itimes_dict[r]
    split_pt = np.random.choice(len(sticks))

    u1 = np.random.rand()
    u2 = np.random.rand()

    itimes_proposal = itimes.copy()
    sticks_proposal = sticks.copy()

    itimes_proposal.insert(split_pt + 1, u1 * itimes_proposal[split_pt])
    itimes_proposal[split_pt] = (1 - u1) * itimes_proposal[split_pt]
    sticks_proposal.insert(split_pt + 1, min(2 * u2 * sticks_proposal[split_pt], 0.99))
    sticks_proposal[split_pt] = min(2 * (1 - u2) * sticks_proposal[split_pt], 0.99)

    #likelihood calculations
    log_accept = log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)
    log_accept -= log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)

    #smart merge on top
    probs = calculate_smart_merge_probs(r, sticks_proposal, itimes_proposal, temporal_probs, interactions, alpha, theta, nu)
    log_accept += np.log(probs[split_pt])

    #dumb split on bottom
    log_accept -= np.log(len(sticks))

    #Jacobian
    log_accept +=  np.log(4 * itimes[split_pt] * sticks[split_pt])

    log_accept = min(log_accept, 0)

    if np.random.rand() < min(1, np.exp(log_accept)):
        temporal_probs.update(r, sticks_proposal, itimes_proposal)
        return np.exp(log_accept), True, temporal_probs
    else:
        return np.exp(log_accept), False, temporal_probs


def dumb_merge(r, temporal_probs, interactions, alpha, theta, nu):
    # Random choice of which ones to aggregate
    sticks = temporal_probs.stick_dict[r]
    itimes = temporal_probs.itimes_dict[r]
    merge_pt = np.random.choice(len(sticks) - 1)

    itimes_proposal = itimes.copy()
    sticks_proposal = sticks.copy()
    itimes_proposal[merge_pt] += itimes_proposal[merge_pt + 1]
    sticks_proposal[merge_pt] = (sticks_proposal[merge_pt] + sticks_proposal[merge_pt + 1]) / 2
    del itimes_proposal[merge_pt + 1]
    del sticks_proposal[merge_pt + 1]

    #likelihood calculations
    log_accept = log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)
    log_accept -= log_likelihood_per_r(r, temporal_probs, interactions, sticks_proposal, itimes_proposal, alpha, theta, nu)

    #smart split on top
    probs = calculate_smart_split_probs(r, sticks_proposal, itimes_proposal, temporal_probs, interactions, alpha, theta, nu)

    if probs[merge_pt] == 0:
        return np.exp(log_accept), False, temporal_probs

    log_accept += np.log(probs[merge_pt])

    #dumb merge on bottom
    log_accept -= np.log(1 / (len(sticks) - 1))

    #Jacobian of transformation
    log_accept +=  np.log(0.5 / (itimes[merge_pt] + itimes[merge_pt + 1]) / (sticks[merge_pt] + sticks[merge_pt + 1]))

    log_accept = min(log_accept, 0)

    if np.random.rand() < min(1, np.exp(log_accept)):
        temporal_probs.update(r, sticks_proposal, itimes_proposal)
        return np.exp(log_accept), True, temporal_probs
    else:
        return np.exp(log_accept), False, temporal_probs


def initialize_random(interactions, temporal_probs, created_times, nu):
    temporal_probs = deepcopy(temporal_probs)
    #Get average interaction frequency
    total_recs = [r for [t, recs] in interactions for r in recs]
    unique_recs, rec_freq = np.unique(total_recs, return_counts=True)

    average_freq = rec_freq / np.sum(rec_freq)
    average_sticks = np.ones_like(average_freq)

    average_sticks[0] = average_freq[0]
    average_sticks[1:] = average_freq[1:] / np.cumprod(1 - average_freq)[:-1]
    final_time = interactions[-1][0]

    for r in temporal_probs.stick_dict.keys():
        #Get number of jumps
        num_jumps = len(temporal_probs.itimes_dict[r])
        while True:
            itimes = np.random.exponential(1 / (nu * average_freq[r]), size=num_jumps)
            if (itimes[:-1].sum() + created_times[r] < final_time) and (itimes.sum() + created_times[r] > final_time):
                break

        sticks = average_sticks[r] + np.random.randn(num_jumps) * 0.01
        sticks[sticks <= 0] = 0.01
        temporal_probs.update(r, sticks, itimes)

    return temporal_probs


def langevin_update_per_r(r, interactions, temporal_probs, tau, alpha, theta, nu):
    # Check to see what type of nu was passed; a vector or number
    if isinstance(tau, list):
        tau_itimes = tau[0]
        tau_sticks = tau[1]
    else:
        tau_itimes = tau
        tau_sticks = tau

    itimes = np.array(temporal_probs.itimes_dict[r])
    sticks = np.array(temporal_probs.stick_dict[r])
    tsticks = np.log(sticks / (1 - sticks))

    itimes = np.array(itimes)
    tstick_grad = get_grad_tsticks(r, temporal_probs, interactions, tsticks, itimes, alpha, theta, nu)

    tstick_proposal = tsticks + tau_sticks * tstick_grad
    tstick_proposal += np.sqrt(2 * tau_sticks) * np.random.randn(tsticks.shape[0])

    stick_proposal = 1 / (1 + np.exp(-tstick_proposal))

    itimes_grad = get_grad_itimes(r, temporal_probs, sticks, itimes, nu)
    itimes_proposal = itimes + tau_itimes * itimes_grad
    itimes_proposal += np.sqrt(2 * tau_itimes) * np.random.randn(itimes_proposal.shape[0])

    # Calculate alpha
    log_accept = log_likelihood_per_r_tstick(r, temporal_probs, interactions, tstick_proposal, itimes_proposal, alpha,
                                             theta, nu)
    log_accept = log_likelihood_per_r_tstick(r, temporal_probs, interactions, tstick_proposal, itimes_proposal, alpha,
                                             theta, nu)
    log_accept -= log_likelihood_per_r_tstick(r, temporal_probs, interactions, tsticks, itimes, alpha, theta, nu)

    new_tstick_grad = get_grad_tsticks(r, temporal_probs, interactions, tstick_proposal, itimes_proposal, alpha, theta,
                                       nu)
    new_itimes_grad = get_grad_itimes(r, temporal_probs, stick_proposal, itimes_proposal, nu)

    log_accept += mvn.logpdf(tsticks, mean=tstick_proposal - tau_sticks * new_tstick_grad,
                             cov=2 * tau_sticks * np.eye(tsticks.shape[0]))
    log_accept += mvn.logpdf(itimes, mean=itimes_proposal - tau_itimes * new_itimes_grad,
                             cov=2 * tau_sticks * np.eye(tsticks.shape[0]))

    log_accept -= mvn.logpdf(tstick_proposal, mean=tsticks - tau_sticks * tstick_grad,
                             cov=2 * tau_sticks * np.eye(tsticks.shape[0]))
    log_accept -= mvn.logpdf(itimes_proposal, mean=itimes - tau_itimes * itimes_grad,
                             cov=2 * tau_sticks * np.eye(tsticks.shape[0]))

    log_accept = min(0, log_accept)

    #Check to ensure that the itimes are all positive
    if np.any(itimes_proposal < 0):
        accepted = False
        return log_accept, accepted, 'negative', temporal_probs

    #Do you fall into the proper timing?
    final_time = interactions[-1][0]
    if itimes_proposal[:-1].sum() + temporal_probs.created_times[r] > final_time:
        accepted = False
        return log_accept, accepted, 'proper_timing', temporal_probs

    if itimes_proposal.sum() + temporal_probs.created_times[r] < final_time:
        accepted = False
        return log_accept, accepted, 'proper_timing', temporal_probs

    if np.log(np.random.rand()) < log_accept:
        temporal_probs.update(r, stick_proposal, itimes_proposal)
        accepted = True
    else:
        accepted = False

    return log_accept, accepted, 'in boundary', temporal_probs


def update_sticks_v2(tp_initial, change_times, recs_initial, interactions, alpha, theta):
    num_recs = len(set([r for t, recs in interactions for r in recs]))

    rec_choice = np.zeros_like(change_times)
    stick_choice = np.zeros_like(change_times)
    interaction_times = np.array([interaction[0] for interaction in interactions])
    max_time = interactions[-1][0]
    #created_set = set()

    permuted_inds = np.random.permutation(len(change_times))
    for ind in permuted_inds:
        ct = change_times[ind]
        #created_recs = np.where(tp_initial.created_times < ct)[0]

        #update_set = set(created_recs).difference(created_set)
        #created_set.update(update_set)
        #for r in list(update_set):
            #draw beta
        #    end_time = tp_initial.get_next_switch(r, tp_initial.created_times[r])
        #    if end_time == -1:
        #        end_time = max_time

        #    begin_ind = bisect_left(interaction_times, tp_initial.created_times[r])
        #    end_ind = bisect_right(interaction_times, end_time)
        #    recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
        #                              return_counts=True)
        #    degree_dict = dict(zip(recs, degrees))
        #    a = 1 - alpha + degree_dict[r] - 1
        #    b = theta + (r + 1) * alpha + np.sum([v for (k,v) in degree_dict.items() if k > r])
        #    tp_initial.stick_dict[r][0] = np.random.beta(a, b)


        probs = np.array([tp_initial.get_stick(r, ct) for r in range(num_recs)] + [1])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])
        new_choice = np.random.choice(num_recs+1, p=probs)
        rec_choice[ind] = new_choice

        if new_choice == recs_initial[ind]:
            #Draw the beta
            end_time = tp_initial.get_next_switch(new_choice, ct)
            if end_time == -1:
                end_time = max_time
            begin_ind = bisect_left(interaction_times, ct)
            end_ind = bisect_right(interaction_times, end_time)
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                      return_counts=True)
            degree_dict = dict(zip(recs, degrees))
            if new_choice not in degree_dict:
                degree_dict[new_choice] = 0
            a = 1 - alpha + degree_dict[new_choice]
            b = theta + (new_choice + 1) * alpha + np.sum([v for (k, v) in degree_dict.items() if k > new_choice])
            change_index = tp_initial.arrival_times_dict[new_choice].index(ct)
            tp_initial.stick_dict[new_choice][change_index] = np.random.beta(a, b)

        if recs_initial[ind] != -1:
            # Delete the current change
            r_delete = int(recs_initial[ind])
            tp_initial.delete_change(r_delete, ct)
            # redraw the beta that we had deleted.

            begin_time, change_ind = tp_initial.get_last_switch(r_delete, ct, return_index=True)
            end_time = tp_initial.get_next_switch(r_delete, ct)
            if end_time == -1:
                end_time = max_time

            begin_ind = bisect_left(interaction_times, begin_time)
            end_ind = bisect_right(interaction_times, end_time)
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                      return_counts=True)
            degree_dict = dict(zip(recs, degrees))
            if r_delete not in degree_dict:
                degree_dict[r_delete] = 0
            if begin_time == tp_initial.created_times[r_delete]:
                degree_dict[r_delete] -= 1
            a = 1 - alpha + degree_dict[r_delete]
            b = theta + (r_delete + 1) * alpha + np.sum([v for (k, v) in degree_dict.items() if k > r_delete])
            tp_initial.stick_dict[r_delete][change_ind] = np.random.beta(a, b)

        if new_choice == num_recs:
            rec_choice[ind] = -1
            stick_choice[ind] = -1
        else:
            # Draw the beta backward
            begin_time, change_ind = tp_initial.get_last_switch(new_choice, ct, return_index=True)
            begin_ind = bisect_left(interaction_times, begin_time)
            end_ind = bisect_right(interaction_times, ct)
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                      return_counts=True)
            degree_dict = dict(zip(recs, degrees))
            if new_choice not in degree_dict:
                degree_dict[new_choice] = 0
            if begin_time == tp_initial.created_times[new_choice]:
                degree_dict[new_choice] -= 1
            a = 1 - alpha + degree_dict[new_choice]
            b = theta + (new_choice + 1) * alpha + np.sum([v for (k, v) in degree_dict.items() if k > new_choice])
            try:
                tp_initial.stick_dict[new_choice][change_ind] = np.random.beta(a, b)
            except ValueError:
                print()
            #Draw the beta forward
            end_time = tp_initial.get_next_switch(new_choice, ct)
            if end_time == -1:
                end_time = max_time
            begin_ind = bisect_left(interaction_times, ct)
            end_ind = bisect_right(interaction_times, end_time)
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                      return_counts=True)
            degree_dict = dict(zip(recs, degrees))
            if new_choice not in degree_dict:
                degree_dict[new_choice] = 0
            a = 1 - alpha + degree_dict[new_choice]
            b = theta + (new_choice + 1) * alpha + np.sum([v for (k, v) in degree_dict.items() if k > new_choice])
            tp_initial.insert_change(new_choice, ct, np.random.beta(a, b))

    # Reupdate all the initial sticks, in case they did not get updated.
    for r in range(num_recs):
            #draw beta
        end_time = tp_initial.get_next_switch(r, tp_initial.created_times[r])
        if end_time == -1:
            end_time = max_time

        begin_ind = bisect_left(interaction_times, tp_initial.created_times[r])
        end_ind = bisect_right(interaction_times, end_time)
        recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                     return_counts=True)
        degree_dict = dict(zip(recs, degrees))
        a = 1 - alpha + degree_dict[r] - 1
        b = theta + (r + 1) * alpha + np.sum([v for (k,v) in degree_dict.items() if k > r])
        tp_initial.stick_dict[r][0] = np.random.beta(a, b)

    return tp_initial, rec_choice, stick_choice


def update_sticks_reverse(tp_initial, change_times, recs_initial, interactions, alpha, theta):
    num_recs = len(set([r for t, recs in interactions for r in recs]))

    rec_choice = np.zeros_like(change_times)
    stick_choice = np.zeros_like(change_times)
    interaction_times = np.array([interaction[0] for interaction in interactions])
    max_time = interactions[-1][0]
    max_times = np.zeros(num_recs)
    for interaction in interactions:
        for r in interaction:
            max_times[r] = interaction[0]

    created_set = set()
    for ind, ct in enumerate(change_times[::-1]):
        created_recs = np.where(tp_initial.created_times < ct)[0]

        update_set = set(created_recs).difference(created_set)
        created_set.update(update_set)
        for r in created_recs:
            #draw beta
            begin_time = tp_initial.get_last_switch(r, tp_initial.max_times[r])
            if begin_time == -1:
                begin_time = 0
            end_time = max_times[r]


            begin_ind = bisect_left(interaction_times, begin_time)
            end_ind = bisect_right(interaction_times, end_time)
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                      return_counts=True)
            degree_dict = dict(zip(recs, degrees))
            a = 1 - alpha + degree_dict[r] - 1
            b = theta + (r + 1) * alpha + np.sum([v for (k,v) in degree_dict.items() if k > r])
            tp_initial.stick_dict[r][0] = np.random.beta(a, b)


        probs = np.array([tp_initial.get_stick(r, ct) for r in range(num_recs)] + [1])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])
        new_choice = np.random.choice(num_recs+1, p=probs)
        rec_choice[ind] = new_choice

        if new_choice == recs_initial[ind]:
            continue

        if recs_initial[ind] != -1:
            # Delete the current change
            tp_initial.delete_change(recs_initial[ind], ct)
            # redraw the beta that we had deleted.
            r = recs_initial[ind]
            begin_time, change_ind = tp_initial.get_last_switch(r, ct, return_index=True)
            end_time = tp_initial.get_next_switch(r, ct)
            if end_time == -1:
                end_time = max_time

            begin_ind = bisect_left(interaction_times, begin_time)
            end_ind = bisect_right(interaction_times, end_time)
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                      return_counts=True)
            degree_dict = dict(zip(recs, degrees))
            if r not in degree_dict:
                degree_dict[r] = 0
            a = 1 - alpha + degree_dict[r]
            b = theta + (r + 1) * alpha + np.sum([v for (k, v) in degree_dict.items() if k > r])
            tp_initial.stick_dict[r][change_ind] = np.random.beta(a, b)

        if new_choice == num_recs:
            rec_choice[ind] = -1
            stick_choice[ind] = -1
        else:
            rec_choice[ind] = new_choice
            #Draw the beta
            end_time = tp_initial.get_next_switch(new_choice, ct)
            if end_time == -1:
                end_time = max_time
            begin_ind = bisect_left(interaction_times, ct)
            end_ind = bisect_right(interaction_times, end_time)
            recs, degrees = np.unique([r for interaction in interactions[begin_ind:end_ind] for r in interaction[1]],
                                      return_counts=True)
            degree_dict = dict(zip(recs, degrees))
            if new_choice not in degree_dict:
                degree_dict[new_choice] = 0
            a = 1 - alpha + degree_dict[new_choice]
            b = theta + (new_choice + 1) * alpha + np.sum([v for (k, v) in degree_dict.items() if k > new_choice])
            tp_initial.insert_change(new_choice, ct, np.random.beta(a, b))


    return tp_initial, rec_choice, stick_choice


def get_grad_tsticks(r, temporal_probs, interactions, tsticks, itimes, alpha, theta, nu):
    sticks = 1 / (1 + np.exp(-tsticks))
    stick_grad = get_grad_sticks(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu)
    # Chain rule for transformation
    tstick_grad = stick_grad * np.exp(-tsticks) / ((1 + np.exp(-tsticks)) ** 2)
    # Gradient of log Jacobian determinant
    tstick_grad = tstick_grad + (np.exp(-tsticks) - 1) / (1 + np.exp(-tsticks))

    return tstick_grad


def get_grad_sticks(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu):
    grad_s = np.zeros(sticks.shape)
    arrival_times = [temporal_probs.arrival_times_dict[r][0]]
    for it in itimes:
        arrival_times.append(arrival_times[-1] + it)

    for i, ((t1, t2), s) in enumerate(zip(zip(arrival_times[:-1], arrival_times[1:]), sticks)):
        created_recs = temporal_probs.get_created_recs(t1, t2)
        v = len([j for j in created_recs if r < j])

        interactions_sub = [inter for inter in interactions if
                            (inter[0] >= t1) and (inter[0] < t2)]
        inds, degrees = np.unique([j for inter in interactions_sub for j in inter[1]],
                                  return_counts=True)

        degree_dict = dict(zip(inds, degrees))
        if r in degree_dict:
            d = degree_dict[r]
        else:
            d = 0

        if t1 == arrival_times[0]:
            d -= 1

        sum_d = sum([d for ind, d in zip(inds, degrees) if ind > r])

        grad_s[i] = (d - alpha + 1) / s - (v + sum_d + theta + (r + 1) * alpha - 1) / (1 - s)
        temp_prod = np.prod([1 - temporal_probs.get_stick(j, t1) for j in range(0, r)])
        grad_s[i] -= nu * itimes[i] * temp_prod

    return grad_s


def get_grad_itimes(r, temporal_probs, sticks, itimes, nu):
    grad_t = np.zeros(itimes.shape)
    arrival_times = [temporal_probs.arrival_times_dict[r][0]]
    for it in itimes:
        arrival_times.append(arrival_times[-1] + it)

    for i, (t, s) in enumerate(zip(arrival_times[:-1], sticks)):
        temp_prod = np.prod([1 - temporal_probs.get_stick(j, t) for j in range(0, r)])
        grad_t[i] -= -nu * s * temp_prod

    return grad_t




def log_likelihood(tstick_dict, log_itimes_dict, created_times, interactions, alpha, theta, nu):
    num_recs = len(set([r for t, recs in interactions for r in recs]))

    stick_dict = defaultdict(list)
    itimes_dict = defaultdict(list)
    for k, v in tstick_dict.items():
        for s in v:
            stick_dict[k].append(expit(s))

    for k, v in log_itimes_dict.items():
        for lit in v:
            itimes_dict[k].append(np.exp(lit))

    ll = 0
    #Take care of the jacobian
    temp = np.array([i for v in tstick_dict.values() for i in v])
    ll += np.sum(-temp - 2 * np.log(1 + np.exp(-temp)))
    #for v in tstick_dict.values():
    #   ll += np.sum(-np.array(v) - 2 * np.log(1 + np.exp(-np.array(v))))

    ll += np.sum([i for v in log_itimes_dict.values() for i in v])
    arrival_times = []
    rec_inds = []
    sticks = []
    tsticks = []
    itimes = []
    for r in range(num_recs):
        arrival_times.append(created_times[r])
        rec_inds.append(r)
        for it, s, t in zip(itimes_dict[r], stick_dict[r], tstick_dict[r]):
            arrival_times.append(arrival_times[-1] + it)
            sticks.append(s)
            tsticks.append(t)
            rec_inds.append(r)
            itimes.append(it)

        sticks.append(-1)
        itimes.append(-1)
        tsticks.append(-1)

    arrival_times = np.array(arrival_times)
    rec_inds = np.array(rec_inds)
    sticks = np.array(sticks)
    tsticks = np.array(tsticks)
    itimes = np.array(itimes)


    sorted_inds = np.lexsort((rec_inds, arrival_times))
    sorted_arrival_times = arrival_times[sorted_inds]
    sorted_rec_inds = rec_inds[sorted_inds]
    sorted_sticks = sticks[sorted_inds]
    sorted_tsticks = tsticks[sorted_inds]
    sorted_itimes = itimes[sorted_inds]

    current_sticks = np.zeros(num_recs)
    current_tsticks = np.zeros(num_recs)
    max_stick = 0

    interaction_times = [inter[0] for inter in interactions]
    start_ind = 0
    for t1, t2, s, t, it, r in zip(sorted_arrival_times[:-1], sorted_arrival_times[1:], sorted_sticks, sorted_tsticks, sorted_itimes, sorted_rec_inds):

        max_stick = max(max_stick, r)

        if s == -1 or it == -1:
            if s != -1 or it != -1:
                raise ValueError()
            continue

        current_sticks[r] = s
        current_tsticks[r] = t
        if s == 0 or s == 1:
            print("Oh no something went wrong")
        assert s == expit(t)
        prob = s * np.prod(1 - current_sticks[:r])
        ll += np.log(nu * prob) - nu * prob * it

        # Add likelihood for the stick
        a = 1 - alpha
        b = theta + (r + 1) * alpha
        log_1s = -np.log(np.exp(t) + 1)
        log_s = t + log_1s

        ll += (a - 1) * log_s + (b - 1) * log_1s - np.log(beta(a, b))

        if t1 == t2:
            continue

        created_recs = np.where((created_times >= t1) & (created_times < t2))[0]

        if created_recs.shape[0] > 0 and np.max(created_recs) > max_stick:
            raise ValueError()

        #interactions_sub = [inter for inter in interactions if
        #                    (inter[0] >= t1) and (inter[0] < t2)]
        #end_ind = np.searchsorted(interaction_times, t2, 'right') - 1
        end_ind = bisect_left(interaction_times, t2)
        #interaction_inds = np.nonzero((interaction_times >= t1) & (interaction_times < t2))[0]

        #if len(interactions_sub) == 0:
        #    continue
        if end_ind - start_ind == 0:
            continue

        inds, degrees = np.unique([j for i in range(start_ind, end_ind) for j in interactions[i][1]],
                                  return_counts=True)
        #inds, degrees = np.unique([j for i in interactions_sub for j in i[1]],
        #                          return_counts=True)

        degree_array = np.zeros(max_stick+1)
        degree_array[inds] = degrees

        sum_d = np.cumsum(degree_array[::-1])[::-1]
        #For vertices that were created in the current time period, subtract 1.
        degree_array[created_recs] -= 1

        log_1s_array = -np.log(np.exp(current_tsticks) + 1)
        log_s_array = current_tsticks + log_1s_array
        ll += np.sum(degree_array * log_s_array[:max_stick+1])
        ll += np.sum(sum_d[1:] * log_1s_array[:max_stick])
        #print('log(s) {}'.format(log_s_array[:max_stick + 1]))
        #print('log(s) new{}'.format(np.log(current_sticks[:max_stick + 1])))
        #ll += np.sum(degree_array * np.log(current_sticks[:max_stick + 1]))
        #ll += np.sum(sum_d[1:] * np.log(1 - current_sticks[:max_stick]))
        #prob = s * np.prod(1 - current_sticks[:max_stick])
        #ll += np.log(nu * prob) - nu * prob * it
        start_ind = end_ind
    return ll

#def sample_sticks_beta_proposal(tstick_dict, log_itimes_dict, created_times, interactions, alpha, theta, nu):


def grad_log_likelihood(tstick_dict, log_itimes_dict, created_times, interactions, alpha, theta, nu):
    num_recs = len(set([r for t, recs in interactions for r in recs]))

    stick_dict = defaultdict(list)
    itimes_dict = defaultdict(list)
    for k, v in tstick_dict.items():
        for s in v:
            stick_dict[k].append(expit(s))

    for k, v in log_itimes_dict.items():
        for lit in v:
            itimes_dict[k].append(np.exp(lit))

    grad_tstick_dict = defaultdict(list)
    grad_log_itimes_dict = defaultdict(list)

    # Take care of the jacobian
    for k, v in tstick_dict.items():
        for tstick in v:
            grad_tstick_dict[k].append(2 / (1 + np.exp(-tstick)) * np.exp(-tstick) - 1)

    for k, v in log_itimes_dict.items():
        for lit in v:
            grad_log_itimes_dict[k].append(1)

    arrival_times = []
    rec_inds = []
    sticks = []
    itimes = []
    for r in range(num_recs):
        arrival_times.append(created_times[r])
        rec_inds.append(r)
        for it, s in zip(itimes_dict[r], stick_dict[r]):
            arrival_times.append(arrival_times[-1] + it)
            sticks.append(s)
            rec_inds.append(r)
            itimes.append(it)

        sticks.append(-1)
        itimes.append(-1)

    arrival_times = np.array(arrival_times)
    rec_inds = np.array(rec_inds)
    sticks = np.array(sticks)
    itimes = np.array(itimes)

    sorted_inds = np.lexsort((rec_inds, arrival_times))
    sorted_arrival_times = arrival_times[sorted_inds]
    sorted_rec_inds = rec_inds[sorted_inds]
    sorted_sticks = sticks[sorted_inds]
    sorted_itimes = itimes[sorted_inds]

    current_sticks = np.zeros(num_recs)
    max_stick = 0
    num_jumps = defaultdict(lambda: -1)

    for t1, t2, s, it, r in zip(sorted_arrival_times[:-1], sorted_arrival_times[1:], sorted_sticks, sorted_itimes,
                                sorted_rec_inds):
        max_stick = max(max_stick, r)

        if s == -1 or it == -1:
            if s != -1 or it != -1:
                raise ValueError()
            continue

        current_sticks[r] = s
        num_jumps[r] += 1
        prob = s * np.prod(1 - current_sticks[:r])
        grad_log_itimes_dict[r][num_jumps[r]] += - nu * prob * it

        # Add gradient for the stick
        a = 1 - alpha
        b = theta + (r + 1) * alpha
        grad_tstick_dict[r][num_jumps[r]] += ((a - 1) / s - (b - 1) / (1 - s)) * s * (1 - s)

        grad_tstick_dict[r][num_jumps[r]] += 1 / s * s * (1 - s)
        grad_tstick_dict[r][num_jumps[r]] += -prob / s * nu * it * s * (1 - s)
        for r_temp in range(r):
            s_temp = current_sticks[r_temp]
            grad_tstick_dict[r_temp][num_jumps[r_temp]] -= 1 / (1 - s_temp) * s_temp * (1 - s_temp)
            grad_tstick_dict[r_temp][num_jumps[r_temp]] += prob / (1 - s_temp) * nu * it * s_temp * (1 - s_temp)

        if t1 == t2:
            continue

        created_recs = np.where((created_times >= t1) & (created_times < t2))[0]

        if created_recs.shape[0] > 0 and np.max(created_recs) > max_stick:
            raise ValueError()

        interactions_sub = [inter for inter in interactions if
                            (inter[0] >= t1) and (inter[0] < t2)]
        if len(interactions_sub) == 0:
            continue

        inds, degrees = np.unique([j for inter in interactions_sub for j in inter[1]], return_counts=True)

        degree_array = np.zeros(max_stick + 1)
        degree_array[inds] = degrees

        sum_d = np.cumsum(degree_array[::-1])[::-1]
        # For vertices that were created in the current time period, subtract 1.
        degree_array[created_recs] -= 1

        try:
            for i in range(max_stick + 1):
                grad_tstick_dict[i][num_jumps[i]] += degree_array[i] * (1 - current_sticks[i])
            for i in range(max_stick):
                grad_tstick_dict[i][num_jumps[i]] -= sum_d[i + 1] * (current_sticks[i])
        except IndexError:
            print('uhoh')
        #ll += np.sum(degree_array * np.log(current_sticks[:max_stick + 1]))
        #ll += np.sum(sum_d[1:] * np.log(1 - current_sticks[:max_stick]))
        #print(ll)
        # prob = s * np.prod(1 - current_sticks[:max_stick])
        # ll += np.log(nu * prob) - nu * prob * it

    return grad_tstick_dict, grad_log_itimes_dict


def log_likelihood_interactions(degrees, sticks, alpha, created_recs):
    ll = np.sum(np.log(sticks[degrees > 0]) * (degrees[degrees > 0] - alpha))


    sum_d = np.cumsum(degrees)
    temp_prods = np.cumprod(1 - sticks)


def log_likelihood_per_r_tstick(r, temporal_probs, interactions, tsticks, itimes, alpha, theta, nu):
    sticks = 1 / (1 + np.exp(-tsticks))
    ll = log_likelihood_per_r(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu)
    # Adding the log determinant Jacobian
    ll = ll + np.sum(-tsticks - 2 * np.log(1 + np.exp(-tsticks)))
    return ll


def log_likelihood_per_r(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu):
    # Find arrival times for s
    log_likelihood = 0
    arrival_times = [temporal_probs.created_times[r]]
    for it in itimes:
        arrival_times.append(arrival_times[-1] + it)

    for i, ((t1, t2), s) in enumerate(zip(zip(arrival_times[:-1], arrival_times[1:]), sticks)):
        created_recs = temporal_probs.get_created_recs(t1, t2)
        v = len([j for j in created_recs if r < j])

        interactions_sub = [inter for inter in interactions if
                            (inter[0] >= t1) and (inter[0] < t2)]
        if len(interactions_sub) == 0:
            continue
        if interactions_sub[0][0] == t1:
            interactions_sub[0][1] = [j for j in interactions_sub[0][1] if r <= j]
        inds, degrees = np.unique([j for inter in interactions_sub for j in inter[1]],
                                  return_counts=True)

        degree_dict = dict(zip(inds, degrees))
        if r in degree_dict:
            d = degree_dict[r]
        else:
            d = 0

        if t1 == arrival_times[0]:
            d -= 1

        sum_d = sum([d for ind, d in zip(inds, degrees) if ind > r])

        temp_prod = np.prod([1 - temporal_probs.get_stick(j, t1) for j in range(0, r)])

        if d > 0:
            log_likelihood += (d - alpha) * np.log(s)

        log_likelihood += (v + sum_d + theta + (r + 1) * alpha - 1) * np.log(1 - s)
        log_likelihood += np.log(nu * s * temp_prod) - nu * s * temp_prod * (t2 - t1)
        log_likelihood -= np.log(beta(1 - alpha, theta + (r + 1) * alpha))
        #log_likelihood += np.log(s * temp_prod) - s * temp_prod * (t2 - t1)
    return log_likelihood


def get_per_interval_ll(r, temporal_probs, interactions, sticks, itimes, alpha, theta, nu):
    # Find arrival times for s
    ll_array = np.zeros_like(sticks)
    arrival_times = [temporal_probs.arrival_times_dict[r][0]]
    for it in itimes:
        arrival_times.append(arrival_times[-1] + it)

    for i, ((t1, t2), s) in enumerate(zip(zip(arrival_times[:-1], arrival_times[1:]), sticks)):
        created_recs = temporal_probs.get_created_recs(t1, t2)
        v = len([j for j in created_recs if r < j])

        interactions_sub = [inter for inter in interactions if
                            (inter[0] >= t1) and (inter[0] < t2)]
        if len(interactions_sub) == 0:
            continue
        if interactions_sub[0][0] == t1:
            interactions_sub[0][1] = [j for j in interactions_sub[0][1] if r <= j]
        inds, degrees = np.unique([j for inter in interactions_sub for j in inter[1]],
                                  return_counts=True)

        degree_dict = dict(zip(inds, degrees))
        if r in degree_dict:
            d = degree_dict[r]
        else:
            d = 0

        if t1 == arrival_times[0]:
            d -= 1

        sum_d = sum([d for ind, d in zip(inds, degrees) if ind > r])

        temp_prod = np.prod([1 - temporal_probs.get_stick(j, t1) for j in range(0, r)])

        if d > 0:
            ll_array[i] += (d - alpha) * np.log(s)
        ll_array[i] += (v + sum_d + theta + (r + 1) * alpha - 1) * np.log(1 - s)
        ll_array[i] += np.log(nu * s * temp_prod) - nu * s * temp_prod * (t2 - t1)

    return ll_array


def get_degree_log_likelihood(s, interactions, created_times, t1, t2, r):
    created_recs = np.nonzero(np.logical_and(created_times >= t1, created_times <= t2))[0]
    created_recs = list(created_recs)

    created_recs = [j for j in created_recs if r < j]
    v = len(created_recs)

    interactions_sub = [inter for inter in interactions if
                        (inter[0] >= t1) and (inter[0] < t2)]
    inds, degrees = np.unique([j for inter in interactions_sub for j in inter[1]],
                              return_counts=True)

    degree_dict = dict(zip(inds, degrees))
    if r in degree_dict:
        d = degree_dict[r]
    else:
        d = 0

    if t1 == created_times[r]:
        d -= 1

    sum_d = sum([d for ind, d in zip(inds, degrees) if ind > r])

    return d * np.log(s) + (v + sum_d) * np.log(1 - s)


def get_ll_itimes(r, sticks, interactions, probs, arrival_times, created_times, nu):
    ll = 0
    for i in range(len(arrival_times) - 1):
        t1 = arrival_times[i]
        t2 = arrival_times[i + 1]
        s = sticks[i]
        interactions_sub = [inter for inter in interactions if
                            (inter[0] >= t1) and (inter[0] <= t2)]
        inds, degrees = np.unique([j for inter in interactions_sub for j in inter[1]],
                                  return_counts=True)

        degree_dict = dict(zip(inds, degrees))
        if r in degree_dict:
            d = degree_dict[r]
        else:
            d = 0

        if t1 == created_times[r]:
            d -= 1

        sum_d = sum([d for ind, d in zip(inds, degrees) if ind > r])

        ll += d * np.log(s) + sum_d * np.log(1 - s)

    t1 = arrival_times[-1]
    s = sticks[-1]
    interactions_sub = [inter for inter in interactions if
                        (inter[0] >= t1)]
    inds, degrees = np.unique([j for inter in interactions_sub for j in inter[1]],
                              return_counts=True)

    degree_dict = dict(zip(inds, degrees))
    if r in degree_dict:
        d = degree_dict[r]
    else:
        d = 0

    sum_d = sum([d for ind, d in zip(inds, degrees) if ind > r])

    ll += d * np.log(s) + sum_d * np.log(1 - s)

    # Now the log likelihood for the interarrival times
    ll += np.sum(np.log(nu * probs[:-1]))
    ll -= nu * np.sum((arrival_times[1:] - arrival_times[:-1]) * probs[:-1])

    return ll


def log_prob(flattened_array, num_jumps, created_times, interactions, alpha, theta, nu):
    tstick_dict = {}
    itimes_dict = {}

    ind = 0

    for k in range(created_times.shape[0]):
        v = num_jumps[k]
        tstick_dict[k] = flattened_array[ind:ind+v]
        ind += v
        itimes_dict[k] = flattened_array[ind:ind+v]
        ind += v

    ll = log_likelihood(tstick_dict, itimes_dict, created_times, interactions, alpha, theta, nu)
    return ll


def negative_log_prob(flattened_array, num_jumps, created_times, interactions, alpha, theta, nu):
    return -log_prob(flattened_array, num_jumps, created_times, interactions, alpha, theta, nu)


def grad_log_prob(flattened_array, num_jumps, created_times, interactions, alpha, theta, nu):
    tstick_dict = {}
    itimes_dict = {}

    ind = 0

    for k in range(created_times.shape[0]):
        v = num_jumps[k]
        tstick_dict[k] = flattened_array[ind:ind+v]
        ind += v
        itimes_dict[k] = flattened_array[ind:ind+v]
        ind += v

    grad_tstick_dict, grad_itimes_dict = grad_log_likelihood(tstick_dict, itimes_dict, created_times, interactions,
                                                             alpha, theta, nu)

    grad_array = []
    for k in range(created_times.shape[0]):
        grad_array.extend(grad_tstick_dict[k])
        grad_array.extend(grad_itimes_dict[k])

    return np.array(grad_array)


def grad_negative_log_prob(flattened_array, num_jumps, created_times, interactions, alpha, theta, nu):
    return -grad_log_prob(flattened_array, num_jumps, created_times, interactions, alpha, theta, nu)


def hamiltonian_monte_carlo(n_samples, negative_log_prob, dVdq, initial_position, num_steps=10,
                            step_size=0.5, scale=None):
    """Run Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
    Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """

    if scale is None:
        scale = 1
    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    acceptance_rates = []
    accepted = []
    U = [negative_log_prob(samples[-1])]
    for p0 in momentum.rvs(size=size):
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog(
            samples[-1],
            scale * p0,
            dVdq,
            num_steps=num_steps,
            step_size=scale * step_size,
        )
        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(samples[-1]) + np.sum(p0**2) / 2
        new_log_p = negative_log_prob(q_new) + np.sum(p_new**2) / 2
        # pdb.set_trace()

        acceptance_rates.append(start_log_p - new_log_p)
        if np.log(np.random.rand()) < min(0, start_log_p - new_log_p):
            samples.append(q_new)
            accepted.append(True)
            U.append(new_log_p)
        else:
            samples.append(np.copy(samples[-1]))
            accepted.append(False)
            U.append(U[-1])
        print('Sample {} completed'.format(len(samples) - 1))

    return np.array(samples), accepted, acceptance_rates, U


def leapfrog(q, p, dVdq, num_steps, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)

    p -= step_size * dVdq(q) / 2  # half step
    for _ in range(num_steps - 1):

        q += step_size * p  # whole step
        p -= step_size * dVdq(q) #whole step
    q += step_size * p  # whole step
    p -= step_size * dVdq(q) / 2  # half step

    # q += step_size * p  # whole step
    # p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, -p


def hmc_discontinuous(n_samples, neg_log_prob, dVdq, continuous_vars, initial_position, num_steps=10,
                            step_size=0.5, scale=None):
    """Run Hamiltonian Monte Carlo sampling.

    Parameters
    ----------
    n_samples : int
    Number of samples to return
    negative_log_prob : callable
        The negative log probability to sample from
    initial_position : np.array
        A place to start sampling from.
    path_len : float
        How long each integration path is. Smaller is faster and more correlated.
    step_size : float
        How long each integration step is. Smaller is slower and more accurate.

    Returns
    -------
    np.array
        Array of length `n_samples`.
    """

    if scale is None:
        scale = 1
    # collect all our samples in a list
    samples = [initial_position]

    # Keep a single object for momentum resampling
    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    acceptance_rates = []
    accepted = []
    U = [neg_log_prob(samples[-1])]
    for i in range(n_samples):
        p0 = np.random.standard_normal(initial_position.shape[:1])
        p0[~continuous_vars] = np.random.laplace(size=p0[~continuous_vars].shape)
        # Integrate over our path to get a new position and momentum
        q_new, p_new = leapfrog_discontinuous(
            samples[-1],
            scale * p0,
            continuous_vars,
            neg_log_prob,
            dVdq,
            num_steps=num_steps,
            step_size=scale * step_size,
        )
        # Check Metropolis acceptance criterion
        start_log_p = neg_log_prob(samples[-1]) + np.sum(p0[continuous_vars]**2) / 2 + np.sum(np.abs(p0[~continuous_vars]))
        new_log_p = neg_log_prob(q_new) + np.sum(p_new[continuous_vars]**2) / 2 + np.sum(np.abs(p_new[~continuous_vars]))
        # pdb.set_trace()

        acceptance_rates.append(start_log_p - new_log_p)
        if np.log(np.random.rand()) < min(0, start_log_p - new_log_p):
            samples.append(q_new)
            accepted.append(True)
            U.append(new_log_p)
        else:
            samples.append(np.copy(samples[-1]))
            accepted.append(False)
            U.append(U[-1])
        print('Sample {} completed'.format(len(samples) - 1))

    return np.array(samples), accepted, acceptance_rates, U


def leapfrog_discontinuous(q, p, continuous_vars, neg_log_prob, dVdq, num_steps, step_size):
    """Leapfrog integrator for Hamiltonian Monte Carlo.

    Parameters
    ----------
    q : np.floatX
        Initial position
    p : np.floatX
        Initial momentum
    dVdq : callable
        Gradient of the velocity
    path_len : float
        How long to integrate for
    step_size : float
        How long each integration step should be

    Returns
    -------
    q, p : np.floatX, np.floatX
        New position and momentum
    """
    q, p = np.copy(q), np.copy(p)

    discrete_var_array = np.nonzero(~continuous_vars.astype('bool'))[0]
    for _ in range(num_steps - 1):
        p[continuous_vars] -= step_size * dVdq(q)[continuous_vars] / 2  # half step
        q[continuous_vars] += step_size * p[continuous_vars] / 2  # half step
        current_U = neg_log_prob(q)
        for j in np.random.permutation(discrete_var_array):
            q_star = q.copy()
            q_star[j] = q[j] + step_size * np.sign(p[j])
            U_star = neg_log_prob(q_star)
            delta_U = U_star - current_U
            if np.abs(p[j]) > delta_U:
                q[j] = q_star[j]
                p[j] = p[j] - np.sign(p[j]) * delta_U
                current_U = U_star
            else:
                p[j] = -p[j]
        q[continuous_vars] += step_size * p[continuous_vars] / 2
        p[continuous_vars] -= step_size * dVdq(q)[continuous_vars] / 2#half step

    #q += step_size * p  # whole step
    #p -= step_size * dVdq(q) / 2  # half step

    # q += step_size * p  # whole step
    # p -= step_size * dVdq(q) / 2  # half step

    # momentum flip at end
    return q, p


def mala(n_samples, log_prob, grad_log_prob, initial_position, step_size):

    samples = [initial_position]

    momentum = st.norm(0, 1)

    # If initial_position is a 10d vector and n_samples is 100, we want
    # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
    # iterate over rows
    size = (n_samples,) + initial_position.shape[:1]
    log_acceptance_rates = []
    accepted = []
    old_grad = grad_log_prob(initial_position)
    old_ll = log_prob(initial_position)
    for chi in momentum.rvs(size=size):
        new_sample = samples[-1].copy()
        new_sample += step_size * old_grad + (2 * step_size)**(0.5) * chi

        new_grad = grad_log_prob(new_sample)
        new_ll = log_prob(new_sample)
        log_acceptance_rate = new_ll - old_ll
        log_acceptance_rate -= 1 / (4 * step_size) * np.linalg.norm(samples[-1] - new_sample - step_size * new_grad)**2
        log_acceptance_rate += 1 / (4 * step_size) * np.linalg.norm(new_sample - samples[-1] - step_size * old_grad)**2

        log_acceptance_rates.append(log_acceptance_rate)

        if np.log(np.random.rand()) < min(0, log_acceptance_rate):
            samples.append(new_sample)
            old_grad = new_grad.copy()
            old_ll = new_ll.copy()
            accepted.append(True)
        else:
            samples.append(samples[-1].copy())
            accepted.append(False)

    return np.array(samples[1:]), accepted, log_acceptance_rates