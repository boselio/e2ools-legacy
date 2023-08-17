import numpy as np
from functools import partial
from collections import defaultdict


def draw_stick(alpha, theta, i):
    if theta + alpha * i == 0:
        print('Warning: detected that theta + alpha * i = 0, assuming finite structure')
        return 1
    return np.random.beta(1 - alpha, theta + alpha * i)

    
def save_interactions(interactions, file_name):

    with open(file_name, 'w') as outfile:
        for interaction in interactions:
            outline = '{} '.format(interaction[0])
            outline += ' '.join([str(i) for i in interaction[1]])
            outline += '\n'
            outfile.write(outline)


def jump_approx(x, stick_old, stick_new, jump, k=1):
    temp = (stick_new - stick_old)
    return temp * (0.5 + 0.5 * np.tanh(k * (x - jump))) + stick_old


def template(x, initial_stick, jump_times, jump_sticks, k=1):
    y = np.ones_like(x) * initial_stick
    old_sticks = np.concatenate([[initial_stick], jump_sticks[:-1]])
    for old_stick, new_stick, time in zip(old_sticks, jump_sticks, 
                                            jump_times):
        y += jump_approx(x, 0, new_stick - old_stick, time, k)

    return y

class SmoothTemporalProbabilities():
    def __init__(self, sticks, receivers, created_times, created_sticks, change_times, k):
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

        self.receiver_fn_dict = {}
        for r in range(len(created_sticks)):
            jump_times = np.array(change_times)[np.array(receivers==r)]
            jump_sticks = np.array(sticks)[np.array(receivers==r)]

            #import pdb
            #pdb.set_trace()
            self.receiver_fn_dict[r] = partial(template, initial_stick=created_sticks[r], 
                                                jump_times=jump_times, 
                                                jump_sticks=jump_sticks, 
                                                k=k)

    def get_receiver_stick_trace(self, r, upper_limit):
        x = np.linspace(0, upper_limit, num=10000)
        y = self.receiver_fn_dict[r](x)

        return x, y

    def get_receiver_probability(self, r, t):

        p = self.receiver_fn_dict[r](t)

        for s in range(r):
            p = p * (1 - self.receiver_fn_dict[s](t))

        return p 

    def sample_interactions(self, num_interactions=1000, max_receivers=20, delta=10):
        #Find time of interactions
        #For each time, get the sticks and probs
        #Sample the interactions
        
        interaction_interarrival_times = np.random.exponential(1.0 / delta, size=num_interactions - 1)
        interaction_times = np.concatenate([[0], np.cumsum(interaction_interarrival_times)])

        interactions = []
        num_receivers = len(self.created_times)
        for i, t in enumerate(interaction_times):
            if i % 100 == 0:
                print(i)
            receivers = []
            probabilities = np.array([self.get_receiver_probability(r, t) for r in range(num_receivers)])
            probabilities = np.concatenate([probabilities, [1 - probabilities.sum()]])
            for j in range(np.random.randint(1, max_receivers)):
                

                receiver = np.random.choice(num_receivers + 1, p=probabilities)
                receivers.append(receiver)


            interactions.append([t, receivers])

        return interactions

def smooth_teem(alpha=0.1, theta=10, nu=1, max_receiver=1e7, max_time=100, tol=1e-5):

    change_times = [np.random.exponential(1/nu)]
    while True:
        itime = np.random.exponential(1/nu)
        if change_times[-1] + itime > max_time:
            break
        else:
            change_times.append(change_times[-1] + itime)

    sticks = []
    total_prob = 0
    while total_prob < 1 - tol and len(sticks) < max_receiver:
        sticks.append(np.random.beta(1 - alpha, theta + (len(sticks) + 1) * alpha))
        total_prob += sticks[-1] * prod([1 - s for s in sticks[:-1]])

    sticks = np.array(sticks)
    created_sticks = sticks.copy()
    current_probabilities = sticks.copy()
    current_probabilities[1:] = current_probabilities[1:] * np.cumprod(1 - current_probabilities[:-1])

    stick_samples = []
    receivers = []

    for t in change_times:
        change_receiver = np.random.choice(num_receivers, p=current_probabilities)
        receivers.append(change_receiver)

        new_stick = draw_stick(alpha, theta, change_receiver + 1)
        stick_samples.append(new_stick)
        sticks[change_receiver] = new_stick

        current_probabilities = sticks.copy()
        current_probabilities[1:] = current_probabilities[1:] * np.cumprod(1 - current_probabilities[:-1])


    return stick_samples, receivers, created_sticks, change_times

        