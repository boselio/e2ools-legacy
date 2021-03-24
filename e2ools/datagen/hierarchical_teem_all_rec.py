import numpy as np
from ..models.teem import TemporalProbabilities
from . import pitman_yor_sticks as pys
from collections import defaultdict
from copy import copy

def save_interactions(interactions, file_name):

    with open(file_name, 'w') as outfile:
        for interaction in interactions:
            outline = '{} '.format(interaction[0])
            outline += ' '.join([str(i) for i in interactions[1]])
            outline += '\n'
            outfile.write(outline)


def load_interactions(file_name):

    interactions = []
    with open(file_name, 'r') as infile:
        for interaction_str in infile:
            temp_list = interaction_str.split()
            interactions.append([float(temp_list[0]), [int(i) for i in temp_list[1:]]])

    return interactions


def draw_stick(alpha, theta, i):
    if theta + alpha * i == 0:
        print('Warning: detected that theta + alpha * i = 0, assuming finite structure')
        return 1
    return np.random.beta(1 - alpha, theta + alpha * i)


def draw_more_labels(current_sticks, alpha, theta):
    while True:
        start_i = len(current_sticks)
        new_sticks = [draw_stick(alpha, theta, i) for i in range(start_i + 1, 2 * start_i + 1)]
        current_sticks.extend(new_sticks)
        probs = np.concatenate([current_sticks, [1]])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])
        p = probs[start_i:] / probs[start_i:].sum()
        proposed_label = np.random.choice(len(p), p=p) + start_i
        if proposed_label < len(current_sticks):
            break

    return current_sticks, proposed_label



def create_hierarchical_temporal_e2_data(alpha=0.1, theta=10, 
                                            num_interactions=1000, delta=10,
                                            alpha_s=0.1, theta_s=10,
                                            theta_local=10,
                                            nu=1, num_recs_per_interaction=None,
                                            senders=None):

    def single_int(num):
        while True:
            yield num

    if num_recs_per_interaction is None:
        def random_gen():
            while True:
                yield np.random.randint(1,11)
        num_recs_per_interaction = random_gen()

    else:
        try:
            #Test if it is a integer
            num_recs = int(num_recs_per_interaction)
            num_recs_per_interaction = single_int(num_recs)
        except TypeError:
            print('Expectation is that num_recs_per_interaction is a generator.')

    # Find the interaction times
    interaction_interarrival_times = np.random.exponential(1.0 / delta, size=num_interactions-1)
    interaction_times = np.concatenate([[0], np.cumsum(interaction_interarrival_times)])

    max_time = interaction_times[-1]

    temp = np.random.exponential(1/nu)
    if temp < max_time:
        change_times = [temp]

        while True:
            itime = np.random.exponential(1/nu)
            if change_times[-1] + itime > max_time:
                break
            else:
                change_times.append(change_times[-1] + itime)
    else:
        change_times = []
    if senders is None:
        sender_sticks, senders = pys.pitman_yor_sticks(alpha_s, theta_s, 
                                                    num_interactions, 
                                                    single_int(1))

    num_senders = len(np.unique([s[0] for s in senders]))

    #Constant upper level statistics
    upper_level_sticks = [draw_stick(alpha, theta, i) for i in range(1, 101)]
    upper_level_probabilities = np.concatenate([upper_level_sticks, [1]])
    upper_level_probabilities[1:] = upper_level_probabilities[1:] * np.cumprod(1 - upper_level_probabilities[:-1])

    t = 0
    #This will hold sticks for all sender distributions, and all "tables"
    #under the sender distribution.
    #First index is sender, second index is the 
    current_local_sticks = defaultdict(list)
    current_local_probabilities = defaultdict(lambda: np.array([1.0]))
    local_labels = defaultdict(list)

    table_counts = defaultdict(list)

    for s in range(num_senders):
        sticks = [draw_stick(0, theta_local, i) for i in range(1, 101)]
        probs = np.concatenate([sticks, [1]])
        probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])
        current_local_sticks[s] = sticks
        current_local_probabilities[s] = probs
        table_counts[s] = [0] * 100
        for _ in current_local_sticks[s]:
            proposed_label = np.random.choice(len(upper_level_probabilities), p=upper_level_probabilities)
            if proposed_label == len(upper_level_sticks):
                upper_level_sticks, proposed_label = draw_more_labels(upper_level_sticks, alpha, theta)
                upper_level_probabilities = np.concatenate([upper_level_sticks, [1]])
                upper_level_probabilities[1:] = upper_level_probabilities[1:] * np.cumprod(1 - upper_level_probabilities[:-1])
            local_labels[s].append(proposed_label)

    
    #Keeping track of changes in the sticks
    drawn_sticks = []
    table_locations = []
    rec_locations = []
    interactions = []
    change_ind = 0
    interaction_ind = 0
    num_receivers = 0
    created_times = defaultdict(list)

    created_local_sticks = copy(current_local_sticks)
    while t < max_time:

        if (change_ind < len(change_times)) and (change_times[change_ind] < interaction_times[interaction_ind]):
            #Choose a receiver to change
            #Change a sender distribution to change
            change_sender = np.random.choice(num_senders)
            num_recs = len(upper_level_sticks)
            temp = np.concatenate([local_labels[s], [-1]])
            rec_probs = np.array([current_local_probabilities[s][temp == r].sum() 
                for r in range(num_recs)])
            rec_probs = np.concatenate([rec_probs, [current_local_probabilities[s][-1]]])

            change_rec = np.random.choice(num_recs+1, p=rec_probs)

            if change_rec == num_recs:
                #Change nothing, in the mass.
                drawn_sticks.append([-1])
                table_locations.append([(-1, -1)])
                rec_locations.append((-1, -1))
            else:
                ds_list = []
                l_list = []
                for table in np.where(np.array(local_labels[change_sender]) == change_rec)[0]:
                    new_stick = draw_stick(0, theta_local, 0)

                    ds_list.append(new_stick)
                    l_list.append((change_sender, table))
                #Accounting for current state
                    current_local_sticks[change_sender][table] = new_stick
                    current_local_probabilities[change_sender] = np.array(current_local_sticks[change_sender] + [1])
                    current_local_probabilities[change_sender][1:] = current_local_probabilities[change_sender][1:] * np.cumprod(1 - current_local_probabilities[change_sender][:-1])
                
                drawn_sticks.append(ds_list)
                table_locations.append(l_list)
                rec_locations.append((change_sender, change_rec))
            #Keep track of the inds and the time
            t = change_times[change_ind]
            change_ind += 1
        else:
            s = senders[interaction_ind][0]
            interaction = [interaction_times[interaction_ind], s, []]

            for _ in range(next(num_recs_per_interaction)):
                table = np.random.choice(len(current_local_probabilities[s]), p=current_local_probabilities[s])
                if table == len(current_local_sticks[s]):
                    old_len = len(current_local_sticks[s])
                    current_local_sticks[s], table = draw_more_labels(current_local_sticks[s], 0, theta_local)
                    
                    for _ in range(len(current_local_sticks[s]) - old_len):
                    #Draw a receiver from upper dist.
                        proposed_label = np.random.choice(len(upper_level_probabilities), p=upper_level_probabilities)
                        if proposed_label == len(upper_level_probabilities):
                            upper_level_sticks, proposed_label = draw_more_labels(upper_level_sticks, alpha, theta)
                            upper_level_probabilities = np.concatenate([upper_level_sticks, [1]])
                            upper_level_probabilities[1:] = upper_level_probabilities[1:] * np.cumprod(1 - upper_level_probabilities[:-1])
                        local_labels[s].append(proposed_label)


                    current_local_probabilities[s] = np.array(current_local_sticks[s] + [1])
                    current_local_probabilities[s][1:] = current_local_probabilities[s][1:] * np.cumprod(1 - current_local_probabilities[s][:-1])
                    table_counts[s].extend([0] * (len(current_local_sticks[s]) - old_len))

                interaction[2].append(local_labels[s][table])

                table_counts[s][table] += 1


            interactions.append(interaction)

            t = interaction_times[interaction_ind]
            interaction_ind += 1

    return (interactions, created_local_sticks, drawn_sticks, table_locations, rec_locations, local_labels, 
        upper_level_sticks, change_times, table_counts)   

