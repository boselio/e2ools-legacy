import numpy as np
from ..models.teem import TemporalProbabilities
from . import pitman_yor_sticks as pys
from collections import defaultdict


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
    change_times = [np.random.exponential(1/nu)]
    while True:
        itime = np.random.exponential(1/nu)
        if change_times[-1] + itime > max_time:
            break
        else:
            change_times.append(change_times[-1] + itime)

    if senders is None:
        sender_sticks, senders = pys.pitman_yor_sticks(alpha_s, theta_s, 
                                                    num_interactions, 
                                                    single_int(1))



    t = 0
    #This will hold sticks for all sender distributions, and all "tables"
    #under the sender distribution.
    #First index is sender, second index is the 
    current_local_sticks = defaultdict(list)
    local_labels = defaultdict(list)
    current_local_probabilities = defaultdict(lambda: np.array([1.0]))
    created_local_sticks = defaultdict(list)
    created_local_times = defaultdict(list)

    #Keeping track of changes in the sticks
    drawn_sticks = []
    locations = []

    #Constant upper level statistics
    upper_level_sticks = []
    upper_level_probabilities = np.array([1.0])

    interactions = []
    change_ind = 0
    interaction_ind = 0
    num_receivers = 0
    created_times = defaultdict(list)
    num_senders = 0

    table_counts = defaultdict(list)
    while t < max_time:

        if (change_ind < len(change_times)) and (change_times[change_ind] < interaction_times[interaction_ind]):
            #Choose a receiver to change
            #Change a sender distribution to change
            change_sender = np.random.choice(num_senders)
            change_table = np.random.choice(len(current_local_probabilities[change_sender]), p=current_local_probabilities[change_sender])
            if change_table == len(current_local_sticks[change_sender]):
                #Change nothing, in the mass.
                drawn_sticks.append(-1)
                locations.append((change_sender, -1))
            else:
                new_stick = draw_stick(0, theta_local, 0)
                drawn_sticks.append(new_stick)
                locations.append((change_sender, change_table))

                #Accounting for current state
                current_local_sticks[change_sender][change_table] = new_stick
                current_local_probabilities[change_sender] = np.array(current_local_sticks[change_sender] + [1])
                current_local_probabilities[change_sender][1:] = current_local_probabilities[change_sender][1:] * np.cumprod(1 - current_local_probabilities[change_sender][:-1])
            #Keep track of the inds and the time
            t = change_times[change_ind]
            change_ind += 1
        else:
            s = senders[interaction_ind][0]
            interaction = [interaction_times[interaction_ind], s, []]

            if s == num_senders:
                num_senders += 1

            for _ in range(next(num_recs_per_interaction)):
                table = np.random.choice(len(current_local_probabilities[s]), p=current_local_probabilities[s])
                if table == len(current_local_sticks[s]):
                    #Draw a receiver from upper dist.
                    new_label = np.random.choice(len(upper_level_probabilities), 
                                                        p=upper_level_probabilities)
                    if new_label == len(upper_level_sticks):
                        #Draw a new receiver
                        new_stick = draw_stick(alpha, theta, new_label+1)
                        upper_level_sticks.append(new_stick)
                        upper_level_probabilities = np.array(upper_level_sticks + [1])
                        upper_level_probabilities[1:] = upper_level_probabilities[1:] * np.cumprod(1 - upper_level_probabilities[:-1])
                    
                    local_labels[s].append(new_label)
                    created_local_times[s].append(interaction_times[interaction_ind])
                    new_stick = draw_stick(0, theta_local, 0)
                    current_local_sticks[s].append(new_stick)
                    created_local_sticks[s].append(new_stick)

                    current_local_probabilities[s] = np.array(current_local_sticks[s] + [1])
                    current_local_probabilities[s][1:] = current_local_probabilities[s][1:] * np.cumprod(1 - current_local_probabilities[s][:-1])
                    table_counts[s].append(0)

                interaction[2].append(local_labels[s][table])

                table_counts[s][table] += 1

            interactions.append(interaction)

            t = interaction_times[interaction_ind]
            interaction_ind += 1

    return (interactions, drawn_sticks, locations, created_local_times, 
        created_local_sticks, local_labels, upper_level_sticks, change_times, table_counts)   

