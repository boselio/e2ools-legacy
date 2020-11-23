import numpy as np
from ..models.teem import TemporalProbabilities

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


def create_temporal_e2_data(alpha=0.1, theta=10, num_interactions=1000, delta=10, nu=1, max_receivers=20,
                            interaction_save_file=None, parameter_save_file=None):

    # Find the interaction times
    interaction_interarrival_times = np.random.exponential(1.0 / delta, size=num_interactions - 1)
    interaction_times = np.cumsum(interaction_interarrival_times)

    receiver_interarrival_inds = []
    current_sticks = []
    all_stick_samples = []
    interarrival_times = []

    interactions = []
    num_receivers = 0
    arrival_times = []
    # First interaction needs to be done before this loop
    interactions.append([0.0, []])
    current_probabilities = np.array([1.0])
    created_times = []

    for j in range(np.random.randint(1, max_receivers)):

        receiver = np.random.choice(num_receivers + 1, p=current_probabilities)

        # Check for new receiver
        if receiver == num_receivers:
            num_receivers += 1

            # time created
            created_times.append(0.0)
            # Generate stick
            current_sticks.append(draw_stick(alpha, theta, receiver + 1))

            all_stick_samples.append(current_sticks[-1])
            # Add to current probabilities
            current_probabilities[-1] = current_sticks[-1] * np.prod([1 - s for s in current_sticks[:-1]])

            # Generate interarrival time for new receiver
            itime = np.random.exponential(1 / (nu * current_probabilities[-1]))

            interarrival_times.append(itime)
            receiver_interarrival_inds.append(receiver)
            # Update arrival times
            arrival_times.append(itime)

            # Add on new receiever probability
            current_probabilities = np.concatenate([current_probabilities,
                                                    [1 - np.sum(current_probabilities)]])

        interactions[0][1].append(receiver)

    arrival_times = np.array(arrival_times)

    for i, interaction_time in enumerate(interaction_times):

        # Do this while time is before the next interaction time
        current_probabilities = np.array(current_sticks)
        current_probabilities[1:] *= np.cumprod(1 - current_probabilities)[:-1]
        while True:
            if np.all(arrival_times > interaction_time):
                break

            # find the min
            next_index = int(np.argmin(arrival_times))


            # new stick
            current_sticks[next_index] = draw_stick(alpha, theta, next_index + 1)

            # update probabilities
            current_probabilities = np.array(current_sticks)
            current_probabilities[1:] *= np.cumprod(1 - current_probabilities)[:-1]
            all_stick_samples.append(current_sticks[next_index])

            # Get new interarrival time
            itime = np.random.exponential(1 / (nu * current_probabilities[next_index]))

            # Update arrival times
            arrival_times[next_index] += itime

            interarrival_times.append(itime)
            receiver_interarrival_inds.append(next_index)
        # interaction time

        interactions.append([interaction_time, []])
        current_probabilities = np.concatenate([current_probabilities,
                                                [1 - np.sum(current_probabilities)]])

        for j in range(np.random.randint(1, max_receivers)):

            receiver = np.random.choice(num_receivers + 1, p=current_probabilities)

            # Check for new receiever
            if receiver == num_receivers:
                num_receivers += 1

                # created times
                created_times.append(interaction_time)
                # Generate stick
                current_sticks.append(draw_stick(alpha, theta, receiver + 1))

                all_stick_samples.append(current_sticks[-1])
                # Add to current probabilities
                current_probabilities[-1] = current_sticks[-1] * np.prod([1 - s for s in current_sticks[:-1]])

                # Generate interarrival time for new receiver
                itime = np.random.exponential(1 / (nu * current_probabilities[-1]))

                interarrival_times.append(itime)
                receiver_interarrival_inds.append(receiver)

                # Update arrival times
                arrival_times = np.concatenate([arrival_times, [interaction_time + itime]])
                # Add on new receiever probability
                current_probabilities = np.concatenate([current_probabilities,
                                                        [1 - np.sum(current_probabilities)]])

            interactions[i + 1][1].append(receiver)

    interarrival_times = np.array(interarrival_times)
    all_stick_samples = np.array(all_stick_samples)
    receiver_interarrival_inds = np.array(receiver_interarrival_inds)
    created_times = np.array(created_times)

    temporal_probs = TemporalProbabilities(all_stick_samples, interarrival_times, receiver_interarrival_inds,
                                           created_times)
    if interaction_save_file is not None:
        save_interactions(interactions, interaction_save_file)

    if parameter_save_file is not None:
        temporal_probs.save(parameter_save_file)

    return interactions, temporal_probs


def create_temporal_e2_data_v2(alpha=0.1, theta=10, num_interactions=1000, delta=10, nu=1, num_recs_per_interaction=None):

    if num_recs_per_interaction is None:
        def random_gen():
            while True:
                yield np.random.randint(1,11)
        num_recs_per_interaction = random_gen()
        
    else:
        try:
            num_recs = int(num_recs_per_interaction)
            def single_int(num):
                while True:
                    yield num
            num_recs_per_interaction = single_int(num_recs)
        except TypeError:
            continue

    # Find the interaction times
    interaction_interarrival_times = np.random.exponential(1.0 / delta, size=num_interactions)
    interaction_times = np.cumsum(interaction_interarrival_times)

    max_time = interaction_times[-1]
    change_times = [np.random.exponential(1/nu)]
    while True:
        itime = np.random.exponential(1/nu)
        if change_times[-1] + itime > max_time:
            break
        else:
            change_times.append(change_times[-1] + itime)


    t = 0
    current_sticks = []
    current_probabilities = np.array([1.0])
    all_stick_samples = []
    receivers = []
    created_sticks = []

    interactions = []
    change_ind = 0
    interaction_ind = 0
    num_receivers = 0
    created_times = []
    while t < max_time:

        if (change_ind < len(change_times)) and (change_times[change_ind] < interaction_times[interaction_ind]):
            #Choose a receiver to change
            change_receiver = np.random.choice(num_receivers+1, p=current_probabilities)
            if change_receiver == num_receivers:
                #Change nothing, in the mass.
                all_stick_samples.append(-1)
                receivers.append(-1)
            else:
                new_stick = draw_stick(alpha, theta, change_receiver+1)
                all_stick_samples.append(new_stick)
                receivers.append(change_receiver)

                #Accounting for current state
                current_sticks[change_receiver] = new_stick
                current_probabilities = np.array(current_sticks + [1])
                current_probabilities[1:] = current_probabilities[1:] * np.cumprod(1 - current_probabilities[:-1])
            #Keep track of the inds and the time
            t = change_times[change_ind]
            change_ind += 1
        else:
            interaction = [interaction_times[interaction_ind], []]

            for _ in range(next(num_recs_per_interaction)):
                receiver = np.random.choice(num_receivers + 1, p=current_probabilities)
                if receiver == num_receivers:
                    created_times.append(interaction_times[interaction_ind])
                    #New receiver
                    new_stick = draw_stick(alpha, theta, receiver+1)
                    current_sticks.append(new_stick)
                    created_sticks.append(new_stick)
                    current_probabilities = np.array(current_sticks + [1])
                    current_probabilities[1:] = current_probabilities[1:] * np.cumprod(1 - current_probabilities[:-1])
                    num_receivers += 1

                interaction[1].append(receiver)

            interactions.append(interaction)

            t = interaction_times[interaction_ind]
            interaction_ind += 1

    return interactions, all_stick_samples, receivers, created_times, created_sticks, change_times

