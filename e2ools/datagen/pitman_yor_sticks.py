import numpy as np


def num_generator(i):
    while True:
        yield i

        
def pitman_yor_sticks(alpha, theta, num_interactions=1000, 
                        num_nodes_per_interaction=None, initial_sticks=None):
    if initial_sticks is None:
        sticks = np.array([])
    else:
        sticks = initial_sticks.copy()
    
    probs = probs = np.concatenate([sticks, [1.0]])
    probs[1:] = probs[1:] * np.cumprod(1 - probs[:-1])
    interactions = []

    for i in range(num_interactions):
        interaction = []
        for j in range(next(num_nodes_per_interaction)):
            try:
                new_rec = np.random.choice(len(probs), p=probs)
            except ValueError:
                import pdb
                pdb.set_trace()
            if new_rec == len(sticks):
                sticks = np.concatenate([sticks, [np.random.beta(1 - alpha, theta + (len(sticks) + 1) * alpha)]])
                probs[-1] = sticks[-1] * np.prod(1 - sticks[:-1])
                probs = np.concatenate([probs, [1 - probs.sum()]])
            
            interaction.append(new_rec)
                
        interactions.append(interaction)
        
    return sticks, interactions
        