import numpy as np
import pdb

def poisson_gen(lam=1, offset=0):
    while True:
        yield np.random.poisson(lam) + offset
        
        
def generate_ddcrp(f, theta, num_edges=1000, rec_gen=None):
    
    
    p_list = []
    interaction_itimes = np.random.exponential(size=num_edges-1)
    interaction_times = np.concatenate([[0], np.cumsum(interaction_itimes)])
    
    interactions = []
    times = []
    labels = []
    if rec_gen is None:
        rec_gen = poisson_gen(offset=1)
    
    max_rec = 0
    for num_edge, it in enumerate(interaction_times):
        if num_edge % 1000 == 0:
            print(num_edge)
            
        num_recs = next(rec_gen)
        interaction = []
        distances = [it - t for t in times]
        discounted_degrees = np.zeros(max_rec+1)
        for v, d in zip(labels, distances):
            discounted_degrees[v] += f(d)
        
        discounted_degrees[max_rec] = theta
        
        for n in range(num_recs):
            #pdb.set_trace()
            probabilities = discounted_degrees / np.sum(discounted_degrees)
            new_rec = np.random.choice(len(discounted_degrees),
                                       p=probabilities)
            p_list.append(probabilities)
            
            if new_rec == max_rec:
                max_rec += 1
                discounted_degrees[-1] = 1
                discounted_degrees = np.concatenate([discounted_degrees,
                                                     [theta]])
            else:
                discounted_degrees[new_rec] += f(0.0)
                
            labels.append(new_rec)
            times.append(it)
            interaction.append(new_rec)
            
        interactions.append([it, interaction])
        
    return interactions, p_list


def exp_decay(d, sigma=0.01):
    
    d = np.array(d)
    flags = d >= 0
    answer = np.zeros_like(d)
    answer[flags] = np.exp(-sigma*d[flags])
    
    return answer


def rec_gen(i=1):
    while True:
        yield i