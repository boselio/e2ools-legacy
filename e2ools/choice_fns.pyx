# encoding: utf-8
# cython: profile=True
#from libc.stdlib import rand, RAND_MAX

def get_agent(list unnormalized_probs, float normalizer, float rnd):
    rnd *= normalizer
    for ind, i in enumerate(unnormalized_probs):
        rnd -= i
        if rnd <= 0:
            return ind
    return -1

def get_agent2(list counts, int num_draws, double alpha, double theta, double rnd):
    rnd *= (num_draws + theta)
    cdef int i
    cdef int ind
    for ind, i in enumerate(counts):
        if ind == 0:
            rnd -= (i * alpha + theta)
        else:
            rnd -= (i - alpha)
        if rnd <= 0:
            return ind
    return ind

def get_agent3(int[:] counts, int num_draws, float alpha, float theta, float rnd):
    rnd *= (num_draws + theta)
    l = len(counts)
    for i in range(l):
        if i == 0:
            rnd = rnd - (counts[i] * alpha + theta)
        else:
            rnd = rnd - (counts[i] - alpha)
        if rnd <= 0:
            return i
    return i


def choice_discrete_unnormalized(list probs, double rand):
    cdef double prob_sum = 0
    cdef double running_total = 0
    cdef double i
    cdef int ind

    for i in probs:
        prob_sum += i

    rand *= prob_sum

    for ind, i in enumerate(probs):
        running_total += i
        if rand < running_total:
            return ind
    return ind
#from libcpp.vector cimport vector
#def get_agent2_here(list counts, int num_draws, float alpha, float theta, float rnd):
#    cdef vector[int] vv = counts
#    #rnd *= (num_draws + theta)

#    cdef int counter = 0
#    for i in vv:
#        if counter == 0:
#            rnd -= (i * alpha + theta)
#        else:
#            rnd -= (i - alpha)
#        if rnd <= 0:
#            return counter
#        counter = counter + 1
#    return counter
