# encoding: utf-8
# cython: profile=True


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
