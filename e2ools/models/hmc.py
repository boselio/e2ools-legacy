import numpy as np


def hamiltonian_monte_carlo(n_samples, negative_log_prob, dVdq, initial_position, num_steps=10,
                            step_size=0.01, scale=None):
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
        #print('Sample {} completed'.format(len(samples) - 1))

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
    #import pdb
    #pdb.set_trace()
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