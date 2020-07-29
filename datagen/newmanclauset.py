from sklearn.base import BaseEstimator
from sklearn.utils import check_array
import numpy as np
import pdb
import scipy.optimize as opt
import pandas as pd
from collections import defaultdict
from numpy.random import poisson, choice
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns
import powerlaw as pl
import functools
from .choice_fns import get_agent, get_agent2


class NewmanClausetPower(object):
    def __init__(self, discrete=False,
                 xmin=None, xmax=None,
                 verbose=False,
                 fit_method='Likelihood',
                 estimate_discrete=True,
                 discrete_approximation='round',
                 sigma_threshold=None,
                 parameter_range=None,
                 fit_optimizer=None,
                 xmin_distance='D',
                 **kwargs):

        self.discrete = discrete
        self.xmin = xmin
        self.xmax = xmax
        self.verbose = verbose
        self.fit_method = fit_method
        self.estimate_discrete = estimate_discrete
        self.discrete_approximation = discrete_approximation
        self.sigma_threshold = sigma_threshold
        self.parameter_range = parameter_range
        self.fit_optimizer = fit_optimizer
        self.xmin_distance = xmin_distance


    def fit(self, data):
        #if isinstance(data, 'SharedHollywoodNetwork'):
         #   data = data.data

        results = pl.Fit(data, discrete=self.discrete,
                        xmin=self.xmin, xmax=self.xmax,
                        verbose=self.verbose,
                        fit_method=self.fit_method,
                        estimate_discrete=self.estimate_discrete,
                        discrete_approximation=self.discrete_approximation,
                        sigma_threshold=self.sigma_threshold,
                        parameter_range=self.parameter_range,
                        fit_optimizer=self.fit_optimizer,
                        xmin_distance=self.xmin_distance)

        data = np.asarray(data)
        self.alpha = results.power_law.alpha
        self.xmin = results.power_law.xmin
        self.xmax = np.max(data)
        self.n_above_min = data[data >= self.xmin].shape[0]
        self.n = len(data)

    def pmf(self, x, alpha=None, xmin=None):
        if xmin is None:
            xmin = self.xmin
        if alpha is None:
            alpha = self.alpha

            return (alpha - 1 ) / xmin * (x / xmin) ** (-alpha)

    def plot_pmf(self, xmin=None, xmax=None, ax=None, prop=None, **kwargs):
        if xmin is None or xmin <= self.xmin:
            xmin = self.xmin

        if xmax is None:
            xmax = self.xmax

        if prop is None:
            prop = self.n_above_min / self.n

        pmf_vals = self.pmf(np.array([xmin, xmax]))


        if not ax:
            fig, ax = plt.subplots()
            ax.loglog(np.array([xmin, xmax]), prop * pmf_vals, **kwargs)
        else:
            fig = ax.get_figure()
            ax.plot(np.array([xmin, xmax]), prop * pmf_vals, **kwargs)
        return fig, ax

