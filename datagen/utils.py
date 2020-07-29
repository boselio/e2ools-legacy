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


def poisson_gen(lam=1, offset=0):
    while True:
        yield poisson(lam) + offset


def pairwise_only():
    while True:
        yield 2

class PY_process():

    def __init__(self, alpha, theta):
        self.alpha = alpha
        self.theta = theta
        self.degrees = defaultdict(int)
#        self.draws = []
        self.num_agents = 0
        self.num_draws = 0
        self.unnormalized_probs = [theta]
        self.normalizer = theta

    def fit(self, num_draws=10):
        raise NotImplementedError('Need to implement this for partial_fit')

    def partial_fit(self):
        return self._partial_fit()

    def _partial_fit(self):
        probs = np.array(self.unnormalized_probs) / self.normalizer
        draw = choice(self.num_agents + 1, p=probs)


        self.normalizer += 1
        self.num_draws += 1
        if draw == 0:
            self.degrees[self.num_agents] += 1
            self.num_agents += 1
            self.unnormalized_probs.append(1 - self.alpha)
            self.unnormalized_probs[0] += self.alpha
            return self.num_agents - 1
        else:
            self.unnormalized_probs[draw] += 1
            self.degrees[draw - 1] += 1
            return draw - 1


def checkunique(data):
    """Quickly checks if a sorted array is all unique elements."""
    for i in range(len(data)-1):
        if data[i]==data[i+1]:
            return False
    return True


def PY_generator(alpha, theta):

    degrees = [0]
    num_draws = 0
    num_agents = 0
    while True:
        draw = get_agent2(degrees, num_draws, alpha, theta, np.random.rand())

        num_draws += 1
        if draw == 0:
            degrees.append(1)
            degrees[0] += 1
            num_agents += 1
            yield num_agents - 1
        else:
            degrees[draw] += 1
            yield draw - 1



class HollywoodBaseNetwork(object):
    def __init__(self):
        pass

    def cdf(self, data, survival=False, **kwargs):
        """
        The cumulative distribution function (CDF) of the data.

        Parameters
        ----------
        data : list or array, optional
        survival : bool, optional
            Whether to calculate a CDF (False) or CCDF (True). False by default.
        xmin : int or float, optional
            The minimum data size to include. Values less than xmin are excluded.
        xmax : int or float, optional
            The maximum data size to include. Values greater than xmin are
            excluded.

        Returns
        -------
        X : array
            The sorted, unique values in the data.
        probabilities : array
            The portion of the data that is less than or equal to X.
        """

        data = np.array(data)
        if not data.any():
            return array([np.nan]), array([np.nan])

        #data = trim_to_range(data, xmin=xmin, xmax=xmax)

        n = float(len(data))

        data = np.sort(data)
        all_unique = not( any( data[:-1]==data[1:] ) )

        if all_unique:
            CDF = np.arange(n)/n
        else:
    #This clever bit is a way of using searchsorted to rapidly calculate the
    #CDF of data with repeated values comes from Adam Ginsburg's plfit code,
    #specifically https://github.com/keflavich/plfit/commit/453edc36e4eb35f35a34b6c792a6d8c7e848d3b5#plfit/plfit.py
            CDF = np.searchsorted(data, data,side='left') / n
            unique_data, unique_indices = np.unique(data, return_index=True)
            data=unique_data
            CDF = CDF[unique_indices]

        if survival:
            CDF = 1-CDF
        return data, CDF


    def pmf(self, data=None, linear_bins=False, xmin=None, xmax=None, **kwargs):
        """
        Returns the probability density function (normalized histogram) of the
        data.

        Parameters
        ----------
        data : list or array
        xmin : float, optional
            Minimum value of the PDF. If None, uses the smallest value in the data.
        xmax : float, optional
            Maximum value of the PDF. If None, uses the largest value in the data.
        linear_bins : float, optional
            Whether to use linearly spaced bins, as opposed to logarithmically
            spaced bins (recommended for log-log plots).

        Returns
        -------
        bin_edges : array
            The edges of the bins of the probability density function.
        probabilities : array
            The portion of the data that is within the bin. Length 1 less than
            bin_edges, as it corresponds to the spaces between them.
        """

        if data is None:
            data = self.data
        data = np.asarray(data)
        if not xmax:
            xmax = np.max(data)
        if not xmin:
            xmin = np.min(data)

        if xmin < 1:  #To compute the pdf also from the data below x=1, the data, xmax and xmin are rescaled dividing them by xmin.
            xmax2 = xmax / xmin
            xmin2 = 1
            raise ValueError('xmin is less than 1.')
        else:
            xmax2 = xmax
            xmin2 = xmin

        if 'bins' in kwargs.keys():
            bins = kwargs.pop('bins')
        elif linear_bins:
            bins = range(int(xmin2), int(xmax2))
        else:
            log_min_size = np.log10(xmin2)
            log_max_size = np.log10(xmax2)
            number_of_bins = np.ceil((log_max_size-log_min_size)*10)
            bins=np.unique(
                    np.floor(
                    np.logspace(
                        log_min_size, log_max_size, num=number_of_bins)))

            hist, edges = np.histogram(data, bins, density=True)

        return edges, hist


    def plot_ccdf(self, data=None, ax=None, survival=True, **kwargs):

        if data is None:
            data = self.data
        return self.plot_cdf(data, ax=ax, survival=True, **kwargs)
        """
        Plots the complementary cumulative distribution function (CDF) of the data
        to a new figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        survival : bool, optional
            Whether to plot a CDF (False) or CCDF (True). True by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """

    def plot_cdf(self, data, ax=None, survival=False, **kwargs):
        """
        Plots the cumulative distribution function (CDF) of the data to a new
        figure or to axis ax if provided.

        Parameters
        ----------
        data : list or array
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        survival : bool, optional
            Whether to plot a CDF (False) or CCDF (True). False by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        bins, CDF = self.cdf(data, survival=survival, **kwargs)
        if not ax:
            fig, ax = plt.subplots()
            ax.loglog(bins, CDF, **kwargs)
        else:
            fig = ax.get_figure()
            ax.plot(bins, CDF, **kwargs)
        return fig, ax

    def plot_pmf(self, data=None, ax=None, linear_bins=False, show_zeros=False, **kwargs):

        """
        Plots the probability density function (PDF) to a new figure or to axis ax
        if provided.

        Parameters
        ----------
        data : list or array
        ax : matplotlib axis, optional
            The axis to which to plot. If None, a new figure is created.
        linear_bins : bool, optional
            Whether to use linearly spaced bins (True) or logarithmically
            spaced bins (False). False by default.

        Returns
        -------
        ax : matplotlib axis
            The axis to which the plot was made.
        """
        if data is None:
            data = self.data

        edges, hist = self.pmf(data, linear_bins=linear_bins, **kwargs)
        bin_centers = (edges[1:]+edges[:-1])/2.0
        #bin_centers = edges[:-1]
        if not show_zeros:
            hist[hist==0] = np.nan

        if not ax:
            fig, ax = plt.subplots()
            ax.loglog(bin_centers, hist, **kwargs)
        else:
            ax.plot(bin_centers, hist, **kwargs)
        return fig, ax
