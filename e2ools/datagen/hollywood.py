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
from ..choice_fns import get_agent, get_agent2
from .utils import HollywoodBaseNetwork

class HollywoodNetwork(HollywoodBaseNetwork):

    def __init__(self, alpha=0.5, theta=100, v=None):

        if v is None:
            self.v = poisson_gen(offset=1)
        else:
            self.v = v
        self.alpha = alpha
        self.theta = theta


    def generate_data(self, num_edges=100):
        """
        Generate an E2 dataset with the above parameters

        A simple generating algorithm for a network with the
        Hollywood model.

        Parameters
        ----------
        num_edges : int, optional (default=100)
            The total number of edges generated.

        v : generator, optional (default=Pois(1) + 1)
            generator for size of interactions

        alpha : float, optional (default=0.5)
            alpha parameter

        theta : float, optional (theta = 100)

        Returns
        -------
        X : list [num_edges]
            The generated edges.
        """

        if (self.alpha > 1) or self.theta <= -self.alpha:
            raise ValueError('Parameters are not in the valid domain.')

        #Initialization
        X = []
        num_agents = 0
        num_draws = 0

        counts = [0]

        for i in range(num_edges):
            if i % 50000 == 0:
                print(i)
            edge = []
            v_draw = next(self.v)
            for n in range(v_draw):
                rnd = np.random.rand()
                agent = get_agent2(counts, num_draws, self.alpha, self.theta, rnd)

                #probs = np.array(unnormalized_probs) / normalizer
                #agent = choice(num_agents + 1, p=probs)


                num_draws += 1

                if agent == -1:
                    pdb.set_trace()
                if agent == 0:
                    agent = num_agents
                    counts[0] += 1
                    counts.append(1)
                    num_agents += 1
                else:
                    counts[agent] += 1
                    agent = agent - 1

                edge.append(agent)

            X.append(edge)
        self.data = counts
        return X, counts 

    def plot_epmf(self, binned=True, data=None, ax=None, linear_bins=False, show_zeros=False, **kwargs):

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

        if binned:
            edges, hist = self.pmf(data, linear_bins=linear_bins, **kwargs)
            bin_centers = 10**((np.log10(edges[1:])+np.log10(edges[:-1]))/2.0)
            if not show_zeros:
                hist[hist==0] = np.nan
        else:
            bin_centers, hist = np.unique(data, return_counts=True)
            hist = hist / sum(hist)

        #bin_centers = (edges[:-1] + edges[1:])/2.0
        if not ax:
            fig, ax = plt.subplots()
            ax.loglog(bin_centers, hist, **kwargs)
            #ax.loglog(bin_centers, theory_pts)
        else:
            fig = ax.get_figure()
            ax.plot(bin_centers, hist, **kwargs)
            #ax.plot(bin_centers, theory_pts)
        return fig, ax


    def plot_pmf(self, xmin=None, xmax=None, ax=None, linear_bins=False, show_zeros=False, **kwargs):

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

        if xmin is None and self.data is not None:
            xmin = np.min(self.data)

        if xmax is None and self.data is not None:
            xmax = np.max(self.data)

        theory_pts = [self.alpha * k**(-(1 + self.alpha)) / gamma(1 - self.alpha) for k in range(xmin, xmax+1)]
        if not ax:
            fig, ax = plt.subplots()
            #ax.loglog(bin_centers, hist, **kwargs)
            ax.loglog(range(xmin, xmax+1), theory_pts, **kwargs)
        else:
            fig = ax.get_figure()
            #ax.plot(bin_centers, hist, **kwargs)
            ax.plot(range(xmin, xmax+1), theory_pts, **kwargs)
        return fig, ax