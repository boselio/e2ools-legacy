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


class SharedHollywoodNetwork(HollywoodBaseNetwork):

    def __init__(self, label_dist=None,
                    v=None, alpha_0=np.random.beta(5,5),
                    theta_0=np.random.gamma(10,10), alphas=None, thetas=None):


        if label_dist is None:
            self.label_dist = PY_generator(alpha=0.5, theta=100)
        else:
            self.label_dist = label_dist

        if v is None:
            self.v = poisson_gen(offset=1)
        else:
            self.v = v

        #def alpha_gen(alpha, phi):
        #    return np.random.beta(phi*alpha, phi * (1 - alpha))
        #def theta_gen(theta, phi):
        #    return np.random.gamma(theta / phi, phi)

        if alphas is None:
            #ag = functools.partial(alpha_gen, alpha=alpha_0, phi=phi_alpha)
            ag = functools.partial(np.random.beta, a=5, b=5)
            alphas = defaultdict(ag)

        if thetas is None:
            #tg = functools.partial(theta_gen, theta=theta_0, phi=phi_theta)
            tg = functools.partial(np.random.gamma, shape=10,scale=10)
            thetas = defaultdict(tg)

        self.alpha_0 = alpha_0
        self.theta_0 = theta_0
        self.alphas = alphas
        self.thetas = thetas

    def generate_data(self, num_edges=100, debug=False):

        X = []

        self.table_counts = defaultdict(lambda: [0])
        self.table_dishes = defaultdict(lambda: [-1])
        self.num_local_draws = defaultdict(int)
        #self.label_counts = []
        self.dish_counts = [0]
        #self.global_num_agents = 0
        self.global_num_draws = 0

        for i in range(num_edges):
            if i % 50000 == 0:
                print(i)
            edge = []
            v_draw = next(self.v)
            sender = next(self.label_dist)
            for n in range(v_draw):
                if debug:
                    import pdb
                    pdb.set_trace()
                table = get_agent2(self.table_counts[sender], self.num_local_draws[sender], self.alphas[sender],
                            self.thetas[sender], np.random.rand())

                if table == 0:
                    self.table_counts[sender][0] += 1
                    self.table_counts[sender].append(1)

                    agent = get_agent2(self.dish_counts, self.global_num_draws, self.alpha_0, self.theta_0, np.random.rand())

                    if agent == 0:
                        agent = self.dish_counts[0]
                        self.dish_counts[0] += 1
                        self.dish_counts.append(1)
                    else:
                        self.dish_counts[agent] += 1
                        agent = agent - 1

                    #self.table_counts[sender][0] += 1
                    #self.table_counts[sender].append(1)
                    self.table_dishes[sender].append(agent)

                    self.global_num_draws += 1

                else:
                    self.table_counts[sender][table] += 1
                    agent = self.table_dishes[sender][table]

                edge.append(agent)
                self.num_local_draws[sender] += 1

            X.append([sender, edge])

        recievers = [r for (s,r) in X]
        _, self.data = np.unique([i for sublist in recievers for i in sublist], return_counts=True)

        return X, self.dish_counts, self.table_counts

    def generate_data_old(self, num_edges=100):

        X = []

        self.table_counts = defaultdict(list)
        self.num_local_draws = defaultdict(int)
        self.num_local_tables = defaultdict(int)

        self.label_counts = []
        self.global_num_agents = 0
        self.global_num_draws = 0

        for i in range(num_edges):
            if i % 50000 == 0:
                print(i)
            edge = []
            v_draw = next(self.v)
            #label = label_dist.partial_fit()
            label = next(self.label_dist)
            for n in range(v_draw):

                denom = self.thetas[label] + self.num_local_draws[label]

                probs = [(table_count - self.alphas[label]) / denom for (_, table_count) in self.table_counts[label]]

                table = choice(self.num_local_tables[label] + 1, p=probs +
                                        [(self.alphas[label] * self.num_local_tables[label] + self.thetas[label]) / denom])


                if table == self.num_local_tables[label]:
                    probs = np.array(self.label_counts + [0])
                    probs = (probs - self.alpha_0) / (self.theta_0 + self.global_num_draws)
                    probs[-1] = (self.theta_0 + self.global_num_agents * self.alpha_0) / (self.theta_0 + self.global_num_draws)

                    #irobs = [(i - self.alpha_0) / (self.theta_0 + self.global_num_draws) for i in self.label_counts]

                    #probs = probs + [(self.theta_0 + self.global_num_agents * self.alpha_0) / (self.theta_0 + self.global_num_draws)]

                    agent = choice(self.global_num_agents + 1, p=probs, replace=False)

                    self.num_local_tables[label] += 1
                    self.table_counts[label].append([agent, 1])

                    if agent == self.global_num_agents:
                        self.label_counts.append(1)
                        self.global_num_agents += 1
                    else:
                        self.label_counts[agent] += 1
                    self.global_num_draws += 1

                else:
                    self.table_counts[label][table][1] += 1
                    agent = self.table_counts[label][table][0]

                edge.append(agent)
                self.num_local_draws[label] += 1

            X.append([label, edge])

        recievers = [r for (s,r) in X]
        _, self.data = np.unique([i for sublist in recievers for i in sublist], return_counts=True)

        return X, self.label_counts, self.table_counts


    def generate_data_fast(self, num_edges=100):

        X = []
        max_label_counts = int(num_edges / 5)
        self.label_counts = np.zeros(max_label_counts)
        self.label_counts[0] = self.theta_0

        self.table_counts = defaultdict(list)
        self.num_local_draws = defaultdict(int)
        self.num_local_tables = defaultdict(int)

        #self.label_counts = []
        self.global_num_agents = 0
        self.global_num_draws = 0

        for i in range(num_edges):
            if i % 50000 == 0:
                print(i)
            edge = []
            v_draw = next(self.v)
            #label = label_dist.partial_fit()
            label = next(self.label_dist)
            for n in range(v_draw):

                denom = self.thetas[label] + self.num_local_draws[label]

                probs = [(table_count - self.alphas[label]) / denom for (_, table_count) in self.table_counts[label]]

                table = choice(self.num_local_tables[label] + 1, p=probs +
                                        [(self.alphas[label] * self.num_local_tables[label] + self.thetas[label]) / denom])


                if table == self.num_local_tables[label]:
                    #probs = np.array(self.label_counts + [0])
                    #probs = (probs - self.alpha_0) / (self.theta_0 + self.global_num_draws)
                    #probs[-1] = (self.theta_0 + self.global_num_agents * self.alpha_0) / (self.theta_0 + self.global_num_draws)

                    #irobs = [(i - self.alpha_0) / (self.theta_0 + self.global_num_draws) for i in self.label_counts]

                    #probs = probs + [(self.theta_0 + self.global_num_agents * self.alpha_0) / (self.theta_0 + self.global_num_draws)]
                    probs = self.label_counts / (self.theta_0 + self.global_num_draws)
                    agent = choice(self.global_num_agents + 1, p=probs[:self.global_num_agents+1], replace=False) - 1

                    self.num_local_tables[label] += 1
                    self.table_counts[label].append([agent, 1])

                    if agent == 0:
                        self.global_num_agents += 1
                        self.label_counts[0] += self.alpha_0
                        self.label_counts[self.global_num_agents] += 1 - self.alpha_0
                    else:
                        self.label_counts[agent + 1] += 1
                    self.global_num_draws += 1

                else:
                    self.table_counts[label][table][1] += 1
                    agent = self.table_counts[label][table][0]

                edge.append(agent)
                self.num_local_draws[label] += 1

            X.append([label, edge])

        recievers = [r for (s,r) in X]
        _, self.data = np.unique([i for sublist in recievers for i in sublist], return_counts=True)

        return X, self.label_counts, self.table_counts

    def plot_epmf(self, binned=True, max_plot=None, data=None, ax=None, linear_bins=False, show_zeros=False, **kwargs):

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
            if max_plot is not None:
                bin_centers = bin_centers[bin_centers <= max_plot]
                hist = hist[bin_centers <= max_plot]
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

        
class ArxivNetwork(HollywoodBaseNetwork):

    def __init__(self, label_dist=None, v_subjects=None,
                 v_authors=None, alpha_0=np.random.beta(5,5),
                 theta_0=np.random.gamma(10,10), alphas=None, phi_alpha=2,
                 phi_theta=10, thetas=None, w=None, num_subjects=20,
                 subjects_alpha=-0.5):


        if label_dist is None:
            self.label_dist = PY_generator(alpha=subjects_alpha, theta=-num_subjects * subjects_alpha)
        else:
            self.label_dist = label_dist

        if v_subjects is None:
            self.v_subjects = poisson_gen(offset=1)
        else:
            self.v_subjects = v_subjects

        if v_authors is None:
            self.v_authors = poisson_gen(offset=1)
        else:
            self.v_authors = v_authors

        def alpha_gen(alpha, phi):
            return np.random.beta(phi*alpha, phi * (1 - alpha))
        def theta_gen(theta, phi):
            return np.random.gamma(theta / phi, phi)

        if alphas is None:
            ag = functools.partial(alpha_gen, alpha=alpha_0, phi=phi_alpha)
            alphas = defaultdict(ag)
            alphas = {i: ag() for i in range(num_subjects)}
        if thetas is None:
            tg = functools.partial(theta_gen, theta=theta_0, phi=phi_theta)
            thetas = defaultdict(tg)
            thetas = {i: tg() for i in range(num_subjects)}

        if w is None:
            w = np.random.dirichlet(np.ones(len(alphas)))
        self.alpha_0 = alpha_0
        self.theta_0 = theta_0
        self.alphas = alphas
        self.thetas = thetas
        self.w = w


    def generate_data(self, num_edges=100):

        X = []
        ps = []
        Zs = []
        #Number of people at each table for each restaurant. 0th is # of tables.
        self.table_counts = defaultdict(lambda: [0])
        #The dish types for each table for each restaurant, 0th doesn't matter.
        self.table_dishes = defaultdict(lambda: [-1])
        #Number of local draws for each restaurant
        self.num_local_draws = defaultdict(int)
        #Global dish counts in the franchise. 0th is number of types of dishes.
        self.dish_counts = [0]
        #self.global_num_agents = 0
        #Number of global dish draws.
        self.global_num_draws = 0

        for i in range(num_edges):
            if i % 50000 == 0:
                print(i)
            subjects = []
            authors = []
            v_subjects_draw = next(self.v_subjects)
            for n in range(v_subjects_draw):
                subjects.append(next(self.label_dist))

            subjects = list(set(subjects))
            v_authors_draw = next(self.v_authors)
            #p = np.random.dirichlet(self.gamma * np.ones(len(subjects)))
            #ps.append(p)

            w_normalized = self.w[subjects] / np.sum(self.w[subjects])
            Z = np.random.choice(subjects, p=w_normalized)
            Zs.append(Z)

            for n in range(v_authors_draw):

                table = get_agent2(self.table_counts[Z], self.num_local_draws[Z],
                                   self.alphas[Z], self.thetas[Z], np.random.rand())

                #New table
                if table == 0:
                    self.table_counts[Z][0] += 1
                    self.table_counts[Z].append(1)

                    agent = get_agent2(self.dish_counts, self.global_num_draws,
                                       self.alpha_0, self.theta_0, np.random.rand())

                    #New agent
                    if agent == 0:
                        agent = self.dish_counts[0]
                        self.dish_counts[0] += 1
                        self.dish_counts.append(1)
                    else:
                        self.dish_counts[agent] += 1
                        agent = agent - 1

                    self.table_dishes[Z].append(agent)

                    self.global_num_draws += 1

                else:
                    self.table_counts[Z][table] += 1
                    agent = self.table_dishes[Z][table]

                authors.append(agent)
                self.num_local_draws[Z] += 1

            X.append([subjects, authors])

        recievers = [r for (s, r) in X]
        _, self.data = np.unique([i for sublist in recievers for i in sublist], return_counts=True)

        return X, Zs



    def generate_data2(self, num_edges=100):

        X = []
        ps = []
        Zs = []
        #Number of people at each table for each restaurant. 
        self.table_counts = defaultdict(list)
        #Number of tables in restaurant
        self.num_tables_in_rest = defaultdict(int)
        #The dish types for each table for each restaurant.
        self.table_dishes = defaultdict(list)
        #Number of local draws for each restaurant
        self.num_local_draws = defaultdict(int)
        #Global dish counts in the franchise.
        self.dish_counts = [0]
        #Number of different types of dishes
        self.num_types_dishes = 0
        #Number of global dish draws.
        self.global_num_draws = 0

        for i in range(num_edges):
            if i % 50000 == 0:
                print(i)
            subjects = []
            authors = []
            v_subjects_draw = next(self.v_subjects)
            for n in range(v_subjects_draw):
                subjects.append(next(self.label_dist))

            subjects = list(set(subjects))
            v_authors_draw = next(self.v_authors)
            p = np.random.dirichlet(self.gamma * np.ones(len(subjects)))
            ps.append(p)

            Z = np.random.choice(subjects, p=p)
            Zs.append(Z)

            for n in range(v_authors_draw):

                table = get_agent2(self.table_counts[Z], self.num_local_draws[Z],
                                   self.alphas[Z], self.thetas[Z], np.random.rand())

                #New table
                if table == 0:
                    self.table_counts[Z][0] += 1
                    self.table_counts[Z].append(1)

                    agent = get_agent2(self.dish_counts, self.global_num_draws,
                                       self.alpha_0, self.theta_0, np.random.rand())

                    #New agent
                    if agent == 0:
                        agent = self.dish_counts[0]
                        self.dish_counts[0] += 1
                        self.dish_counts.append(1)
                    else:
                        self.dish_counts[agent] += 1
                        agent = agent - 1

                    self.table_dishes[Z].append(agent)

                    self.global_num_draws += 1

                else:
                    self.table_counts[Z][table] += 1
                    agent = self.table_dishes[Z][table]

                authors.append(agent)
                self.num_local_draws[Z] += 1

            X.append([subjects, authors])

        recievers = [r for (s, r) in X]
        _, self.data = np.unique([i for sublist in recievers for i in sublist], return_counts=True)

        return X, ps, Zs

