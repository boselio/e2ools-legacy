from sklearn.base import BaseEstimator
from sklearn.utils import check_array
import numpy as np
import pdb
import scipy.optimize as opt
import pandas as pd
from collections import defaultdict
import random
import time
from functools import partial
import scipy
import scipy.sparse as sps
import scipy.stats as stats
import pickle
import os.path
from copy import deepcopy
from ..choice_fns import choice_discrete_unnormalized
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#from .generators import SharedHollywoodNetwork, NewmanClausetPower
from concurrent.futures import ProcessPoolExecutor



class E2Estimator:
    """
    Inference/estimation for the E2 model.

    This code is an implementation of the model in the following paper:

    "Edge exchangeable models for interaction networks", Crane, H. and Dempsey, W.
    Journal of the American Statistical Association (JASA). In print.

    Methods
    ----------
    get_values_from_list(X, flat_list=False)
        Gets the relevant values for inference from a dataset in interaction list
        form or flat list form
    fit(G, list_flag=False, list_flag=False, flat_list=False, degree_flag=False)
        Fits the Hollywood model to G. flat_list and degree_flag determine the 
        form of G. If list_flag = True, and flat_list=False, degree_flag=False, 
        then G should be in the form [[r11, r12, r13], [r21, r22], ...]. This 
        should be the primary use case.
    """

    def __init__(self, max_iter=20, finite_pop=False, alpha_zero=False):
        self.max_iter = max_iter
        self.finite_pop = finite_pop
        self.alpha_zero = alpha_zero

    def get_values_from_list(self, X, flat_list=False):

        if flat_list:
            flat_list = X
            self.m = len(flat_list)
        else:
            flat_list = [i for sublist in X for i in sublist]
            node_sizes = [len(i) for i in X]
            m_ks, m_vals = np.unique(node_sizes, return_counts=True)
            self.m = np.sum(m_ks * m_vals)


        nodes, node_degrees = np.unique(flat_list, return_counts=True)
        self.v_total = len(nodes)

        N_k, N_k_vals = np.unique(node_degrees, return_counts=True)
        self.N_k = pd.Series(dict(zip(N_k, N_k_vals)))
        self.node_degrees_dict = dict(zip(nodes, node_degrees))

    def fit(self, G, list_flag=True, flat_list=False, degree_flag=False):
        #For now, just assume its a pandas df with To and From

        if list_flag:
            self.get_values_from_list(G, flat_list)
        elif degree_flag:
            self.v_total = len(G)
            self.m = sum(G)
            N_k, N_k_vals = np.unique(G, return_counts=True)
            self.N_k = pd.Series(dict(zip(N_k, N_k_vals)))
            #self.node_degrees_dict = dict(zip(nodes, node_degrees))
        else:
            self.interaction_list = G.copy()

            self.interaction_list = self.interaction_list[~pd.isnull(self.interaction_list.To)]
            self.interaction_list.loc[:, 'node_list'] = (self.interaction_list.To.map(list) +
                                                         self.interaction_list.From.map(list))
            self.interaction_list.loc[:, 'edge_size'] = self.interaction_list.node_list.map(len)

            all_nodes = [item for sublist in self.interaction_list.node_list.tolist()
                         for item in sublist]

            self.node_set = set(all_nodes)

            #Get v(Y)
            self.v_total = len(self.node_set)

            #Get M_k
            self.M_k = self.interaction_list.groupby('edge_size').size()

            #Get m(Y)
            self.m = np.sum(np.array(self.M_k) * np.array(self.M_k.index))

            #Get N_k
            self.node_degrees = pd.Series(all_nodes).value_counts()
            self.N_k = pd.Series(self.node_degrees).value_counts()

            self.N_k.sort_index(inplace=True)

        if self.finite_pop:

            self.alphas = [-0.5]
            self.thetas = [self.v_total * 0.5]

            for iterations in range(self.max_iter):
                alpha = opt.fsolve(self._f_alpha_neg, self.alphas[-1]) 
                self.alphas.append(alpha)
                self.thetas.append(-self.v_total * alpha)

        elif self.alpha_zero:

            self.thetas = [10]
            self.alphas = [0]
            alpha = 0
            for iterations in range(self.max_iter):
                #print(iterations)
                theta = opt.fsolve(self._f_theta_dp, self.thetas[-1])
                self.thetas.append(theta)

        else:
            self.thetas = [10]
            self.alphas = [0.5]

            for iterations in range(self.max_iter):
                #print(iterations)
                alpha = opt.fsolve(self._f_alpha, self.alphas[-1], args=(self.thetas[-1]))
                theta = opt.fsolve(self._f_theta, self.thetas[-1], args=(alpha))
                self.alphas.append(alpha)
                self.thetas.append(theta)


        return self


    def _f_alpha(self, alpha_0, theta):
        val = self.v_total / alpha_0
        val += np.sum((-theta / alpha_0**2) / ((theta / alpha_0) + np.array(range(self.v_total))))
        inner_sums = [v * np.sum(1 / (1 - alpha_0 + np.arange(k-1)))
                      for (k, v) in self.N_k.items() if k >= 2]
        val -= np.sum(inner_sums)
        return val


    def _f_theta(self, theta_0, alpha):
        val = np.sum((1/alpha) / ((theta_0 / alpha) + np.arange(self.v_total)))
        val -= np.sum(1 / (theta_0 + np.arange(self.m)))
        return val

    def _f_theta_dp(self, theta_0):
        val = self.v_total / theta_0
        val -= np.sum(1 / (theta_0 + np.arange(self.m)))
        return val

    def _f_alpha_neg(self, alpha_0):
        val = self.v_total / alpha_0
        val += np.sum(self.v_total / (np.arange(self.m) - self.v_total * alpha_0))
        inner_sums = [v * np.sum(1 / (1 - alpha_0 + np.arange(k-1)))
                      for (k, v) in self.N_k.items() if k >= 2]
        val -= np.sum(inner_sums)
        return val

    def _fit_hollywood_model(self, v_total, N_k, m, alpha_init=0.9, theta_init=1, max_iter=20):
        alphas = [alpha_init]
        thetas = [theta_init]
        for iterations in range(max_iter):
            #print(iterations)
            alpha = opt.fsolve(f_alpha, alphas[-1], args=(thetas[-1], v_total, N_k))
            theta = opt.fsolve(f_theta, thetas[-1], args=(alpha, v_total, m))
            alphas.append(alpha)
            thetas.append(theta)

        return self.alphas[-1], self.thetas[-1]





