#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt


# In[ ]:


def autocov(seq: np.ndarray, offset: int):
    if offset == 0:
        return np.mean(seq ** 2)
    return np.mean(seq[:-offset] * seq[offset:])


# In[ ]:


def plot_conv_comparison(x, means, np_means, exact_mean: float = 0.5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(x, np.abs(means - exact_mean))
    ax2.plot(x, np.abs(np_means - exact_mean))
    ax1.set_xlabel('# samples')
    ax1.set_ylabel('Divergence from mean')
    ax1.set_title('Mean convergence plot')
    ax2.set_xlabel('# samples')
    ax2.set_ylabel('Divergence from mean')
    ax2.set_title('Mean convergence plot for np')
    plt.show()


# In[ ]:


def plot_autocov_comparison(seq, np_seq):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot([autocov(seq, k) for k in range(51)])
    ax2.plot([autocov(np_seq, k) for k in range(51)])
    ax1.set_xlabel('Offset')
    ax1.set_ylabel('Autocovariance')
    ax1.set_title('Autocovariance function')
    ax2.set_xlabel('Offset')
    ax2.set_ylabel('Autocovariance')
    ax2.set_title('Autocovariance function for np')
    plt.show()


# In[ ]:


def plot_scatter_comparison(seq, np_seq):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.scatter(seq[::2], seq[1::2])
    ax2.scatter(np_seq[::2], np_seq[1::2])
    ax1.set_ylabel('Elements with even indices')
    ax1.set_xlabel('Elements with odd indices')
    ax1.set_title('Sequence scatter plot')
    ax2.set_ylabel('Elements with even indices')
    ax2.set_xlabel('Elements with odd indices')
    ax2.set_title('Sequence scatter plot for np')
    plt.show()


# In[ ]:


def plot_hist_comparison(seq, np_seq, bins: int = 16):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.hist(seq, bins=bins)
    ax2.hist(np_seq, bins=bins)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Occurences')
    ax1.set_title('Distribution histogram')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Occurences')
    ax2.set_title('Distribution histogram for np')
    plt.show()


# In[ ]:


from scipy import stats

def discrete_chi2(seq, possible_values, gt_proba):
    unique_values, unique_counts = np.unique(seq, return_counts=True)
    seq_dist = np.zeros_like(possible_values)
    if unique_values.size < possible_values.size:
        i = 0
        for (value, count) in zip(unique_values, unique_counts):
            while possible_values[i] != value:
                i += 1
            seq_dist[i] = count
    else:
        seq_dist = unique_counts
    gt_dist = gt_proba * seq.size
    return np.sum((seq_dist - gt_dist) ** 2 / gt_dist)
    

def chi2(seq, dist_func, low: float = 0, high: float = 1, k: int = 5):
        bins = np.linspace(low, high, k + 1)
        dist = np.array([dist_func(value) for value in bins])
        dist = dist[1:] - dist[:-1]
        hist = plt.hist(seq, bins=bins)[0]
        return seq.size * np.sum((hist / seq.size - dist) ** 2 / dist)

def find_chi2_p_value(chi2: float, r: int):
    return 1 - stats.chi2.cdf(chi2, r)

