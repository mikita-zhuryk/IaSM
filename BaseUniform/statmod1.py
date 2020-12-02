#!/usr/bin/env python
# coding: utf-8

# ## Лабораторная работа 1
# # Моделирование базовой случайной величины
# ## Вариант 4
# 
# ### Выполнил:
# Журик Никита Сергеевич,
# 4 курс, 6 группа
# ### Преподаватель:
# Пирштук Иван Казимирович

# In[1]:


import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import sys
sys.path.append('../')

from utils import *


# Реализуем мультипликативный конгруэнтный генератор. Это по сути простейший генератор псевдослучайных чисел, посмотрим, как он покажет себя на практике.

# In[2]:


class UniformDistribution:
    def __init__(self, beta: float = 1203248318,
                 low: float = 0, high: float = 1,
                 seed: int = 1):
        self.a = seed
        self.beta = beta
        self.M = 2 ** 31 - 1
        self.low = low
        self.high = high
        
    def _map_to_range(self, value):
        center = (self.low + self.high) / 2
        diff = (self.high - self.low)
        return diff * (value - 0.5) + center
    
    def mean(self):
        return (self.low + self.high) / 2
        
    def __call__(self, shape: tuple = (1,)):
        n = np.prod(shape)
        random_value = np.empty(n)
        for i in range(n):
            self.a = (self.beta * self.a) % self.M
            random_value[i] = self._map_to_range(self.a / self.M)
        return random_value.reshape(*shape)


# Ниже приведена последовательность действий для анализа характеристик полученного генератора и сравнения с генератором, реализованным в numpy (из интереса, реализовать свою версию того, чем часто пользуешься, всегда интересно).

# In[3]:


def rv_test_pipeline_uniform(dist, np_dist, dist_cdf):
    sampling_sizes = [30, 50, 100, 300, 500, 1000]
    np.random.seed(1203248318)

    seqs = []
    means = []
    stds = []

    np_seqs = []
    np_means = []
    np_stds = []

    for sampling_size in sampling_sizes:
        generated = dist(shape=(sampling_size,))
        np_seqs.append(np_dist(size=sampling_size))
        seqs.append(generated)
        means.append(seqs[-1].mean())
        stds.append(seqs[-1].std())
        np_means.append(np_seqs[-1].mean())
        np_stds.append(np_seqs[-1].std())
        print(f'Mean for {sampling_size} values: {means[-1]}')
        print(f'Std for {sampling_size} values: {stds[-1]}')

    means = np.array(means)
    stds = np.array(means)
    np_means = np.array(np_means)
    np_stds = np.array(np_stds)
    plot_conv_comparison(sampling_sizes, means, np_means, exact_mean=dist.mean())
    plot_autocov_comparison(seqs[-1], np_seqs[-1])
    plot_scatter_comparison(seqs[-1], np_seqs[-1])
    plot_hist_comparison(seqs[-1], np_seqs[-1])
    
    n_bins_chi2 = 5
    EPS = 0.05

    for i in range(len(seqs)):
        seq = seqs[i]
        p_value = find_chi2_p_value(chi2(seq, dist_cdf, k=n_bins_chi2), r=n_bins_chi2-1)
        np_p_value = find_chi2_p_value(chi2(np_seqs[i], dist_cdf, k=n_bins_chi2), r=n_bins_chi2-1)
        print(f'P-value for mcm with {seq.size} elements: {p_value}')
        print(f'P-value for np with {seq.size} elements: {np_p_value}')
        print(f'Null-hypothesis (uniformly distributed) is correct: {p_value > EPS}')
        print()


# In[4]:


if __name__ == "__main__":
    rv_test_pipeline_uniform(UniformDistribution(), np.random.uniform, lambda x: x)


# # Выводы
# 
# 1. Рассмотренный мультипликативный конгруэнтный генератор позволяет моделировать равномерно распределённую случайную величину, во всех случаях сгенерированная случайная величина удовлетворяет критерию согласия Пирсона, поэтому мы не может отвергнуть нулевую гипотезу - то, что СВ распределена равномерно.
# 2. Что касается качества генерации случайной величины, в сравнении с генератором, используемым в np.random, во всех тестах МКМ показывает схожие результаты, отличия не заметны невооружённым глазом. Для более доскональной проверки можно воспользоваться и другими статистическими тестами помимо хи-квадрат.
# 3. Автоковариация также близка к нулю, что значит, что период генератора больше 50.
# 4. В общем данный генератор показал себя неплохо. Одним из его преимуществ является простота и быстродействие, однако он не является криптостойким.
# 
# P.S. Было замечено, что от перезапуска к перезапуску (в зависимости от сида) генераторы иногда производят последовательности с низким p-value (один раз даже было $0.003$).

# In[ ]:




