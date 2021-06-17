#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:13:01 2021

@author: atemple
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def f(x):
    return (2 ** (-2 * (x - 0.1 / 0.9) ** 2) * (math.sin(5 * math.pi * x)) ** 6)

def get_neighbours(solution, learner, c_rate, learn_rate=1):
    neighbours = []
    const = c_rate / learn_rate
    neighbours_up = solution + const if solution + const < 1 else solution
    neighbours_down = solution - const if solution - const > 0 else solution
    neighbours.append(neighbours_up)
    neighbours.append(neighbours_down)
    return neighbours

def simulated_annealing(domain, f, temp, cooling, c_rate=0.005, step = 1):
    # random.seed(a=0)
    solution = random.random()
    costs = []
    count = 1
    stop_plato = 0
    while temp > 0.1:
        neighbour = get_neighbours(solution, count, c_rate)
        current = f(solution)
        best = current 
        solution_current = solution
        costs.append(current)
        for i in range(len(neighbour)):
            if stop_plato == 20:
                break
            cost = f(neighbour[i])
            prob = pow(math.e, (cost - best) / temp)
            if cost >= best or random.random() < prob:
                stop_plato = stop_plato + 1 if cost == best else 0
                best = cost
                solution = neighbour[i]
        temp = temp * cooling
    return solution, costs


t = 10000.0 # temperatura inicial
c = 0.95    # percentual de resfriamento

simulated_annealing_sol = simulated_annealing([0, 1], f, t, c)
simulated_annealing_cost = f(simulated_annealing_sol[0])

print('Menor custo', simulated_annealing_cost)
print('Custos', simulated_annealing_sol[1])

t = np.arange(0.0, len(simulated_annealing_sol[1]), 1)

# plt.figure(figsize=(12,8))
plt.title('Simulated Annealing')
plt.plot(t, simulated_annealing_sol[1], linewidth=0.5)
plt.show()