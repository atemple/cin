#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:09:04 2021

@author: atemple
"""

import math
import random
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def f(x):
    return (2 ** (-2 * (x - 0.1 / 0.9) ** 2) * (math.sin(5 * math.pi * x)) ** 6)

def get_neighbours(solution, learner):
    neighbours = []
    learner = learner / 10 if learner >= 10 else 1
    const = 0.005 / learner
    neighbours_up = solution + const if solution + const < 1 else solution
    neighbours_down = solution - const if solution - const > 0 else solution
    neighbours.append(neighbours_up)
    neighbours.append(neighbours_down)
    return neighbours

def get_random_value(space, x=0):
    start = random.random()
    value = []
    for i in space:
        diff = i - start
        if diff > 0.05 or diff < -0.05:
            value.append(diff)
    if len(value) == len(space) or x > 300:
        return start
    else:
        return get_random_value(space, x = x + 1)
    
def hill_climbing(f, initial_solution):
    # random.seed(a=0)
    solution = initial_solution
    costs = []
    count = 1
    stop_plato = 0
    while count <= 400:
        neighbours = get_neighbours(solution, count)
        current = f(solution)
        best = current 
        current_solution = solution
        costs.append(current)
        for i in range(len(neighbours)):
            cost = f(neighbours[i])
            if cost >= best:
                stop_plato = stop_plato + 1 if cost == best else 0
                best = cost
                solution = neighbours[i]
        count += 1
        if best == current and current_solution == solution or stop_plato == 20:
            if stop_plato == 20: print('plato')
            break
    return solution, costs

costs = []
solution = []
space_solution = []

for i in range(10):
    space_solution.append(get_random_value(space_solution))
    
    hill_climb_sol = hill_climbing(f, space_solution[len(space_solution) - 1])
    solution.append(hill_climb_sol[0])
    costs.append(hill_climb_sol[1])

    if len(costs) > 1:
        if max(costs[1]) > max(costs[0]):
            costs.pop(0)
        else:
            costs.pop(1)
            
print('Valor X:', hill_climb_sol[0])
print('Custos', hill_climb_sol[1])

t = np.arange(0.0, len(hill_climb_sol[1]), 1)

# plt.figure(figsize=(12,8))
plt.title('Hill climbing')
plt.plot(t, hill_climb_sol[1], linewidth=0.5)
plt.show()