#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 21:31:57 2021

@author: atemple
"""

from aco_lib import *
import numpy as np
import matplotlib.pyplot as plt

def aco(space, iterations = 80, colony = 50, alpha = 1.0, beta = 1.0, tau = 1.0, rho = 0.5):
    inv_distances = inverseDistances(space)
    inv_distances = inv_distances ** beta
    pheromones = np.zeros((space.shape[0], space.shape[0]))
    min_distance = None
    min_path = None

    for i in range(iterations):
        positions = initializeAnts(space, colony)
        paths = moveAnts(space, positions, inv_distances, pheromones, alpha, beta, tau)
        pheromones *= (1 - rho)

        for path in paths:
            distance = 0
            for node in range(1, path.shape[0]):
                distance += np.sqrt(((space[int(path[node])] - space[int(path[node - 1])]) ** 2).sum())

            if not min_distance or distance < min_distance:
                min_distance = distance
                min_path = path
            
        min_path = np.append(min_path, min_path[0])

        return (min_path, min_distance)
    
def inverseDistances(space):
    distances = np.zeros((space.shape[0], space.shape[0]))

    for index, point in enumerate(space):
        distances[index] = np.sqrt(((space - point) ** 2).sum(axis = 1))

    with np.errstate(all = 'ignore'):
        inv_distances = 1 / distances

    inv_distances[inv_distances == np.inf] = 0

    return inv_distances


def initializeAnts(space, colony):
    return np.random.randint(space.shape[0], size = colony)

def moveAnts(space, positions, inv_distances, pheromones, alpha, beta, tau):
    paths = np.zeros((space.shape[0], positions.shape[0]), dtype = int) - 1
    paths[0] = positions

    for node in range(1, space.shape[0]):
        for ant in range(positions.shape[0]):
            next_location_probability = (inv_distances[positions[ant]] ** alpha + pheromones[positions[ant]] ** beta /
                                            inv_distances[positions[ant]].sum() ** alpha + pheromones[positions[ant]].sum() ** beta)
            next_position = np.argwhere(next_location_probability == np.amax(next_location_probability))[0][0]

            while next_position in paths[:, ant]:
                next_location_probability[next_position] = 0.0
                next_position = np.argwhere(next_location_probability == np.amax(next_location_probability))[0][0]

            paths[node, ant] = next_position
            pheromones[node, next_position] = pheromones[node, next_position] + tau

    return np.swapaxes(paths, 0, 1)

# Get TSP data from file
TSP = getTspData('berlin52.tsp')

# Display TSP file headers
displayTspHeaders(TSP)

# Get Space
space = np.array(TSP['node_coord_section'])

# parameters
n = 100
average = 0
iterations = 100
colony = 10
alpha = 1
beta = 10
rho = 0.5
tau = 10

min_distances = []
min_paths = []

for i in range(n):
    min_path, min_distance = aco(space, iterations, colony, alpha, beta, tau, rho)
    min_distances.append(min_distance)
    min_paths.append(min_path)    
    print('Result iter #{} - min distance {}'.format(i + 1, min_distance))

best_result_idx = np.argmin(min_distances) + 1
best_distance = np.min(min_distances)

# Plot nodes
plt.scatter(space[:, 0], space[:, 1], s = 15)

plt.xlabel('Latitude')
plt.ylabel('Longitude')

plt.scatter(space[:, 0], space[:, 1], marker='o', s=15)
plt.plot(space[min_path, 0], space[min_path, 1], c='g', linewidth=0.8, linestyle="--")

# Plot city distances
plt.title('Best result #{} of {} for a minimum distance of {}'.format(best_result_idx, n, best_distance), fontsize = 10)
plt.suptitle('Mininum Path for {}'.format(TSP['name']))
plt.xlabel('Latitude')
plt.ylabel('Longitude')
    
plt.show()
plt.close()

# Plot mininum distances
plt.title('Minimum Distances')
plt.xlabel('interactions')
plt.ylabel('distances')

plt.plot(np.arange(1, n+1), min_distances)
plt.show()

print('Min Distance for the last {} results is {}'.format(best_result_idx, best_distance))

