#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:33:53 2021

@author: atemple
"""

import random
import numpy as np 
import matplotlib.pyplot as plt

class Particle():
    def __init__(self):
        self.position = np.array([random.uniform(-5, 5), random.uniform(-5, 5)])
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.array([random.uniform(0, 3), random.uniform(0, 3)])

    def __str__(self):
        print("I am at ", self.position, " my pbest is ", self.pbest_position)
    
    def move(self):
        self.position = self.position + self.velocity


class PSO():

    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.array([random.random()*50, random.random()*50])
        self.particle_positions = []
        self.particle_fitness = []

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
   
    def fitness(self, x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle.position[0], particle.position[1])
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle.position[0], particle.position[1])
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            global W
            self.particle_fitness.append(particle.pbest_value)
            self.particle_positions.append(particle.pbest_position)
            new_velocity = (W*particle.velocity) + (c1*random.random()) * (particle.pbest_position - particle.position) + \
                            (random.random()*c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()
       
W = 0.5
c1 = 0.9
c2 = 0.8

n_iterations = 500
n_particles = 10
target_error = 1

search_space = PSO(1, target_error, n_particles)
particles_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particles_vector
for particle in search_space.particles:
    print(particle.position)

particle_mean_fitness_per_generation = []
particle_min_fitness_per_generation = []

iteration = 0
while(iteration < n_iterations):
    search_space.set_pbest()   
    search_space.set_gbest()

    if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
        break

    search_space.move_particles()

    particle_mean_fitness_per_generation.append(np.mean(search_space.particle_fitness))
    particle_min_fitness_per_generation.append(np.min(search_space.particle_fitness))

    iteration += 1
    
print("\n")
search_space.print_particles()    
print("The best solution is: ", search_space.gbest_position, "with best value: ", search_space.gbest_value, " in n_iterations: ", iteration)

plt.plot(particle_mean_fitness_per_generation, color='blue')
plt.plot(particle_min_fitness_per_generation, color='red')

plt.legend(['media', 'mínimos'], loc='upper right')
plt.ylabel('Aptidão')
plt.xlabel('Gerações')
plt.grid(True)

plt.show()
