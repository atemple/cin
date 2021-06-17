#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:35:09 2021

@author: atemple
"""

import numpy as np
import matplotlib.pyplot as plt
import ga
from numpy.random import randint

# parametros do exercicio
zero = '111101101111'  
target = np.array(list(zero), dtype=int)

n_iter  = 180  # total de iteracoes
n_bits  = 12   # tamanho do bitstring
n_pop   = 10   # tamanho da populacao
r_cross = 0.8  # taxa de crossover
r_mut   = 0.05 # taxa de mutacao

def f_aptitude(pop, target, n_bits):
    ham = ga.hamming_distance(pop, target)
    apt = ga.hamming_aptitude(ham, n_bits)
    return apt

def genetic(n_iter, n_pop, n_bits, r_cross, r_mut, target):
    
    # inicializa a populacao com um bitstring randomico com o tamanho da populacao n_pop
    pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    
    # aptidao media
    avg_apt = []
    # melhor aptidao
    bst_apt = []
    
    # gera a populacao
    pop = np.reshape(pop, (n_pop, n_bits))
    
                                   
    # cria varias geracoes com base na n_iter
    for gen in range(n_iter):
        
        if gen == 0 or gen == 179:
            print(">Geracao %d" % (gen))
            print("Populacao", *pop, sep="\n")
        
        # obtem a aptidao da populacao
        apt = f_aptitude(pop, target, n_bits)
        
        if 12.0 in apt:
            print(">Geracao %d" % (gen))
            print("Populacao", *pop, sep="\n")
            break
        
        # obtem a nova populacao com selecao por roleta
        pop = ga.roulette_selection(pop, apt)
        # pop = ga.tournament_selection(pop, apt, n_bits)
        
        # gera uma nova geracao
        # pop = ga.reproduce(pop, n_pop, r_cross, r_mut)
        # pop = ga.reproduce_crossover_only(pop, n_pop, r_cross, r_mut)
        pop = ga.reproduce_mutation_only(pop, n_pop, r_cross, r_mut)    
    
        avg_apt.append(apt.sum()/len(pop))
        bst_apt.append(np.amax(apt))
        
        # print("", sep="\n")
    
    return avg_apt, bst_apt
        
avg_apt, bst_apt = genetic(n_iter, n_pop, n_bits, r_cross, r_mut, target)

# print("Aptidao Media", avg_apt, sep="\n")
# print("Distancia Melhor Individuo", bst_apt, sep="\n")

# plt.figure(figsize=(12,8))

plt.plot(bst_apt, 'g', label='Melhor Aptidão da Geração', linewidth=1, linestyle='dashed')
plt.plot(avg_apt, 'r', label='Media das Aptidões da Geração', linewidth=0.5)

plt.title('Aptidão media')
plt.xlabel('Geração')
plt.ylabel('Aptidão')
plt.legend()
plt.show()