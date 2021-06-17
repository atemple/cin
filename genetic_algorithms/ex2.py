#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:23:03 2021

@author: atemple
"""

import numpy as np
import matplotlib.pyplot as plt
import ga
import math

# parametros do exercicio
zero    = 0.1  # target que minimiza a funcao 

n_iter  = 180  # total de iteracoes
n_bits  = 10   # tamanho do bitstring
n_pop   = 20   # tamanho da populacao
r_cross = 0.8  # taxa de crossover
r_mut   = 0.01 # taxa de mutacao

def f_aptitude(pop, target):
    pop_size = pop.shape[0]
    # cria o array da função de aptidão
    apt = np.zeros(shape=(pop_size),dtype=float)
    # Converte e calcula a aptidão
    for i in range(pop_size):
        value = ga.bitstring_convert(pop[i, ])
        apt[i] = round((1/(f(target) - f(value) + 1)) * 10 , 2)
    return apt

def f(x):
    return (2 ** (-2 * (x - 0.1 / 0.9) ** 2) * (math.sin(5 * math.pi * x)) ** 6)

def genetic(n_iter, n_pop, n_bits, r_cross, r_mut, target):

    # aptidao media
    avg_apt = []
    # melhor aptidao
    bst_apt = []
    
    t_value = ga.cut(target * 1000, nmin=0,nmax=1000)
    t_value = format(int(t_value), '010b')
    t_value = np.array(list(t_value), dtype=int)
    
    target_n  = ga.bitstring_convert(t_value)
    
    # inicializa a populacao
    init = np.random.randint(2, size=n_pop * n_bits)
    # gera a populacao
    pop  = np.reshape(init, (n_pop, n_bits))
    
    # cria varias geracoes com base na n_iter
    for gen in range(n_iter):
            
        if gen == 0 or gen == 179:
            print(">Geracao %d" % (gen))
            print("Populacao", *pop, sep="\n")
    
        # obtem a aptidao da populacao
        apt = f_aptitude(pop, target_n)
        
        if 10.0 in apt:
            print(">Geracao %d" % (gen))
            print("Populacao", *pop, sep="\n")
            break
        
        # obtem a nova populacao com selecao por roleta
        pop = ga.roulette_selection(pop, apt)
        # pop = ga.tournament_selection(pop, apt, n_bits)
        
        # gera uma nova geracao
        pop = ga.reproduce(pop, n_pop, r_cross, r_mut)
        # pop = ga.reproduce_crossover_only(pop, n_pop, r_cross, r_mut)
        # pop = ga.reproduce_mutation_only(pop, n_pop, r_cross, r_mut)  
    
        avg_apt.append(apt.sum()/len(pop))
        bst_apt.append(np.amax(apt))

                
    return avg_apt, bst_apt

avg_apt, bst_apt = genetic(n_iter, n_pop, n_bits, r_cross, r_mut, zero)


# plt.figure(figsize=(12,8))

plt.plot(bst_apt, 'g', label='Melhor Aptidão da Geração', linewidth=1, linestyle='dashed')
plt.plot(avg_apt, 'r', label='Media das Aptidões da Geração', linewidth=0.5)

plt.title('Aptidão media')
plt.xlabel('Geração')
plt.ylabel('Aptidão')
plt.legend()
plt.show()