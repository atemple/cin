#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:48:52 2021

@author: atemple
"""

import numpy as np
import matplotlib.pyplot as plt
import ga

# parametros do exercicio
target_x = 1    # valor alvo que minimiza a funcao no valor x
target_y = 1    # valor alvo que minimiza a funcao no valor y

n_iter   = 180  # total de iteracoes
n_bits   = 20   # tamanho do bitstring
n_pop    = 20   # tamanho da populacao
r_cross  = 0.8  # taxa de crossover
r_mut    = 0.02 # taxa de mutacao

def f_aptitude(pop):
    pop_size = pop.shape[0]
    # gera o array de aptidão
    apt = np.zeros(shape=(pop_size), dtype=float)
    # separa as partes x e y do bitstring e testa na função, obtendo a aptidão
    for i in range(pop_size):
        value_x = ga.bitstring_convert_signal(pop[i,:10])
        value_y = ga.bitstring_convert_signal(pop[i,10:])
        apt[i] = round((1/(f(value_x,value_y) + 1)) * 10,4)
    return apt

def f(x,y):
    return ((1 - x) ** 2) + (100 * ((y - x ** 2) ** 2))

def genetic(n_iter, n_pop, n_bits, r_cross, r_mut, target_x, target_y):

    # converte a parte x do valor alvo para bitstring, executando a estrategia de corte
    target_x = ga.cut(target_x * 1, nmin=-5,nmax=5)
    target_x = ga.signal_str(target_x) + format(int(abs(target_x)), '09b')
    target_x = np.array(list(target_x), dtype=int)
    
    # converte a parte y do valor alvo para bitstring, executando a estrategia de corte
    target_y = ga.cut(target_y * 1, nmin=-5, nmax=5)
    target_y = ga.signal_str(target_y) + format(int(abs(target_y)), '09b')
    target_y = np.array(list(target_y), dtype=int)
    
    # Junta os alvos em um bitstring unico [target_x;target_y]
    target = np.concatenate((target_x,target_y), axis=None)
    
    # aptidao media
    avg_apt = []
    # melhor aptidao
    bst_apt = []
    
    # Gera a população inicial
    init = np.random.randint(2, size=n_pop * n_bits)
    pop = np.reshape(init, (n_pop, n_bits))
    
    # cria varias geracoes com base na n_iter
    for gen in range(n_iter):
        
        if gen == 0 or gen == 179:
            print(">Geracao %d" % (gen))
            print("Populacao", *pop, sep="\n")
    
        # obtem a aptidao da populacao
        apt = f_aptitude(pop)
        
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

avg_apt, bst_apt = genetic(n_iter, n_pop, n_bits, r_cross, r_mut, target_x, target_y)
    
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