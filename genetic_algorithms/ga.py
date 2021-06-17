import random

from numpy.random import randint
from numpy.random import rand
import numpy as np

def hamming_distance(pop, target):
    # calcula a quantidade de elementos diferentes
    ham = np.array([np.count_nonzero(target != pop[i,]) for i in range(len(pop))])
    return ham

def hamming_aptitude(ham, n_bits):
    # calcula a aptidao subtraindo a distancia de hamming 
    apt = np.array([(n_bits - ham[i]) for i in range(ham.size)])
    return apt

def roulette(wh_dgr):
    # sorteia um numero de 0 a 360
    n_sort = random.uniform(0,360)
    k, l = 0, 0
    # avalia toda a roleta
    for j in range(len(wh_dgr)):
        # porcao de avaliacao da roleta
        k += wh_dgr[j]
        if n_sort <= k:
            l = j
            break
    return l
            
def roulette_selection(pop, fit):
    pop_size = pop.shape[0]
    # atribui uma porcao da roleta para o individuo baseado na sua aptidao
    wh_dgr = np.array([(360*fit[i])/(fit.sum()) for i in range(pop_size)])
    # gera uma nova população baseado na roleta
    new_pop = np.array([pop[roulette(wh_dgr)] for i in range(pop_size)])
    return new_pop

def tournament_selection(pop, apt, n_bits, k=3):
    pop_size = pop.shape[0]
    new_pop = np.zeros(shape=(pop_size, n_bits), dtype=int)
    for j in range(pop_size):
        # selecao randomica
        apt_i = randint(pop_size)
        for i in randint(0, pop_size, k-1):
            # verificar se eh melhor
            if apt[i] > apt[apt_i]:
                apt_i = i
        new_pop[j] = pop[apt_i]
    return new_pop

def crossover(p1, p2, r_cross):
    # cria filhos com a copia dos pais
    c1, c2 = p1.copy(), p2.copy()
    # verifica se vai realizar crossover
    if rand() < r_cross:
        # selecao do ponto de crossover
        pt = randint(1, len(p1)-2)
        # realiza o crossover
        c1 = np.concatenate((p1[:pt], p2[pt:])) 
        c2 = np.concatenate((p2[:pt], p1[pt:]))
    return [c1, c2]

def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # verifica se deve realizar mutacao
        if rand() < r_mut:
            # inverte o bit
            bitstring[i] = 1 - bitstring[i]

def reproduce(parents, n_pop, r_cross, r_mut):
    # cria a proxima geracao
    children = []
    for i in range(0, n_pop, 2):
        # obtem os pais
        p1, p2 = parents[i], parents[i+1]
        # realiza o crossover e a mutacao
        for c in crossover(p1, p2, r_cross):
            # reliza a mutacao
            mutation(c, r_mut)
            # salva o filho para a proxima geracao
            children.append(c)
    # substitui a populacao com os filhos
    return np.array(children)

def reproduce_crossover_only(parents, n_pop, r_cross, r_mut):
    # cria a proxima geracao
    children = list()
    for i in range(0, n_pop, 2):
        # obtem os pais
        p1, p2 = parents[i], parents[i+1]
        # realiza o crossover e a mutacao
        for c in crossover(p1, p2, r_cross):
            # salva o filho para a proxima geracao
            children.append(c)
    # substitui a populacao com os filhos
    return np.array(children)

def reproduce_mutation_only(parents, n_pop, r_cross, r_mut):
    # cria a proxima geracao
    children = list()
    for i in range(0, n_pop, 2):
        # obtem os pais
        p1, p2 = parents[i], parents[i+1]
        # realiza o crossover e a mutacao
        for c in (p1, p2):
            # reliza a mutacao
            mutation(c, r_mut)
            # salva o filho para a proxima geracao
            children.append(c)
    # substitui a populacao com os filhos
    return np.array(children)

def bitstring_convert(bit_value):
    value = bit_value
    # transforma array em string
    value_string = np.array2string(value,separator='')
    value_string = value_string.lstrip('[').rstrip(']')
    # converte valores para inteiro
    value_int = int(value_string, 2)
    # executa a estrategia de corte
    value_int = cut(value_int,nmin=0,nmax=1000)
    # gera valor final
    final_value = float(value_int/1000)
    return final_value

def bitstring_convert_signal(bit_value):
    value = bit_value
    # transforma array em string
    value_string = np.array2string(value, separator='')
    value_string = value_string.lstrip('[').rstrip(']')
    # obtem o sinal
    signal = signal_int(value_string[0])
    # converte valores para inteiro
    value_int = value_string[1:]
    value_int = int(value_int,2) * signal
    # executa a estrategia de corte
    value_int = cut(value_int,nmin=-500,nmax=500)
    # gera valor final
    final_value = float(value_int/100)
    return final_value

def cut(value_int,nmin,nmax):
    return min(max(value_int,nmin),nmax)

def signal_int(bit):
    if bit == "0": 
        signal = -1 
    else: 
        signal = 1
    return signal

def signal_str(value):
    if value < 0:
        signal = '0'
    else:
        signal = '1'
    return signal