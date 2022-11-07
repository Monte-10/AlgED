import numpy as np
import itertools

from typing import List, Dict, Callable, Iterable

def init_cd(n: int) -> np.ndarray:
    array = [-1] * n
    return array

def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    if p_cd[rep_2] < p_cd[rep_1]:
        p_cd[rep_1] = rep_2
        return rep_2
    
    elif p_cd[rep_2] > p_cd[rep_1]:
        p_cd[rep_2] = rep_1
        return rep_1
    
    else:
        p_cd[rep_2] = rep_1
        p_cd[rep_1] -= 1
        return rep_1

def find(ind: int, p_cd: np.ndarray) -> int:
    aux = ind
    while p_cd[aux] >= 0:
        aux = p_cd[aux]
        
    while p_cd[ind] >= 0:
        aux2 = p_cd[ind]
        p_cd[ind] = aux
        ind = aux2
    
    return aux

def cd_2_dict(p_cd: np.ndarray) -> dict:
    dict = {}
    
    for i in range(len(p_cd)):
        if p_cd[i] < 0:
            dict[i] = [i]

    for i in range(len(p_cd)):
        if p_cd[i] >= 0:
            dict[find(i, p_cd)].append(i)

                
    return dict
    
def ccs(n: int, l: List)-> dict:
    table = init_cd(n)
    for u,v in l:
        rep_u = find(u,table)
        rep_v = find(v,table)

        if rep_u != rep_v:
            union(rep_u, rep_v, table)

    dict = cd_2_dict(table)

    return dict

#######################################################################################
#######################################################################################
#######################################################################################
#--------------------------------------Parte 2----------------------------------------#
#######################################################################################
#######################################################################################
#######################################################################################


'''Escribir una función (06) dist_matrix(n_nodes: int, w_max=10)-> np.ndarray
que genere la matriz de distancias de un grafo con n_nodes nodos, valores enteros con un máximo w_max ; observar
que dicha matriz debe ser simétrica con diagonal 0. Usar para ello funciones de Numpy como np.random.randint o
fill_diagonal .'''
def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    matrix = np.random.randint(0, w_max, size=(n_nodes, n_nodes))
    # simetrica
    matriz_symm = (matrix + matrix.T)//2
    np.fill_diagonal(matriz_symm, 0)
    return matriz_symm

    



'''Escribir una función (07) greedy_tsp(dist_m: np.ndarray, node_ini=0)-> List
que reciba una matriz de distancias y un nodo inicial y devuelva un circuito codicioso como una lista con valores entre 0
y el número de nodos menos 1.
'''
def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    print()
    
    
    
'''Escribir una función (08) len_circuit(circuit: List, dist_m: np.ndarray)-> int
que reciba un circuito y una matriz de distancias y devuelva la longitud de dicho circuito.'''
def len_circuit(circuit: List, dist_m: np.ndarray) -> int:
    print()
    
    
    
'''TSP repetitivo. Una forma sencilla de mejorar nuestro primer algoritmo TSP codicioso es aplicar nuestra función
greedy_tsp a partir de todos los nodos del grafo y devolver el circuito con la menor longitud.
Escribir una función (09) repeated_greedy_tsp(dist_m: np.ndarray)-> List
que implemente esta idea.
'''
def repeated_greedy_tsp(dist_m: np.ndarray) -> list:
    print()
    
    
    
'''TSP exhaustivo. Para grafos pequeños podemos intentar resolver TSP simplemente examinando todos los posibles
circuitos y devolviendo aquel con la distancia más corta.
Escribir una función (10) exhaustive_greedy_tsp(dist_m: np.ndarray)-> List
que implemente esta idea usando la librería itertools . Entre los métodos de iteración implementados en la biblioteca, se
encuentra la función permutations(iterable, r=None) que devuelve un objeto iterable que proporciona sucesivamente
todas las permutaciones de longitud r en orden lexicográfico. Aquí r es por defecto la longitud del iterable pasado como
parámetro, es decir, se generan todas las permutaciones con len(iterable) elementos.'''
def exhaustive_greedy_tsp(dist_m: np.ndarray)-> list:
    print()
    