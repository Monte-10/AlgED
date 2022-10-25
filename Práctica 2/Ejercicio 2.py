import numpy as np
import itertools

from typing import List, Dict, Callable, Iterable

'''Escribir una función (06) dist_matrix(n_nodes: int, w_max=10)-> np.ndarray
que genere la matriz de distancias de un grafo con n_nodes nodos, valores enteros con un máximo w_max ; observar
que dicha matriz debe ser simétrica con diagonal 0. Usar para ello funciones de Numpy como np.random.randint o
fill_diagonal .'''
def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    



'''Escribir una función (07) greedy_tsp(dist_m: np.ndarray, node_ini=0)-> List
que reciba una matriz de distancias y un nodo inicial y devuelva un circuito codicioso como una lista con valores entre 0
y el número de nodos menos 1.
'''
def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    
    
    
    
'''Escribir una función (08) len_circuit(circuit: List, dist_m: np.ndarray)-> int
que reciba un circuito y una matriz de distancias y devuelva la longitud de dicho circuito.'''
def len_circuit(circuit: List, dist_m: np.ndarray) -> int:
    
    
    
    
'''TSP repetitivo. Una forma sencilla de mejorar nuestro primer algoritmo TSP codicioso es aplicar nuestra función
greedy_tsp a partir de todos los nodos del grafo y devolver el circuito con la menor longitud.
Escribir una función (09) repeated_greedy_tsp(dist_m: np.ndarray)-> List
que implemente esta idea.
'''
def repeated_greedy_tsp(dist_m: np.ndarray) -> list:
    
    
    
    
'''TSP exhaustivo. Para grafos pequeños podemos intentar resolver TSP simplemente examinando todos los posibles
circuitos y devolviendo aquel con la distancia más corta.
Escribir una función (10) exhaustive_greedy_tsp(dist_m: np.ndarray)-> List
que implemente esta idea usando la librería itertools . Entre los métodos de iteración implementados en la biblioteca, se
encuentra la función permutations(iterable, r=None) que devuelve un objeto iterable que proporciona sucesivamente
todas las permutaciones de longitud r en orden lexicográfico. Aquí r es por defecto la longitud del iterable pasado como
parámetro, es decir, se generan todas las permutaciones con len(iterable) elementos.'''
def exhaustive_greedy_tsp(dist_m: np.ndarray)-> list: