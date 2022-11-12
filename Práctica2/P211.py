import numpy as np
import itertools
from itertools import permutations

from typing import List, Dict, Callable, Iterable


def init_cd(n: int) -> np.ndarray:
    """
    Función que devuelve un array con los valores -1 en las posiciones {0,1,...,n-1}.

    Args:
        n: número de elementos del array.

    Return:
        np.ndarray: Array con los valores -1 en las posiciones {0,1,...,n-1}.
    """
    array = [-1] * n
    return array


def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    """
    Función que une dos conjuntos disjuntos.
    
    Args:
        rep_1: representante del primer conjunto.
        rep_2: representante del segundo conjunto.
        p_cd: array con los representantes de los conjuntos.
    
    Return:
        int: representante del conjunto unido.
    """
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
    """
    Función que devuelve el representante del conjunto al que pertenece el elemento ind.
    
    Args:
        ind: elemento del conjunto.
        p_cd: array con los representantes de los conjuntos.
    
    Return:
        int: representante del conjunto al que pertenece el elemento ind.
    """
    aux = ind
    while p_cd[aux] >= 0:
        aux = p_cd[aux]
        
    while p_cd[ind] >= 0:
        aux2 = p_cd[ind]
        p_cd[ind] = aux
        ind = aux2
    
    return aux


def cd_2_dict(p_cd: np.ndarray) -> dict:
    """
    Función que devuelve un diccionario con los conjuntos disjuntos.
    
    Args:
        p_cd: array con los representantes de los conjuntos.
    
    Return:
        dict: diccionario con los conjuntos disjuntos.
    """
    dict = {}
    
    for i in range(len(p_cd)):
        if p_cd[i] < 0:
            dict[i] = [i]

    for i in range(len(p_cd)):
        if p_cd[i] >= 0:
            dict[find(i, p_cd)].append(i)

                
    return dict
    
    
def ccs(n: int, l: List)-> dict:
    """
    Función que devuelve las componentes conexas de un grafo.
    
    Args:
        n: número de nodos del grafo.
        l: ramas del grafo.
        
    Return:
        dict: diccionario con las componentes conexas del grafo.
    """
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


def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    """
    Genera la matriz de distancias de un grafo con n_nodes nodos, y simétrica con diagonal 0
    
    Args:
        n_nodes: Número de nodos que tendrá la matriz.
        w_max: Valor máximo que puede tener la matriz en una posición.
    
    Return: 
        matriz_symm: Matriz resultante creada.
    """
    matrix = np.random.randint(1, w_max, size=(n_nodes, n_nodes))
    # simetrica
    matriz_symm = (matrix + matrix.T)//2
    np.fill_diagonal(matriz_symm, 0)
    return matriz_symm

    
def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> List:
    """
    Recibe una matriz de distancias y un nodo inicial y devuelva un circuito codicioso como una lista con valores entre 0
    y el número de nodos menos 1.
    
    Args:
        dist_m: Matriz de distancias.
        node_ini: Nodo inicial desde el que se creará el circuito.
    
    Return: 
        circuito: Lista resultante que corresponde al circuito obtenido.
    """
    # comprobacion de error
    if len(dist_m) <= 0:
       raise Exception("El argumento dist_m esta vacío.")
       
    ciudades = dist_m.shape[0]
    circuito = [node_ini]
    while len(circuito) < ciudades:
        # ciudad actual es la ultima del circuito
        ciudad_act = circuito[-1]

        # se ordenan las distancias para obtener la menor
        distancias = np.argsort(dist_m[ciudad_act])

        # para cada ciudad se comprueba 
        for ciudad in distancias:
            if ciudad not in circuito:
                circuito.append(ciudad)
                break
            
    return list(circuito)
        
    
    
def len_circuit(circuit: List, dist_m: np.ndarray) -> int:
    """
    Recibe un circuito y una matriz de distancias y devuelve la longitud de dicho circuito.
    
    Args:
        circuit: Circuito codicioso.
        dist_m: Matriz de distancias.
    
    Return: 
        longitud: Longitud del circuito.
    """
    # si el circuito esta vacio devuelve error
    if len(circuit) <= 0 :
        raise Exception("El argumento circuit esta vacío")

    # si el circuito esta vacio devuelve error
    if len(dist_m) <= 0 :
        raise Exception("El argumento dist_m esta vacío")
        

    longitud = 0
    # se calcula la longitud sumando las distancias
    for i in range(len(circuit)-1):
        longitud += dist_m[circuit[i]][circuit[i+1]]
        
    return longitud
    
    

def repeated_greedy_tsp(dist_m: np.ndarray) -> list:
    """
    Aplica greedy_tsp con todos los nodos del grafo para devolver el circuito de menor longitud
    
    Args:
        dist_m: Matriz de distancias.
    
    Return: 
        circuito_min: Circuito con la menor longitud de todos.
    """
    circuitos = {} # diccionario auxiliar 
    ciudades = dist_m.shape[0] # numero de ciudades

    # por cada ciudad obtiene el circuito partiendo de ella y la longitud en el diccionario
    for i in range(ciudades):
        # circuito de la ciudad
        circuito = greedy_tsp(dist_m, i)
        # diccionario con el circuito y su longitud
        circuitos[tuple(circuito)] = len_circuit(circuito, dist_m)
        
    # se obtiene el circuito con longitud minima
    circuito_min = min(circuitos, key=circuitos.get)
    
    return list(circuito_min)
    
    
def exhaustive_tsp(dist_m: np.ndarray) -> list:
    """
    Función que recibe una matriz de distancias y examina todas las distancias, luego devuelve el circuito con la longitud mínima
    
    Args:
        dist_m: Matriz de distancias
    
    Return:
        circuito_min: Lista con el circuito con la longitud mínima
    """
    circuitos = {} # diccionario auxiliar 
    ciudades = dist_m.shape[0] # numero de ciudades

    # se prueban todas las combinaciones posibles
    for circuito in itertools.permutations(range(ciudades)):
         # diccionario con el circuito y su longitud
        circuitos[tuple(circuito)] = len_circuit(circuito, dist_m)
        
    # se obtiene el circuito 
    circuito_min = min(circuitos, key=circuitos.get)
    
    return list(circuito_min)