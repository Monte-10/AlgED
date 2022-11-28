import numpy as np
import itertools
from typing import List, Tuple, Dict, Callable, Iterable, Union

# 1-A. QuickSelect b ́asico
def split(t: np.ndarray)-> Tuple[np.ndarray, int, np.ndarray]:
    # se obtienen los elementos mayores a t[0]
    mayores = t[t>t[0]]

    # se obtienen los elementos menores a t[0]
    menores = t[t<t[0]]

    return menores, t[0], mayores

"""Escribir una funcion 'qsel(t: np.ndarray, k: int)-> Union[int, None]' que aplique de manera recursiva el algoritmo 
QuickSelect usando la funcion split anterior y devuelva el valor del elemento que ocupar´ıa el ´ındice k en una ordenacion de ´ 
t si ese elemento existe y None si no."""
def qsel(t: np.ndarray, k: int)-> Union[int, None]:
    # si la longitud de t es 1 devuelve el valor que queda
    if len(t) == 1:
        return t[0]
        
    else:
        # si no, realiza un split para dividir el array en dos
        menores, p, mayores = split(t)

        # si k es menor que la longitud se utilizan los menores
        if k < len(menores):
            return qsel(menores, k)

        # si k es igual a la longitud significa que K se encuentra en el medio,
        # ej 4563 7 89 con k = 4
        elif k == len(menores):
            return p

        # si k es mayor que la longitud se usa mayores y se recalcula el valor de k
        else:
            return qsel(mayores, k-len(menores)-1)

print(qsel(np.array([8,10,12,2,4,6,7,0,3]),1))
"""Escribir una funcion no recursiva 'qsel_nr(t: np.ndarray, k: int)-> 
Union[int, None]' que elimine la recursion de cola de la función anterior."""
def qsel_nr(t: np.ndarray, k: int)-> Union[int, None]:
    # podria evitarse esto y trabajar directamente sobre t
    aux_t = t

    # mientras la longitud sea mayor a 1 realiza split y comprobaciones iguales a qsel
    while(len(aux_t) > 1):
        # realiza el split
        menores, p, mayores = split(aux_t)

        if k < len(menores):
            aux_t = menores

        elif k == len(menores):
            return p

        else:
            k = k-len(menores)
            aux_t = mayores

    return aux_t[0]

"""Escribir una función 'split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]' que modifique la funcion split 
anterior de manera que use el valor mid para dividir t ."""
def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    # se obtienen los elementos mayores a t[0]
    mayores = t[t>t[mid]]

    # se obtienen los elementos menores a t[0]
    menores = t[t<t[mid]]

    return menores, t[mid], mayores

"""Escribir una función "pivot5(t: np.ndarray)-> int" que devuelve el “pivote 5” del array t de acuerdo al 
procedimiento “mediana de medianas de 5 elementos” y llamando a la función qsel5_nr que se define a continuacion. """
def pivot5(t: np.ndarray)-> int:
    if len < 5:
        return qsel5_nr(t,0)

    # se obtienen las medianas
    t_aux = t[:5*len(t//5)]
    medianas = np.median(t_aux.reshape(-1,5), axis = 1)

    # se comprueba la longitud
    if len(medianas) <= 5:
        if len(medianas)%2 == 0 : # tamaño par
            pivot = sorted(medianas)[len(medianas)//2-1]
        else: # tamaño impar
            pivot = sorted(medianas)[len(medianas)//2]
    else: # si la longitud es menor se llama a quickselect con k actualizado
        pivot =  qsel5_nr(medianas, len(medianas)//2)
    return pivot

print(pivot5(np.array([1,6,2,4,3,0,9,5,7,8])))

    
def qsel5_nr(t: np.ndarray, k: int)-> Union[int, None]:
    # Caso base (con cutoff)
    if len(t) <= 5:
        return sorted(t)[k-1]
    
    # Obtiene el pivote (mediana de las medianas)
    pivote_5 = pivot5(t)
    
    # Obtiene las particiones utilizando el pivote 
    menores, pivote, mayores = split_pivot(t, pivote_5)

    if k == len(menores):
        return pivote
    elif k < len(menores) :
        return qsel5_nr(menores, k)
    else:
        return qsel5_nr(mayores, k-len(menores))

print(qsel5_nr([8,7,3,6,0,1,2,4,5,9], 4))
