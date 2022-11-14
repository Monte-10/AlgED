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

#print(split(np.array([8,7,3,6,0,1,2,4,5,9])))
"""Escribir una funcion 'qsel(t: np.ndarray, k: int)-> Union[int, None]' que aplique de manera recursiva el algoritmo 
QuickSelect usando la funcion split anterior y devuelva el valor del elemento que ocupar´ıa el ´ındice k en una ordenacion de ´ 
t si ese elemento existe y None si no."""
# 12345678
# 4563 7 
def qsel(t: np.ndarray, k: int)-> Union[int, None]:
    print("ENTRA EN QSEL:", t, k)
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
    while len(t) > 1:
        t1, p, mayores = split(t)
        if k < len(t1):
            t = t1
        elif k == len(t1):
            return p
        else:
            t = mayores
            k = k-len(t1)-1
    return t[0]

"""Escribir una función 'split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]' que modifique la funcion split anterior de manera que use el valor mid para dividir t ."""
def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    # se obtienen los elementos mayores a t[0]
    mayores = t[t>t[mid]]

    # se obtienen los elementos menores a t[0]
    menores = t[t<t[mid]]

    return menores, t[mid], mayores

"""Escribir una función "pivot5(t: np.ndarray)-> int" que devuelve el “pivote 5” del array t de acuerdo al procedimiento “mediana de medianas de 5 elementos” y llamando a la función qsel5_nr que se define a continuacion. """
def pivot5(t: np.ndarray)-> int:
    if len(t) < 5:
        return qsel5_nr(t, len(t)//2)
    else:
        # se obtienen los elementos mayores a t[0]
        mayores = t[t>t[0]]

        # se obtienen los elementos menores a t[0]
        menores = t[t<t[0]]

"""Escribir una función no recursiva 'qsel5_nr(t: np.ndarray, k: int)-> Union[int, None]' que devuelve el elemento en el índice k de una ordenacion de t utilizando la funciones pivot5, split_pivot anteriores"""
def qsel5_nr(t: np.ndarray, k: int)-> Union[int, None]:
    while len(t) > 1:
        mid = pivot5(t)
        menores, p, mayores = split_pivot(t, mid)
        if k < len(menores):
            t = menores
        elif k == len(menores):
            return p
        else:
            t = mayores
            k = k-len(menores)-1
    return t[0]

"""Escribir una función qsort_5(t: np.ndarray)-> np.ndarray que utilice las funciones anteriores split_pivot, pivot_5 para devolver una ordenacion de la tabla t."""
def qsort_5(t: np.ndarray)-> np.ndarray:
    if len(t) <= 1:
        return t
    else:
        mid = pivot5(t)
        t1, p, t2 = split_pivot(t, mid)
        return np.concatenate((qsort_5(t1), np.array([p]), qsort_5(t2)))

########################################################################################
####################### PROGRAMACION DINAMICA ##########################################
########################################################################################

"""Escribir una función "edit_distance(str_1: str, str_2: str)-> int" que devuelva la distancia de edicion entre las cadenas str_1, str_2 utilizando la menor cantidad de memoria posible."""
def edit_distance(str_1: str, str_2: str)-> int:
    # se obtiene la longitud de las cadenas
    m = len(str_1)
    n = len(str_2)

    # se crea una matriz de m+1 x n+1
    d = np.zeros((m+1, n+1), dtype=int)

    # se rellena la primera fila y columna con los valores de 0 a m y 0 a n
    for i in range(m+1):
        d[i, 0] = i
    for j in range(n+1):
        d[0, j] = j

    # se rellena la matriz
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str_1[i-1] == str_2[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j-1], d[i-1, j], d[i, j-1]) + 1

    return d[m, n]

""" Escribir una función max_subsequence_length(str_1: str, str_2: str)-> int que devuelva la longitud de una subsecuencia comun a las cadenas str_1, str_2 aunque no necesariamente consecutiva. Dicha funcion deber a usar la menor cantidad de memoria posible. """
def max_subsequence_length(str_1: str, str_2: str)-> int:
    # se obtiene la longitud de las cadenas
    m = len(str_1)
    n = len(str_2)

    # se crea una matriz de m+1 x n+1
    d = np.zeros((m+1, n+1), dtype=int)

    # se rellena la matriz
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str_1[i-1] == str_2[j-1]:
                d[i, j] = d[i-1, j-1] + 1
            else:
                d[i, j] = max(d[i-1, j], d[i, j-1])

    return d[m, n]

"""Escribir una función max_common_subsequence(str_1: str, str_2: str)-> str que devuelva una subcadena comun a las cadenas str_1, str_2 aunque no necesariamente consecutiva."""
def max_common_subsequence(str_1: str, str_2: str)-> str:
    # se obtiene la longitud de las cadenas
    m = len(str_1)
    n = len(str_2)

    # se crea una matriz de m+1 x n+1
    d = np.zeros((m+1, n+1), dtype=int)

    # se rellena la matriz
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str_1[i-1] == str_2[j-1]:
                d[i, j] = d[i-1, j-1] + 1
            else:
                d[i, j] = max(d[i-1, j], d[i, j-1])

    # se obtiene la subcadena
    subcadena = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if str_1[i-1] == str_2[j-1]:
            subcadena = str_1[i-1] + subcadena
            i -= 1
            j -= 1
        elif d[i-1, j] > d[i, j-1]:
            i -= 1
        else:
            j -= 1

    return subcadena
    
"""Escribir una función min_mult_matrix(l_dims: List[int])-> int que devuelva el mínimo numero de productos para multiplicar n matrices cuyas dimensiones estan contenidas en la lista l_dims con n+1 ints, el primero de los cuales nos da las filas de la primera matriz y el resto las columnas de todas ellas."""
def min_mult_matrix(l_dims: List[int])-> int:
    n = len(l_dims) - 1

    # se crea una matriz de n x n
    d = np.zeros((n, n), dtype=int)

    # se rellena la matriz
    for i in range(1, n):
        for j in range(n-i):
            d[j, j+i] = min([d[j, k] + d[k+1, j+i] + l_dims[j]*l_dims[k+1]*l_dims[j+i+1] for k in range(j, j+i)])

    return d[0, n-1]