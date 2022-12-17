import numpy as np
import itertools
from typing import List, Tuple, Dict, Callable, Iterable, Union

def split(t: np.ndarray)-> Tuple[np.ndarray, int, np.ndarray]:
    # se obtienen los elementos mayores a t[0]
    mayores = t[t>t[0]]

    # se obtienen los elementos menores a t[0]
    menores = t[t<t[0]]

    return menores, t[0], mayores

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

# print(qsel(np.array([8,10,12,2,4,6,7,0,3]),1))
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

def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    # se obtienen los elementos mayores a mid
    mayores = t[t>mid]

    # se obtienen los elementos menores a mid
    menores = t[t<mid]

    return menores, mid, mayores

def pivot5(t: np.ndarray)-> int:
    if len(t) < 5:
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

# print(pivot5(np.array([1,6,2,4,3,0,9,5,7,8])))

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

# print(qsel5_nr(np.array([8,7,3,6,0,1,2,4,5,9]), 4))

def edit_distance(str_1: str, str_2: str)-> int:
    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)

    # se crea la matriz de distancias
    matriz = np.zeros((len_1+1, len_2+1), dtype = int)

    # se rellena la matriz en primera fila y columna del 1 al len(str)
    matriz[0,:] = np.arange(len_2+1)
    matriz[:,0] = np.arange(len_1+1)
    
    # se recorre la matriz de distancias
    for i in range(1, len_1+1):
        for j in range(1, len_2+1):

            # si los caracteres son iguales se copia el valor de la diagonal
            if str_1[i-1] == str_2[j-1]:
                # copia
                matriz[i,j] = matriz[i-1,j-1]
            # caracteres distintos
            else:
                # se obtiene el minimo de las anteriores posiciones
                matriz[i,j] = min(matriz[i-1,j], matriz[i,j-1]) +1

    # valor de la ultima posicion
    return matriz[-1,-1]

def max_subsequence_length(str_1: str, str_2: str)-> int:
    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)

    # se crea una matriz de len_1+1 x len_2+1
    matriz = np.zeros((len_1+1, len_2+1), dtype = int)

    # se recorre la matriz
    for i in range(1, len_1+1):
        for j in range(1, len_2+1):
            # si los caracteres son iguales se copia el valor de la diagonal +1
            if str_1[i-1] == str_2[j-1]:
                matriz[i,j] = matriz[i-1,j-1] +1
            # si no, se coge el maximo de la fila y columna anterior
            else:
                matriz[i,j] = max(matriz[i-1,j], matriz[i,j-1])
                
    # se devuelve el valor de la ultima posicion
    return matriz[-1,-1]

print(max_subsequence_length("cataratas", "tarta"))

# Escribir una función max_common_subsequence(str_1: str, str_2: str)-> str que devuelva una subcadena comun a las cadenas str_1, str_2 aunque no necesariamente consecutiva.
def max_common_subsequence(str_1: str, str_2: str)-> str:
    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)

    # se crea una matriz de len_1+1 x len_2+1
    matriz = np.zeros((len_1+1, len_2+1), dtype = int)

    # se recorre la matriz
    for i in range(1, len_1+1):
        for j in range(1, len_2+1):
            # si los caracteres son iguales se copia el valor de la diagonal +1
            if str_1[i-1] == str_2[j-1]:
                matriz[i,j] = matriz[i-1,j-1] +1
            # si no, se coge el maximo de la fila y columna anterior
            else:
                matriz[i,j] = max(matriz[i-1,j], matriz[i,j-1])
                
    # se obtiene la subcadena
    subcadena = ""
    i = len_1
    j = len_2
    while i > 0 and j > 0:
        if str_1[i-1] == str_2[j-1]:
            subcadena = str_1[i-1] + subcadena
            i -= 1
            j -= 1
        elif matriz[i-1,j] > matriz[i,j-1]:
            i -= 1
        else:
            j -= 1
    return subcadena

print(max_common_subsequence("cataratas", "tarta"))

# Escribir una función min_mult_matrix(l_dims: List[int])-> int que devuelva el m´ınimo numero de productos para multiplicar ´ n matrices cuyas dimensiones estan contenidas en la lista l_dims con n+1 ints, el primero de los cuales nos da las filas de la primera matriz y el resto las columnas de todas ellas.
def min_mult_matrix(l_dims: List[int])-> int:
    # se obtiene el numero de matrices
    n = len(l_dims) -1

    # se crea la matriz de costes
    matriz = np.zeros((n,n), dtype = int)

    # se recorre la matriz
    for i in range(1, n):
        for j in range(n-i):
            # se obtiene el minimo de la multiplicacion de las matrices
            matriz[j,j+i] = min(matriz[j,k] + matriz[k+1,j+i] + l_dims[j]*l_dims[k+1]*l_dims[j+i+1] for k in range(j,j+i))
    return matriz[0,-1]

print(min_mult_matrix([10,20,50,1,100]))