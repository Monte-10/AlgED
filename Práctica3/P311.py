import numpy as np
import itertools
from typing import List, Tuple, Dict, Callable, Iterable, Union

def split(t: np.ndarray)-> Tuple[np.ndarray, int, np.ndarray]:
    """
    Función que divide un array en dos partes, una con los elementos menores que el pivote y otra con los mayores.

    Args:
        t: array a dividir.

    Return:
        Tuple[np.ndarray, int, np.ndarray]: tupla con los elementos menores, el pivote y los mayores.
    """
    # comprueba errores
    if len(t) <= 0 or t is None:
        return None
    # se obtienen los elementos mayores a t[0]
    mayores = t[t>t[0]]

    # se obtienen los elementos menores a t[0]
    menores = t[t<t[0]]

    return menores, t[0], mayores

def qsel(t: np.ndarray, k: int)-> Union[int, None]:
    """
    Función que devuelve el elemento k de un array ordenado de manera recursiva.

    Args:
        t: array ordenado.
        k: elemento a devolver.

    Return:
        Union[int, None]: elemento k del array ordenado.
    """
    # comprueba que se pase un valor aceptado de k y otros errores
    if k < 0 or k >= len(t) or len(t) <= 0 or t is None:
        return None
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
        elif k == len(menores):
            return p

        # si k es mayor que la longitud se usa mayores y se recalcula el valor de k
        else:
            return qsel(mayores, k - (len(menores) +1))

def qsel_nr(t: np.ndarray, k: int)-> Union[int, None]:
    """
    Lo mismo que qsel pero de manera no recursiva.

    Args:
        t: array ordenado.
        k: elemento a devolver.

    Return:
        Union[int, None]: elemento k del array ordenado.
    """
    # comprueba que se pase un valor aceptado de k y otros errores
    if k < 0 or k >= len(t) or len(t) <= 0 or t is None:
        return None
    
    else:
        # mientras la longitud sea mayor a 1 realiza split y comprobaciones iguales a qsel
        while (len(t) >= 1):
            # realiza el split
            menores, p, mayores = split(t)

            if k < len(menores):
                t = menores

            elif k == len(menores):
                return p

            else:
                k = k - (len(menores) +1)
                t = mayores

        return

def split_pivot(t: np.ndarray, mid: int)-> Tuple[np.ndarray, int, np.ndarray]:
    """
    Función que divide un array en dos partes, una con los elementos menores que el pivote y otra con los mayores.

    Args:
        t: array a dividir.
        mid: pivote.

    Return:
        Tuple[np.ndarray, int, np.ndarray]: tupla con los elementos menores, el pivote y los mayores.
    """
    # comprobacion de error
    if t is None or len(t) < 1:
        return None

    # empleamos un metodo diferente al usado anteriormente pues daba problemas de ejecucion
    mayores = [y for y in t[0 : len(t)] if y > mid]
    menores = [x for x in t[0 : len(t)] if x < mid]
    
    return (menores, mid, mayores)


def pivot5(t: np.ndarray)-> int:
    """
    Función que devuelve el pivote de un array de 5 elementos.

    Args:
        t: array de 5 elementos.

    Return:
        int: pivote.
    """
    # comprueba errores
    if len(t) <= 0 or t is None:
        return None

    # si la longitud es menor a 5 no se calculan las medianas
    if len(t) < 5:
        return qsel5_nr(t,0)

    # se obtienen las medianas
    num_group = len(t)//5
    # indice donde se encuenta la mediana en una lista de 5 elementos ordenada
    indices = 5//2

    # se descartan grupos de menor tamaño a 5
    sublistas = [t[i:i+ 5] for i in range(0, len(t), 5)][:num_group]

    # se obtienen las medianas
    medianas = [sorted(sub)[indices] for sub in sublistas]
    
    # si el numero de medianas es menor a 5 se ordena
    if len(medianas) <= 5:
        if len(medianas)%2 == 0 : # tamaño par
            pivot = sorted(medianas)[len(medianas)//2-1]
        else: # tamaño impar
            pivot = sorted(medianas)[len(medianas)//2]
    else: # si la longitud es menor se llama a quickselect con k actualizado
        pivot =  qsel5_nr(medianas, len(medianas)//2)
    return pivot



def qsel5_nr(t: np.ndarray, k: int)-> Union[int, None]:
    """
    Función que devuelve el elemento k de un array de 5 elementos de manera no recursiva.

    Args:
        t: array de 5 elementos.
        k: elemento a devolver.

    Return:
        Union[int, None]: elemento k del array ordenado.
    """
    # comprueba que se pase un valor aceptado de k y otros errores
    if k < 0 or k >= len(t) or len(t) <= 0 or t is None:
        return None

    # Caso base (con cutoff 5)
    while len(t) > 5:
        # Obtiene el pivote (mediana de las medianas)
        pivote_5 = pivot5(t)
        
        # Obtiene las particiones utilizando el pivote 
        menores, pivote, mayores = split_pivot(t, pivote_5)

        if k == len(menores):
            return pivote
        elif k < len(menores) :
            t = menores
        else:
            t = mayores
            k = k - (len(menores) + 1)

    return sorted(t)[k]


def qsort_5(t: np.ndarray)-> np.ndarray:
    """
    Función que ordena un array de 5 elementos.

    Args:
        t: array de 5 elementos.
        
    Return:
        np.ndarray: array ordenado.
    """

    if t is None or len(t) < 1:
        return t

    if len(t) <= 5:
        return sorted(t)

    pivote_5 = pivot5(t)
    menores, pivote, mayores = split_pivot(t, pivote_5)

    t[: len(menores)+1] = qsort_5(menores) + [pivote]
    t[len(menores) :] = [pivote] + qsort_5(mayores)

    return t


def edit_distance(str_1: str, str_2: str)-> int:
    """
    Función que calcula la distancia de edición entre dos cadenas de caracteres.
    
    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.
    
    Return:
        int: distancia de edición entre las dos cadenas.
    """
    
    # control de errores
    if str_1 is None or str_2 is None:
        return -1

    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)

    if len_1 == 0 and len_2 == 0: return 0

    elif (len_1 == 0 and len_2 > 0) or (len_2 == 0 and len_1 > 0):
        return len_1 + len_2

    else:
        # se crea la matriz de distancias
        matriz = np.zeros((len_1+1, len_2+1))

        # se rellena la matriz en primera fila y columna del 1 al len(str)
        matriz[0,:] = np.arange(len_2+1)
        matriz[:,0] = np.arange(len_1+1)
        
        # se recorre la matriz de distancias
        for i in range(1, len_1+1):
            for j in range(1, len_2+1):

                # se obtiene el minimo de las anteriores posiciones con diagonal +1
                if str_1[i-1] == str_2[j-1]:
                    matriz[i,j] = min(matriz[i-1,j] + 1 , matriz[i,j-1] + 1, matriz[i-1,j-1])
                # se obtiene el minimo de las anteriores posiciones con diagonal +1
                else:
                    matriz[i,j] = min(matriz[i-1,j] + 1 , matriz[i,j-1] + 1, matriz[i-1,j-1] + 1) 

        # valor de la ultima posicion
        return int(matriz[-1,-1])


def max_distance_matrix(str_1: str, str_2: str) -> np.ndarray :
    """
    Función que calcula la matriz de distancias máximas entre dos cadenas de caracteres.

    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.

    Return:
        np.ndarray: matriz de distancias máximas entre las dos cadenas.
    """
    # comprobacion de errores
    if str_1 is None or str_2 is None:
        return -1

    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)
    # se crea una matriz de len_1+1 x len_2+1
    matriz = np.zeros((len_1+1, len_2+1))

    # se recorre la matriz
    for i in range(1, len_1+1):
        for j in range(1, len_2+1):
            # si los caracteres son iguales se copia el valor de la diagonal +1
            if str_1[i-1] == str_2[j-1]:
                matriz[i,j] = matriz[i-1,j-1] +1
            # si no, se coge el maximo de la fila y columna anterior
            else:
                matriz[i,j] = max(matriz[i-1,j], matriz[i,j-1])
    
    # devolvemos la matriz obtenida)
    return matriz


def max_subsequence_length(str_1: str, str_2: str) -> int:
    """
    Función que calcula la longitud de la subsecuencia común más larga entre dos cadenas de caracteres.

    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.

    Return:
        int: longitud de la subsecuencia común más larga entre las dos cadenas.
    """
    # comprobacion de errores
    if str_1 is None or str_2 is None:
        return -1

    # se llama a la función que crea la matriz de distancias 
    matriz = max_distance_matrix(str_1, str_2)
                
    # se devuelve el valor de la ultima posicion
    return matriz[-1,-1]


def max_common_subsequence(str_1: str, str_2: str)-> str:
    """
    Función que calcula la subsecuencia común más larga entre dos cadenas de caracteres.

    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.

    Return:
        str: subsecuencia común más larga entre las dos cadenas.
    """
    # comprobacion de errores
    if str_1 is None or str_2 is None:
        return -1

    # se llama a la función que crea la matriz de distancias 
    matriz = max_distance_matrix(str_1, str_2)
            
    # se obtiene la subcadena
    subcadena = ""
    i = len(str_1)
    j = len(str_2)
    """for i, j in zip(reversed(range(0,len_1+1)),reversed(range(0, len_2+1))):
        if str_1[i-1] == str_2[j-1]:
            # aumenta el elemento a la subcadena y avanza en diagonal
            subcadena = str_1[i-1] + subcadena
        elif matriz[i-1,j] > matriz[i,j-1]:
            # solo queremos que avance en i
            j +=1
        else:
            # solo queremos que avance en j
            i += 1"""

    while i > 0 and j > 0:
        # si el caracter coincide se añade a la subcadena, retrocede en diagonal pues no hizo nada
        if str_1[i-1] == str_2[j-1]:
            subcadena = str_1[i-1] + subcadena
            i -= 1
            j -= 1
        # si el valor arriba es mayor al valor a la izquierda se desplaza arriba
        elif matriz[i-1,j] > matriz[i,j-1]:
            i -= 1
        # en caso contrario se desplaza a la izquierda
        else:
            j -= 1
    return subcadena


# Escribir una función min_mult_matrix(l_dims: List[int])-> int que devuelva el m´ınimo numero de productos para multiplicar ´ n matrices cuyas dimensiones estan contenidas en la lista l_dims con n+1 ints, el primero de los cuales nos da las filas de la primera matriz y el resto las columnas de todas ellas.
def min_mult_matrix(l_dims: List[int])-> int:
    """
    Función que calcula el mínimo número de productos para multiplicar n matrices cuyas dimensiones están contenidas en la lista l_dims con n+1 ints, el primero de los cuales nos da las filas de la primera matriz y el resto las columnas de todas ellas.

    Args:
        l_dims: lista con las dimensiones de las matrices.

    Return:
        int: mínimo número de productos para multiplicar n matrices.
    """
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
