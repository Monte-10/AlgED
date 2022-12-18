import numpy as np
import itertools
from typing import List, Tuple, Dict, Callable, Iterable, Union


def split(t: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Función que reparte los elementos en un array entre dos arrays con los elementos menores y mayores que el primer valor en el
    array pasado por argumento.

    Args:
        t: array a dividir.

    Return:
        Tuple[np.ndarray, int, np.ndarray]: tupla con los elementos menores, el pivote y los mayores.
        En caso de error devuelve None.
    """
    # comprueba errores
    if len(t) <= 0 or t is None:
        return None
    # se obtienen los elementos mayores a t[0]
    mayores = t[t > t[0]]

    # se obtienen los elementos menores a t[0]
    menores = t[t < t[0]]

    return menores, t[0], mayores


def qsel(t: np.ndarray, k: int) -> Union[int, None]:
    """
    Función que devuelve el valor del elemento que ocupa la posicion k 
    en el array ordenado empleando recursividad.

    Args:
        t: array del que se quiere obtener el valor del elemento en k si el array estuviera ordenado.
        k: índice del elemento cuyo valor se quiere encontrar en el array ordenado.

    Return:
        Union[int, None]: valor del elemento en el índice k del array ordenado.
        En caso de error devuelve None.
    """
    print("Estoy en qsel:", t)
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
            return qsel(mayores, k - (len(menores) + 1))


def qsel_nr(t: np.ndarray, k: int) -> Union[int, None]:
    """
    Función que devuelve el valor del elemento que ocupa la posicion k 
    en el array ordenado sin emplear recursividad.

    Args:
        t: array del que se quiere obtener el valor del elemento en k si el array estuviera ordenado.
        k: índice del elemento cuyo valor se quiere encontrar en el array ordenado.

    Return:
        Union[int, None]: valor del elemento en el índice k del array ordenado.
        En caso ed error devuelve None.
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
                k = k - (len(menores) + 1)
                t = mayores

        return


def split_pivot(t: np.ndarray, mid: int) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Función que reparte los elementos en un array entre dos arrays con los elementos menores y mayores que el valor mid pasado por argumento.

    Args:
        t: array a dividir.
        mid: pivote empleado para dividir el array.

    Return:
        Tuple[np.ndarray, int, np.ndarray]: tupla con los elementos menores, el pivote y los mayores.
        En caso de error devuelve None.
    """
    # comprobacion de error
    if t is None or len(t) < 1:
        return None

    # empleamos un metodo diferente al usado anteriormente pues daba problemas de ejecucion
    mayores = [y for y in t[0: len(t)] if y > mid]
    menores = [x for x in t[0: len(t)] if x < mid]

    return (menores, mid, mayores)


def pivot5(t: np.ndarray) -> int:
    """
    Función que devuelve el pivote de un array de 5 elementos empleando el metodo mediana de medianas y llamando a la función
    qsel5_nr.

    Args:
        t: array del que se quiere obtener el pivote.

    Return:
        int: pivote.
        En caso de error devuelve None.
    """
    # comprueba errores
    if len(t) <= 0 or t is None:
        return None

    # si la longitud es menor a 5 no se calculan las medianas
    if len(t) < 5:
        return qsel5_nr(t, 0)

    # se obtienen las medianas
    num_group = len(t)//5
    # indice donde se encuenta la mediana en una lista de 5 elementos ordenada
    indices = 5//2

    # se descartan grupos de menor tamaño a 5
    sublistas = [t[i:i + 5] for i in range(0, len(t), 5)][:num_group]

    # se obtienen las medianas
    medianas = [sorted(sub)[indices] for sub in sublistas]

    # si el numero de medianas es menor a 5 se ordena
    if len(medianas) <= 5:
        if len(medianas) % 2 == 0:  # tamaño par
            pivot = sorted(medianas)[len(medianas)//2-1]
        else:  # tamaño impar
            pivot = sorted(medianas)[len(medianas)//2]
    else:  # si la longitud es menor se llama a quickselect con k actualizado
        pivot = qsel5_nr(medianas, len(medianas)//2)
    return pivot


def qsel5_nr(t: np.ndarray, k: int) -> Union[int, None]:
    """
     Devuelve el elemento en el indice k de una ordenacion del array de entrada de forma no recursiva.

    Args:
        t: array del que se quiere obtener el valor del elemento en k si el array estuviera ordenado.
        k: índice del elemento cuyo valor se quiere encontrar en el array ordenado.

    Return:
        Union[int, None]: valor del elemento en el índice k del array ordenado.
        En caso de error devuelve None.
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
        elif k < len(menores):
            t = menores
        else:
            t = mayores
            k = k - (len(menores) + 1)

    return sorted(t)[k]


def qsort_5(t: np.ndarray) -> np.ndarray:
    """
    Función recursiva que devuelve una ordenación del array pasado por agumento empleando las funciones pivote_5 y split_pivot.

    Args:
        t: array que se quiere ordenar.

    Return:
        np.ndarray: array ordenado.
        En caso de error devuelve None.
    """

    if t is None or len(t) < 1:
        return None

    if len(t) <= 5:
        return sorted(t)

    pivote_5 = pivot5(t)
    menores, pivote, mayores = split_pivot(t, pivote_5)

    t[: len(menores)+1] = qsort_5(menores) + [pivote]
    t[len(menores):] = [pivote] + qsort_5(mayores)

    return t


def edit_distance(str_1: str, str_2: str) -> int:
    """
    Función que calcula la distancia de edición entre dos cadenas de caracteres.
    Esta versión emplea dos arrays en vez de una matriz para minimizar el uso de memoria, la version usando matriz
    se encuentra comentada en el codigo.

    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.

    Return:
        int: distancia de edición entre las dos cadenas.
        En caso de error devuelve -1.
    """

    # control de errores
    if str_1 is None or str_2 is None:
        return -1

    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)

    if len_1 == 0 and len_2 == 0:
        return 0

    elif (len_1 == 0 and len_2 > 0) or (len_2 == 0 and len_1 > 0):
        return len_1 + len_2

    else:
        # se eutilizan dos arrays
        # utilizamos dos arrays
        fila1 = np.arange(len_2+1)
        fila2 = np.zeros(len_2+1)

        # se crea la matriz de distancias
        """matriz = np.zeros((len_1+1, len_2+1))"""

        # se rellena la matriz en primera fila y columna del 1 al len(str)
        """matriz[0,:] = np.arange(len_2+1)
        matriz[:,0] = np.arange(len_1+1)"""

        # se recorre la matriz de distancias
        for i in range(1, len_1+1):

            fila2[0] = i
            for j in range(1, len_2+1):
                # el primer valor de la siguiente fila corresponde a los valores de 0 a len1 + 1

                # se obtiene el minimo de las anteriores posiciones con diagonal +0
                if str_1[i-1] == str_2[j-1]:
                    fila2[j] = min(fila2[j-1] + 1, fila1[j] + 1, fila1[j-1])
                    """matriz[i,j] = min(matriz[i-1,j] + 1 , matriz[i,j-1] + 1, matriz[i-1,j-1])"""
                # se obtiene el minimo de las anteriores posiciones con diagonal +1
                else:
                    fila2[j] = min(fila2[j-1] + 1, fila1[j] +
                                   1, fila1[j-1] + 1)
                    """matriz[i,j] = min(matriz[i-1,j] + 1 , matriz[i,j-1] + 1, matriz[i-1,j-1] + 1)"""

            # para pasar a la siguiente fila
            fila1 = np.copy(fila2)
            fila2[fila2 > 0] = 0

        # valor de la ultima posicion
        """return int(matriz[-1,-1])"""
        return int(fila1[-1])


def max_subsequence_length(str_1: str, str_2: str) -> int:
    """
    Función que calcula la longitud de la subsecuencia común más larga entre dos cadenas de caracteres, esta subcadena
    no tiene porque ser necesariamente consecutiva, por ejemplo con ace y abcde se devolvería longitud 3.
    Esta versión emplea dos arrays en vez de una matriz para consumir menos memoria.

    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.

    Return:
        int: longitud de la subsecuencia común más larga entre las dos cadenas.
        En caso de error devuelve -1.
    """
    # comprobacion de errores
    if str_1 is None or str_2 is None:
        return -1

    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)

    # utilizamos dos arrays
    fila1 = np.zeros(len_2+1)
    fila2 = np.zeros(len_2+1)

    # se recorre por filas
    for i in range(1, len_1+1):

        # para pasar a la siguiente fila
        fila1 = np.copy(fila2)
        fila2[fila2 > 0] = 0

        # para cada posicion de las listas, es decir columnas
        for j in range(1, len_2+1):
            # # si los caracteres son iguales se copia el valor de la diagonal +1
            if str_1[i-1] == str_2[j-1]:
                fila2[j] = fila1[j-1] + 1
            # si no, se coge el maximo de la fila y columna anterior
            else:
                fila2[j] = max(fila2[j-1], fila1[j])

    # se devuelve el ultimo valor
    return int(fila2[-1])


def max_common_subsequence(str_1: str, str_2: str) -> str:
    """
    Función que calcula la subsecuencia común más larga entre dos cadenas de caracteres, no es necesario
    que la subcadena sea consecutiva, es decir, si tenemos ace y abcde, devolverá como resultado ace.

    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.

    Return:
        str: subsecuencia común más larga entre las dos cadenas.
        En caso de error devuelve None.
    """
    # comprobacion de errores
    if str_1 is None or str_2 is None:
        return None

    # se llama a la función que crea la matriz de distancias
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
                matriz[i, j] = matriz[i-1, j-1] + 1
            # si no, se coge el maximo de la fila y columna anterior
            else:
                matriz[i, j] = max(matriz[i-1, j], matriz[i, j-1])

    # se obtiene la subcadena
    subcadena = ""
    i = len_1
    j = len_2

    while i > 0 and j > 0:
        # si el caracter coincide se añade a la subcadena, retrocede en diagonal pues no hizo nada
        if str_1[i-1] == str_2[j-1]:
            subcadena = str_1[i-1] + subcadena
            i -= 1
            j -= 1
        # si el valor arriba es mayor al valor a la izquierda se desplaza arriba
        elif matriz[i-1, j] > matriz[i, j-1]:
            i -= 1
        # en caso contrario se desplaza a la izquierda
        else:
            j -= 1
    return subcadena


# Escribir una función min_mult_matrix(l_dims: List[int])-> int que devuelva el m´ınimo numero de productos para multiplicar ´ n matrices cuyas dimensiones estan contenidas en la lista l_dims con n+1 ints, el primero de los cuales nos da las filas de la primera matriz y el resto las columnas de todas ellas.
def min_mult_matrix(l_dims: List[int]) -> int:
    """
    Función que calcula el mínimo número de productos para multiplicar n matrices cuyas dimensiones están contenidas en la lista l_dims con n+1 ints, 
    el primero de los cuales nos da las filas de la primera matriz y el resto las columnas de todas ellas.
    Se asume compatibilidad de matrices, es decir A.columnas = B.filas.

    Args:
        l_dims: lista con las dimensiones de las matrices.

    Return:
        int: mínimo número de productos para multiplicar n matrices.
        En caso de error devuelve -1.
    """
    # control de errores
    if len(l_dims) < 2 or l_dims is None:
        return -1

    # si solo hay dos es suficiente con multiplicar ambas
    if len(l_dims) == 2:
        return l_dims[0] * l_dims[1]

    # se obtiene el numero de matrices
    n = len(l_dims) - 1

    # se crea la matriz de costes inicializada a 0, necesitamos la diagonal a 0 pero es mas eficiente inicializarla directamente a 0
    matriz = np.zeros((n, n))

    # lista auxiliar para guardar todos los posibles valores
    valores = []

    # los valores de i corresponden con la diagonal, i=1 es la primera diagonal
    for i in range(1, n):
        # los valores de j corresponden a las posiciones en la diagonal
        for j in range(n-i):
            for k in range(j, j+i):
                valores.append(matriz[j, k] + matriz[k+1, j+i] +
                               l_dims[j]*l_dims[k+1]*l_dims[j+i+1])

            # se obtiene el minimo de la multiplicacion de las matrices
            matriz[j, j+i] = min(valores)
            valores.clear()

    return int(matriz[0, -1])


def max_common_substring(str_1: str, str_2: str) -> str:
    """Función que calcula la subsecuencia común más larga entre dos cadenas de caracteres, es necesario
    que la subcadena sea consecutiva, es decir, si tenemos abce y abcde, devolverá como resultado ab.

    Args:
        str_1: primera cadena de caracteres.
        str_2: segunda cadena de caracteres.

    Return:
        str: subsecuencia común consecutiva más larga entre las dos cadenas.
        None en caso de error.
    """
    if str_1 is None or str_2 is None:
        return None

    if len(str_1) == 0 or len(str_2) == 0:
        return ""

    # se obtienen las longitudes de las cadenas
    len_1 = len(str_1)
    len_2 = len(str_2)

    # se inicializa la matriz
    matrix = np.zeros((len_1+1, len_2+1))

    # variable auxiliar para saber la maxima distancia
    max = 0
    ret = []
    for i in range(1, len_1+1):
        for j in range(1, len_2+1):
            if str_1[i - 1] == str_2[j - 1]:
                if i == 1 or j == 1:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = matrix[i - 1][j - 1] + 1
                if matrix[i][j] > max:
                    max = matrix[i][j]
                    ret = [str_1[i - int(max): i]]
                elif matrix[i][j] == max:
                    ret.append(str_1[i - int(max): i])
            else:
                matrix[i][j] = 0

    # print(matrix)
    return ret
