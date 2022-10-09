import numpy as np
from typing import List, Callable



def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray :
    """
    Función que recibe dos matrices numpy de dimensiones compatibles y realiza 
    una multiplicacion entre ambas.
    
    Args:
        m_1: Matriz 1 
        m_2: Matriz 2
    
    Return: 
        result: Matriz resultante de multiplicar la matriz 1 con la matriz 2
    """
    result = np.zeros((len(m_1),len(m_1)))
    
    for i in range(len(m_1)):

        for j in range(len(m_2[0])):

            for k in range(len(m_2)):
                result[i][j] += m_1[i][k] * m_2[k][j]

    return result



def rec_bb(t: list, f: int, l: int, key: int) -> int :
    """
    Función que recibe una lista con su primer y último indice, asi como una clave a buscar y
    aplica una versión recursiva de la búsqueda binaria para encontrar la posición
    de la clave(key) entre el indice primero(first) y el indiice último(last).
    
    Args:
        t: Lista sobre la que se buscará la clave
        f: Primer índice
        l: Último índice
        key: Clave a buscar

    
    Return: 
        Si encuentra la clave devuelve el índice donde es encontrado, de no encontrarse 
        la clave devuelve None.
    """
    if l>=f:

        mid = (f+l) // 2

        if t[mid] == key:
            return mid
        
        elif t[mid] < key:
            return rec_bb(t, mid+1, l, key)

        else:
            return rec_bb(t, f, mid-1, key)

    else:
        return None



def bb(t: list, f: int, l: int, key: int) -> int :
    """
    Versión iterativa de la función rec_bb.

    Args:
        t: Lista sobre la que se buscará la clave
        f: Primer índice
        l: Último índice
        key: Clave a buscar

    
    Return: 
        Si encuentra la clave devuelve el índice donde es encontrado, de no encontrarse 
        la clave devuelve None.
    """

    while f <= l:

        mid = (f+l) // 2

        if t[mid] == key:
            return mid

        elif t[mid] < key:
            f = mid + 1

        elif t[mid] > key:
            l = mid -1
    
    return None

def min_heapify(h: np.ndarray, i: int):
    """
    Función que recibe un array de numpy aplica heapify sorbre el elemento situado en el índice indicado como argumento.
    Esto es que dado un nodo se asegura que el subárboll desde ese nodo es un min heap.

    Args:
        h: Array numpy que contiene el nodo sobre el que se quiere realizar el heapify
        i: Índice del nodo sobre el que se quiere realizar heapify
    """
    # Mientras no estes en el nodo hoja
    while 2*i+1 < len(h):
        n_i = i

        # Si nodo enviado > que hijo izquierdo entonces guardas posicion hijo
        if h[i] > h[2*i+1]:
            n_i = 2*i+1

        #Comprueba si hay hijo derecho y si lo hay comprueba nodo estamos > derecho y si posicion guardada es > hijo derecho, si ambos son mayores se actualiza posicion"""
        if 2*i+2 < len(h) and h[i] > h[2*i+2] and h[2*i+2] < h[n_i]:
            n_i = 2*i+2

        #Si no hijo izq y drc es menor, y nodo que hemos guardado tiene valor menor o posicion baja, intercambia valores de i y n_i"""
        if n_i > i:
            h[i], h[n_i] = h[n_i], h[i]
            i = n_i

        #Si no hace nada, devuelve
        else:
            return


def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    """
    Función que inserta en el min heap pasado por argumento un elemento.

    Args:
        h: Min heap en el que se quiere insertar el elemento
        k: Elemento que se quiere insertar en el min heap

    Return:
        h: Min heap con el nuevo elemento insertado.
        Si el min heap pasado por argumento se encuentra vacio o es None se devuelve
        un array numpy que contenga el elemento k.
    """

    #Comprobación"""
    if h is None or len(h) < 1: 
        return [k]

    #primero añade el elemento al heap"""
    h = np.append(h,k)
    j = len(h) - 1

    #coloca el elemento en su lugar correspondiente"""
    while j >= 1 and h[(j-1) // 2] > h[j]:
        h[(j-1) // 2], h[j] = h[j], h[(j-1) // 2]
        j = (j-1) // 2
    
    return h
    

def create_min_heap(h: np.ndarray):
    """
    Función que crea un min heap sobre el array numpy pasado por argumento.

    Args:
        h: Array numpy sobre el que se quiere crear el min heap
    """

    #realiza un heapify de los padres de todos los subarboles de abajo arriba"""
    k = (len(h)-1)//2
    while k >= 0:
        min_heapify(h, k)
        k-=1
    return
    

def pq_ini() -> np.ndarray:
    """
    Inicializa una cola de prioridad vacía

    Return:
        q: Cola de proiridad vacía
    """

    q = []
    return q

def pq_insert(h: np.ndarray, k: int)-> np.ndarray:
    """
    Función que inserta el elemento pasado por argumento en la cola de prioridad y devuelve 
    la nueva cola

    Args:
        h: Cola en la que se quiere insertar el elemento
        k: Elemento a insertar

    Return:
        Cola de proiridad vacía
    """
    
    return insert_min_heap(h, k)

def pq_remove(h: np.ndarray)-> np.ndarray:
    """
    Función que elimina el elemento con el menor valor de prioridad de la cola de prioridad pasada por 
    argumento.

    Args:
        h: Cola de prioridad de la que se quiere eliminar el elemento con menor valor de prioridad

    Return:
        Devuelve el elemento eliminado y la nueva cola. 
        Si la cola pasada por argumento está vacía devuelve None

    """
    #Comprobación
    if h is None or len(h) == 0: return None
    
    elim = h[0]
    h = np.delete(h,0)
    min_heapify(h,0)
    return elim, h


def select_min_heap(h: np.ndarray, k: int)-> int:
    """
    Función que devuelve el elemento en la posición K pasada por argumento si el array pasado 
    por argumento estuviera estuveira ordenado de menor a mayor
    ejemplo: array = [3,1,2,4,5] ordenado = [1,2,3,4,5] K = 2 entonces numero = 2
    
    Args:
        h: Array del que se quiere obtener el elemento en la posición K de estar ordenado de menor a mayor
        k: Posición dentro del array ordenado

    Return:
        Elemento en la posición K dentro del array ordenado

    """
    aux = h.copy()
    # invierte el array 
    aux = np.multiply(aux, -1)
    # se cogen los k primeros
    aux_mh = aux[:k]
    
    # realiza el create_min_heap sobre el array invertido
    create_min_heap(aux_mh)

    for i in range (k, len(h)):
        if aux[i] > aux_mh[0]:
            aux_mh[0] = aux[i]
            min_heapify(aux_mh, 0)
    
    return aux_mh[0]*-1


def obtiene_menores(h: np.ndarray)-> np.ndarray:
    """
    Función que encuentra los 2 menores elemento dentro del array pasado por argumento

    Args: 
        h: Array del que se quieren encontrar los 2 menores elementos
    
    Return:
        Array que contiene los dos menores elementos.
        Si el array tiene 2 o menos elementos se devuelven esos elementos.
        Si el array pasado por argumento está vacío devuelve None.
    """

    # si el array tiene 2 o menos elementos se devuelven estos
    if len(h) == 1 or len(h) == 2: return h

    # si esta vacio se devuelve None
    if len(h) == 0 or h is None: return None

    m1 = h[0]
    m2 = h[1]
    for i in range(2, len(h)):
        if m1 > h[i] or m2 > h[i]:
            if m1 > m2:
                m1 = h[i]
            else:
                m2 = h[i]

    return [m1,m2]

