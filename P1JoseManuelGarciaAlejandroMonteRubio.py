import numpy as np

def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray :

    result = np.zeros((len(m_1),len(m_1)))
    
    for i in range(len(m_1)):

        for j in range(len(m_2[0])):

            for k in range(len(m_2)):
                result[i][j] += m_1[i][k] * m_2[k][j]

    return result

def rec_bb(t: list, f: int, l: int, key: int) -> int :
    
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

    #Mientras no estes en el nodo hoja
    while 2*i+1 < len(h):
        n_i = i

        #Si nodo enviado > que hijo izquierdo entonces guardas posicion hijo
        if h[i] > h[2*i+1]:
            n_i = 2*i+1

        #Comprueba si hay hijo derecho y si lo hay comprueba nodo estamos > derecho y si posicion guardada es > hijo derecho, si ambos son mayores se actualiza posicion
        if 2*i+2 < len(h) and h[i] > h[2*i+2] and h[2*i+2] < h[n_i]:
            n_i = 2*i+2

        #Si no hijo izq y drc es menor, y nodo que hemos guardado tiene valor menor o posicion baja, intercambia valores de i y n_i
        if n_i > i:
            h[i], h[n_i] = h[n_i], h[i]
            i = n_i

        #Si no hace nada, devuelve
        else:
            return

# inserta un elemento en el heap 
def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    if h is None: 
        return [k]
    #primero añade el elemento al heap
    h = np.append(h,k)
    j = len(h) - 1

    # coloca el elemento en su lugar correspondiente
    while j >= 1 and h[(j-1) // 2] > h[j]:
        h[(j-1) // 2], h[j] = h[j], h[(j-1) // 2]
        j = (j-1) // 2
    
    return h
    
def create_min_heap(h: np.ndarray):

    # opcion 1: realizar un heapify sobre el array pasado por argumento tantas veces como sea necesario
    # esto permite evitar la necesidad de realizar un segundo array, de arriba a abajo
    """j = (len(h)-1) // 2
    for i in range(0,j+1):
        for k in range(0,i+1):
            min_heapify(h,k)
            
    return h"""

    # opcion 2: realiza un heapify de los padres de todos los subarboles de abajo arriba
    k = (len(h)-1)//2
    while k >= 0:
        min_heapify(h, k)
        k-=1
    return h
    
#función que inicializa una cola de prioridad vacia
def pq_ini() -> np.ndarray:
    q = []
    return q

def pq_insert(h: np.ndarray, k: int)-> np.ndarray:
    return insert_min_heap(h, k)

def pq_remove(h: np.ndarray)-> np.ndarray:
    if h is None or len(h) == 0: return None
    
    elim = h[0]
    h = np.delete(h,0)
    min_heapify(h,0)
    return elim, h

"""h=[1,2,8,5,3]
h = min_heap.create_min_heap(h)
print(pq_remove(h))"""

import numpy as np

# Función que devuelve el numero dentro del array con K numeros mayores que este
# ejemplo: array = [1,2,3,4,5] K = 2 entonces numero = 3
# bastaria con obtener la raiz del heap y realizar heapify, se realizará K veces.
def select_min_heap(h: np.ndarray, k: int)-> int:
    
    aux = h.copy()
    print(aux)
    # invierte el array 
    for i in range(0,len(aux)):
        aux[i] = aux[i]*-1
    # se cogen los k primeros
    aux_mh = aux[:k]
    
    # realiza el min_heap sobre el array invertido
    aux_mh = create_min_heap(aux_mh)
    
    for i in range (k, len(h)):
        if aux[i] > aux_mh[0]:
            aux_mh[0] = aux[i]
            min_heapify(aux_mh, 0)

    return aux_mh[0]*-1


