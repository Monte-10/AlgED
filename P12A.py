import numpy as np


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
    
h = [5,2,3,1,4]
#print(create_min_heap(h))