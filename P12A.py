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

def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    h = np.append(h,k)
    j = len(h) - 1

    while j >= 1 and h[(j-1) // 2] > h[j]:

        h[(j-1) // 2], h[j] = h[j], h[(j-1) // 2]
        j = (j-1) // 2
    
    return h
def create_min_heap(h: np.ndarray):
    i = 0
    k = 0
    j = (len(h)-1) // 2
    for i in range(0,j+1):
        for k in range(0,i+1):
            min_heapify(h,k)
            
    
    return h

