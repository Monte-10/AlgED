import numpy as np
import P12A as minheap
import P12B as pque

# Función que devuelve el numero dentro del array con K numeros mayores que este
# ejemplo: array = [1,2,3,4,5] K = 2 entonces numero = 3
# bastaria con obtener la raiz del heap y realizar heapify, se realizará K veces.
def select_min_heap(h: np.ndarray, k: int)-> int:

    # invierte el array 
    for i in range(0, len(h)):
        h[i] = h[i]*-1

    # realiza el min_heap sobre el array invertido
    h = minheap.create_min_heap(h)
    for j in range (0, k):
        # podemos emplear un delete de la libreria np
        h = np.delete(h,0)
         # realiza el min_heap sobre el array invertido
        h = minheap.create_min_heap(h)
    
    return h[0]*-1



h=[0,1,3,4,8]
print(select_min_heap(h,1))