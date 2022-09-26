import numpy as np
import P12A as minheap

# Función que devuelve el numero dentro del array con K numeros mayores que este
# ejemplo: array = [1,2,3,4,5] K = 2 entonces numero = 3
def select_min_heap(h: np.ndarray, k: int)-> int:

    h = minheap.create_min_heap(h)
    for i in range(0, len(h)):
        h[i] = h[i]*-1

    
    return h

h = [3,5,4,8,1,2,9]
print(select_min_heap(h,3))