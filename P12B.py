import numpy as np
import P12A as min_heap


#función que inicializa una cola de prioridad vacia
def q_ini() -> np.ndarray:
    q = []
    return q

def pq_insert(h: np.ndarray, k: int)-> np.ndarray:
    return min_heap.insert_min_heap(h, k)

def pq_remove(h: np.ndarray)-> np.ndarray:
    elim = h[0]
    h = np.delete(h,0)
    return elim, h

