import numpy as np


def min_heapify(h: np.ndarray, i: int):
    while 2*i+1 < len(h):
        n_i = I

        if h[i] > h[2*i+1]:
            n_i = 2*i+1

        if 2*i+2 < len(h) and h[i] > h[2*i+2] and h[2*i+2] < h[n_i]:
            n_i = 2*i+2

        if n_i > i:
            h[i], h[n_i] = h[n_i], h[i]
            i = n_i
        
        else:
            return

def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    h += [k]
    j =len(h) - 1

    while j>= 1 and h[(j-1) // 2] > h[j]:
        h[(j-1) // 2], h[j], h[(j-1) // 2]
        j = (j-1) // 2

def create_min_heap(h: np.ndarray):
