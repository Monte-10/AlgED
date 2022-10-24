import numpy as np
from typing import List, Callable

def init_cd(n: int) -> np.ndarray:
    array = [-1] * n
    return array

def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    if p_cd[rep_1] < p_cd[rep_2]:
        p_cd[rep_1] = rep_2
        return rep_2
    
    elif p_cd[rep_1] > p_cd[rep_2]:
        p_cd[rep_2] = rep_1
        return rep_1
    
    else:
        p_cd[rep_2] = rep_1
        rep_1 -= -1
        return rep_1

def find(ind: int, p_cd: np.ndarray) -> int:
    aux = ind
    while p_cd[aux] >= 0:
        aux = p_cd[aux]
        
    while p_cd[ind] >= 0:
        aux2 = p_cd[ind]
        p_cd[ind] = aux
        ind = aux2
    
    return aux

def cd_2_dict(p_cd: np. ndarray) -> Dict:
    