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

def cd_2_dict(p_cd: np.ndarray) -> dict:
    dict = {}
    for i in range(len(p_cd)):
        if p_cd[i] < 0:
            dict[i] = [i]

    for i in range(len(p_cd)):
        if p_cd[i] >= 0:
            dict[find(i, p_cd)].append(i)

                
    return dict
    
def ccs(n: int, l: List)-> dict:
    table = init_cd(n)
    for u,v in l:
        rep_u = find(u,table)
        rep_v = find(v,table)

        if rep_u != rep_v:
            union(rep_u, rep_v, table)

    dict = cd_2_dict(table)

    return dict

n = 7
l = ((0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (5, 6))
print(ccs(n,l))
    