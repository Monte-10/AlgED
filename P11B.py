import numpy as np

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
