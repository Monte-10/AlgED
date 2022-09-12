import numpy as np

def rec_bb(t: list, f: int, l: int, key: int) -> int :
    
    mid = (f+l) // 2

    if t[mid] == key:
        return mid
    
    elif t[mid] < key:
        return rec_bb(t, mid+1, l, key)

    elif t[mid > key]:
        return rec_bb(t, f, mid-1, key)

    else:
        return None

print(rec_bb((1,2,3,4,5,6,7,8,9,10),1,10,7))