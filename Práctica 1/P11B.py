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

l_times = []
for i, size in enumerate(range(5, 15)):
    t = list(range(2**i * size))
    key = 8
    timings = %timeit -n 100 -r 10 -o -q rec_bb(t, 0, len(t) - 1, key)
    l_times.append([len(t), timings.best])
    times = np.array(l_times)