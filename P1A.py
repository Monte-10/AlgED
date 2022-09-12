import numpy as np

def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray :

    result = np.zeros((len(m_1),len(m_1)))
    
    for i in range(len(m_1)):

        for j in range(len(m_2[0])):

            for k in range(len(m_2)):
                result[i][j] += m_1[i][k] * m_2[k][j]

    return result

l_timings = []
for i in range(11):
    dim = 10+i
    m = np.random.uniform(0., 1., (dim, dim))
    timings = %timeit -o -n 10 -r 5 -q matrix_multiplication(m, m)
    l_timings.append([dim, timings.best])

print(l_timings)