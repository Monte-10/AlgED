import numpy as np

def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray :

    result = [[0,0,0],
             [0,0,0],
             [0,0,0]]
    
    for i in range(len(m_1)):

        for j in range(len(m_2[0])):

            for k in range(len(m_2)):
                result[i][j] +=m_1[i][k] * m_2[k][j]

    return result

m1 = np.array([[1,2,3],[4,5,6], [7,8,9]])
m2 = np.array([[9,8,7],[6,5,4],[3,2,1]])

print(matrix_multiplication(m1, m2))