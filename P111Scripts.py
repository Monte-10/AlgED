%matplotlib inline
from p115 import *
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import numpy as np
from typing import List, Callable

l_timings = []
l_timings2 = []
list_dim = []
for i in range(11):
    dim = 10+i
    m = np.random.uniform(0., 1., (dim, dim))
    timings = %timeit -o -n 10 -r 5 -q matrix_multiplication(m,m)
    timings2 = %timeit -o -n 10 -r 5 -q np.dot(m,m)
    l_timings.append([dim,timings.best])
    l_timings2.append([dim,timings2.best])
    list_dim.append(dim)

def fit_func_2_times(timings: np.ndarray, func_2_fit: Callable):

    if len(timings.shape) == 1:
        timings = timings.reshape(-1, 1)
    values = func_2_fit(timings[ :, 0]).reshape(-1, 1)
    #normalizar timings
    times = timings[ : , 1] / timings[0, 1]
    #ajustar a los valores en times un modelo lineal sobre los valores en values
    lr_m = LinearRegression()
    lr_m.fit(values, times)
    return lr_m.predict(values)

def func_2_fit(n):
    return n**3

x = np.linspace(10,20,11)
y = [a*a*a/2000 for a in x]

plt.subplot(2,1,1)
plt.plot(list_dim, fit_func_2_times(np.asarray(l_timings), func_2_fit), label="matrix_multiplication()", color = 'r')
plt.title("matrix_multiplication()")
plt.ylabel("Tiempo(s)")
plt.xlabel("Dimension")
plt.subplot(2,1,2)
plt.plot(x,y)
plt.title("O(n^3)")
plt.ylabel("Tiempo(s)")
plt.xlabel("Dimension")
plt.suptitle("O(n^3) - matrix_multiplication()")
plt.tight_layout()
plt.show()
