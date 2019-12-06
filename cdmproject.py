import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pylab as plt
from numpy import array


data = pd.read_excel("test1.xls", names = np.array(range(129)))
data = data.replace(0, np.NaN)
A = np.array(data.iloc[0:len(data.index), 1:], dtype = np.float32)

temp = A.T.dot(A)
S, V = np.linalg.eig(temp)
S = np.diag(np.sqrt(S))
U = A.dot(V).dot(np.linalg.inv(S))

print('------------------')
print("Matrix U : \n", U)
print("Matrix S : \n", S)
print("Matrix V : \n", V)

print('\n------------------')
reconstructed = U.dot(S).dot(V.T)
print("Reconstructed matrix : \n", reconstructed)
print("Original matrix : \n", A)
