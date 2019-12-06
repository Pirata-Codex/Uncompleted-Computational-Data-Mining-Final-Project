import pandas as pd
import numpy as np
from scipy.linalg import svd
import matplotlib.pylab as plt
from numpy import array

# read excel
data = pd.read_excel("test1.xls", names = np.array(range(129)))
# replace NaN with zero
data = data.replace(0, np.NaN)
# extract matric from dataframe
A = np.array(data.iloc[0:len(data.index), 1:], dtype = np.float32)

# A.T . A
temp = A.T.dot(A)
# get eigen values and eigen vectors from matrix
S, V = np.linalg.eig(temp)
# sigma is sqrt of eigen values
S = np.diag(np.sqrt(S))
# Left Orthogonal Matrix (U) is 
# A = USVT then U = AVS-1
U = A.dot(V).dot(np.linalg.inv(S))

print('------------------')
print("Matrix U : \n", U)
print("Matrix S : \n", S)
print("Matrix V : \n", V)

print('\n------------------')
reconstructed = U.dot(S).dot(V.T)
print("Reconstructed matrix : \n", reconstructed)
print("Original matrix : \n", A)
