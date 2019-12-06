import pandas as pd
import numpy as np
from scipy.linalg import svd
import re
from numpy import array
from re import findall


# Read the file
def read_file():
    # data = pd.read_excel("breast_preprocessed.xls", names = np.array(range(129)))
    data = pd.read_excel("test1.xls", names = np.array(range(129)))
    #
    data = data.replace(0, np.NaN)
    data = data.drop_duplicates()
    #
    # data.iloc[0:len(data.index)-3,1:]
    matrix = np.array(data.iloc[0:len(data.index) - 3, 1:], dtype = np.float32)
    # matrix = np.array([[3, 2, 2], [2, 3, -2]])
    return matrix


# def transpose(m):
#     rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
#
#     return rez


def eignenVal(m):
    m = np.array(m)
    return  np.linalg.eigvals(m)


def eignenVec(m):
    m = np.array(m)
    return  np.linalg.eig(m)


value_matrix = read_file()
n, m = value_matrix.shape
if n > m:
    A = np.dot(value_matrix.T, value_matrix)
else:
    A = np.dot(value_matrix, value_matrix.T)

eigV, sigma = eignenVec(A)
U, s, VT = svd(A)
print(eigV)
print(sigma)
print(U)
print(s)
print(VT)