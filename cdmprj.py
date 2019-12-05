import pandas as pd
import numpy as np
import re
from numpy import array
from re import findall


# Read the file
def read_file():
    data = pd.read_excel("breast_preprocessed.xls", names = np.array(range(129)))

    data = data.replace(0, np.NaN)
    data = data.drop_duplicates()

    data.iloc[0:len(data.index)-3,1:]
    value_matrix = np.array(data.iloc[0:len(data.index)-3, 1:], dtype = np.float16)
    return value_matrix


def transpose(m):
    # for row in m:
    #     print(row)
    rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    # print("\n")
    # for row in rez:
    #     print(row)
    return rez


def eignenVal(m):
    m = np.array(m)
    return  np.linalg.eigvals(m)


def eignenVec(m):
    m = np.array(m)
    return  np.linalg.eig(m)


value_matrix = read_file()
transpose_matrix = transpose(value_matrix)
A = value_matrix.dot(transpose_matrix)
eigV, sigma = eignenVec(A)
print(eigV)
print(sigma)