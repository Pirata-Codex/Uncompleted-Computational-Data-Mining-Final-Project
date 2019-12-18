import pandas as pd
import numpy as np
from scipy.linalg import svd
from numpy.linalg import svd
import matplotlib.pylab as plt
from numpy import array
# import xlsxwriter

file = "breast_preprocessed.xls"


def mysvd(A):
    # get eigen values and eigen vectors from matrix
    S, V = np.linalg.eig(A.T.dot(A))
    # sigma is sqrt of eigen values
    S = np.diag(np.sqrt(S))
    # Left Orthogonal Matrix (U) is
    # A = USVT then U = AVS-1
    U = A.dot(V).dot(np.linalg.inv(S))
    # reconstructed = U.dot(S).dot(V.T)
    # workbook = xlsxwriter.Workbook('arrays.xlsx')
    # worksheet = workbook.add_worksheet()
    #
    # for col, data in enumerate(array):
    #     worksheet.write_column(row, col, data)
    return U, S, V.T

def readdata():
    """
    read excel
    """
    data = pd.read_excel(file, names = np.array(range(129)))
    # extract matric from dataframe
    # Taking the whole dataset ignoring the class labels
    X = np.array(data.iloc[0:len(data.index)-2, 1:], dtype = np.float32)

    return X


def mean_centered(A):
    """
    subtract mean from A matrix
    return: mean centered matrix
    """
    mean_vector = np.mean(X, axis=1)
    for i in range(A.shape[0]):
        A[i,] = X[i] - mean_vector[i]

    return A


def calculateCovariance(X):
    lenX = X.shape[1]
    covariance = X.T.dot(X)/lenX
    return covariance


X = readdata()
X = mean_centered(X)

mat_cov = calculateCovariance(X)
U, S, VT = mysvd(X)
# U, s, VT = svd(X, full_matrices=False)
loading_matrix = U
score_matrix = s.dot(VT)
reconstructed = U.dot(S).dot(VT)

cov_mat = np.cov(eig_vec)
eig_vec = VT
eig_val = S
# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_vec))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])

# Choosing k eigenvectors with the largest eigenvalues
matrix_w = np.hstack((eig_pairs[0][1].reshape(y,1), eig_pairs[1][1].reshape(y,1)))

