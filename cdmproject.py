import pandas as pd
import numpy as np
from scipy.linalg import svd
from numpy.linalg import svd
import matplotlib.pylab as plt
from numpy import array
import os

file = "/files/breast_preprocessed.xls"


def read_data():
    """
    read excel
    """
    dirname = os.path.dirname(__file__)
    filepath = dirname + file
    data = pd.read_excel(filepath, header=None)
    # Taking the whole dataset ignoring the class labels
    X = np.array(data.iloc[0:len(data.index)-2, 1:], dtype = np.float64).transpose()

    return X, data.to_numpy().T


def mysvd(A):
    # get eigen values and eigen vectors from matrix
    S, U = np.linalg.eig(A.dot(A.T))
    # sigma is sqrt of eigen values
    S = np.diag(sorted(np.sqrt(S), reverse=True))
    return U, S


def mean_centered(A):
    """
    subtract mean from A matrix
    return: mean centered matrix
    """
    mean_vector = np.mean(X, axis = 0)
    A = A - mean_vector
    # for i in range(A.shape[0]):
    #     A[i,] = X[i] - mean_vector[i]

    return A


def calculateCovariance(X):
    lenX = X.shape[1]
    covariance = X.T.dot(X)/lenX-1
    return covariance


X, data = read_data()
X = mean_centered(X)

U, S = mysvd(X)
zeros = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
zeros[:, :A.shape[0] - A.shape[1]] = S
S = zeros
XV = U.dot(S)
# U, s, VT = svd(X, full_matrices=False)

# cov_mat = np.cov(eig_vec)
# eig_vec = VT
# eig_val = S
# Make a list of (eigenvalue, eigenvector) tuples
# eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_vec))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
# eig_pairs.sort(key=lambda x: x[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
# for i in eig_pairs:
#     print(i[0])

# Choosing k eigenvectors with the largest eigenvalues
# matrix_w = np.hstack((eig_pairs[0][1].reshape(128, 1), eig_pairs[1][1].reshape(128, 1)))
# print(matrix_w)

# Transforming the samples onto the new subspace
# transformed = matrix_w.T.dot(X)
