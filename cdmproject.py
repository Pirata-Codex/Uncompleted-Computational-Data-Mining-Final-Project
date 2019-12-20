import pandas as pd
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import os

file = "/files/breast_preprocessed.xls"
varRetained = 0.90
colors = {'luminal': '#0D76BF', 'non-luminal': '#00cc96'}


def read_data():
    """
    read excel file
    """
    dirname = os.path.dirname(__file__)
    filepath = dirname + file
    data = pd.read_excel(filepath, header=None)
    # Taking the whole dataset ignoring the class labels
    X = np.array(data.iloc[0:len(data.index)-2, 1:], dtype = np.float64).transpose()

    return X, data


def mean_centered(A):
    """
    subtract mean from A matrix
    return: mean centered matrix
    """
    mean_vector = np.mean(X, axis = 0)
    A = A - mean_vector

    return A


def SVD(A):
    """
    SVD decomposition
    """
    # get eigen values and eigen vectors from matrix
    eig_vals, eig_vecs  = np.linalg.eig(A.dot(A.T))
    # for ev in eig_vecs:
    #      np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    # print('ok!')
    # print('Eigenvalues in descending order:')
    sorted_eigen_vals = sorted(eig_vals, reverse=True)
    # for eig_val in sorted_eigen_vals:
    #     print(eig_val)
    # sigma is sqrt of eigen values
    S = np.diag(np.sqrt(sorted_eigen_vals))
    S = np.nan_to_num(S)
    return eig_vecs, S


def cumulative_var(S, row):
    """
    calculate cumulative sum of explained variances
    """
    # compute total of the diagonal matrix: sigma
    sTot = np.nansum(S)
    # the cumulative variance
    var_i = np.array([(np.sum(S[: i + 1]) / sTot) * 100.0 for i in range(row)])
    # k: number of minimum features that retains the given variance
    k = len(var_i[var_i < (varRetained * 100)])
    print('%.2f %% variance retained in %d dimensions' % (var_i[k], k))
    return k, var_i


def cumulative_variance_plot(var_i, k):
    """
    plot cumulative explained variance
    """
    # plt.bar(range(1, k+1), var_i[:k], alpha=0.5, align='center', label='individual explained variance')
    plt.bar(range(1, k + 1), var_i[:k], alpha=0.5, align='center', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.show()


def score_matrix_plot(U, S):
    """
    plot the score matrix
    """
    Score_matrix = U.dot(S)
    plt.scatter(Score_matrix[:, 0], Score_matrix[:, 1])
    plt.show()


def calc_loading_matrix(X, U, S, k):
    """
    calculate VT matrix
    """
    U_reduced = U[:, : k]
    S_reduced = S[: k, : k]
    S_inv = np.linalg.inv(S_reduced)
    VT = S_inv.dot(U_reduced.T).dot(X)
    loading_matrix = VT
    return loading_matrix


def loading_matrix_plot(X, U, S, k):
    # compute loading matrix
    loading_matrix = calc_loading_matrix(X, U, S, k)
    # scatter plot score matrix
    plt.scatter(loading_matrix[:, 0], loading_matrix[:, 1])
    plt.show()


def transformed_matrix_plot(U, k):
    matrix_w = U[:, : k]
    transformed = matrix_w.T.dot(X)
    plt.plot(transformed[0, 0:k], transformed[1, 0:k], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
    plt.plot(transformed[0, 20:40], transformed[1, 20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show()

# read dataset
X, data = read_data()
# matrix dimensions
(row, col) = X.shape
genes = data[0].values[0:col]
lables = data.iloc[col][1:].to_numpy()
# subtracting the mean from Matrix
X = mean_centered(X)
# SVD decomposition
U, S = SVD(X)
# compute cumulative variance
k, cum_var = cumulative_var(S, row)
# plot cumulative variance
cumulative_variance_plot(cum_var, k)
# plot the score matrix
score_matrix_plot(U, S)
# plot the loading matrix
loading_matrix_plot(X, U, S, k)
# transformed matrix
transformed_matrix_plot(U, k)
