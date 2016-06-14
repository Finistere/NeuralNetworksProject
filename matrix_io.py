import numpy as np
import csv
from scipy.sparse import csr_matrix
from scipy.io import loadmat


def matlab_matrix(path, **kwargs):
    return loadmat(path, **kwargs)


def regular_matrix(path, **kwargs):
    return np.loadtxt(path, **kwargs)


def sparse_matrix(path, nb_features, delimiter=' ', index_value_delimiter=':'):
    indptr = [0]
    indices = []
    data = []
    shape = [0, nb_features]

    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=delimiter)

        for row in reader:
            indptr.append(indptr[-1] + len(row))
            shape[0] += 1
            for i in range(len(row)):
                try:
                    index, value = row[i].split(index_value_delimiter)
                    indices.append(int(index))
                    data.append(float(value))
                except ValueError:
                    indptr[-1] -= 1

    return csr_matrix((np.array(data), indices, indptr), shape).T.toarray()


def sparse_binary_matrix(path, nb_features, delimiter=' '):
    indptr = [0]
    indices = []
    shape = [0, nb_features]

    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=delimiter)

        for row in reader:
            indptr.append(indptr[-1] + len(row))
            shape[0] += 1
            for i in range(len(row)):
                if row[i]:
                    indices.append(int(row[i]))
                else:
                    indptr[-1] -= 1

    return csr_matrix((np.ones(len(indices)), indices, indptr), shape).T.toarray()
