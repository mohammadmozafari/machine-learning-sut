import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, input_x):
        dist_matrix = self.__compute_distance_matrix(input_x)
        preds = np.zeros(dist_matrix.shape[0])
        for i in range(dist_matrix.shape[0]):
            knn_indices = np.argsort(dist_matrix[i, :])[:self.k]
            counter = Counter(self.Y[knn_indices])
            pred = max(counter, key=counter.get)
            preds[i] = pred
        return preds

    def __compute_distance_matrix(self, input_x):
        dist_matrix = np.zeros((input_x.shape[0], self.X.shape[0]))
        dist_matrix = ((dist_matrix.T) + np.sum(input_x ** 2, axis=1)).T
        dist_matrix = dist_matrix + np.sum(self.X ** 2, axis=1)
        dist_matrix = dist_matrix - 2 * (input_x @ self.X.T)
        return dist_matrix


