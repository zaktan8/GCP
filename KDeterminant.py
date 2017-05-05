from math import ceil, sqrt
from sklearn import cluster


class KDeterminant:
    def __init__(self):
        self.cash_values = {}

    def _f(self, edge_matrix, k):
        cashed_k = self.cash_values.get(k)
        if cashed_k:
            fs = cashed_k[0]
            s_k = cashed_k[1]
        else:
            n_d = len(edge_matrix)
            kmeans = cluster.KMeans(n_clusters=k).fit(edge_matrix)
            s_k = kmeans.inertia_

            def a(k, n_d):
                if k == 2:
                    return 1 - 3 / (4 * n_d)
                else:
                    prev_a = a(k - 1, n_d)
                    return prev_a + (1 - prev_a) / 6

            fs = 1 if (k == 1) else s_k / (a(k, n_d) * self._f(edge_matrix, k - 1)[1])
            self.cash_values[k] = (fs, s_k)
        return fs, s_k

    def get_best_k(self, edge_matrix):
        results = {}
        for k in range(1, ceil(sqrt(len(edge_matrix[0]))) + 1):
            f_k, s_k = self._f(edge_matrix, k)
            # print("at k = {}:\tf_k = {:.4f},\ts_k = {:.4f}".format(k, f_k, s_k))
            results[k] = f_k
        return min(results, key=results.get)
