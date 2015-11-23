""" Question 12 """
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def run():
    sigmak = 1.0
    N = 100
    mu = np.zeros(N)
    X = np.column_stack(
        (np.linspace(-1, 1, N), np.zeros(N))
    )

    List = [0.1, 0.5, 1.0, 10.0]
    pidx = 1
    for L in List:
        K = sigmak**2 * np.exp(-1/L**2 * cdist(X, X, 'sqeuclidean'))

        plt.subplot(len(List), 2, pidx)
        plt.title('l: {}'.format(L))
        pidx += 1
        for _ in range(4):
            fhat = np.random.multivariate_normal(mu, K)
            plt.plot(X[:,0], fhat)
