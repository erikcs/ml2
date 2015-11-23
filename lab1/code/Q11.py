""" Question 11 """
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def run(tau):
    N = 201
    w_prf = [-1.3, 0.5]
    sigma = np.sqrt(0.3)

    X = np.row_stack((np.linspace(-1, 1, N), np.ones(N)))
    epsilon = np.random.normal(0, sigma, N)
    y_prf = np.sum(X.transpose() * w_prf, axis=1) 
    y = y_prf + epsilon

    Xx, Yy = np.meshgrid(np.arange(-3.0, 3.0, 0.02), np.arange(-3.0, 3.0, 0.01))
    mu_prior = [0, 0]
    sigma_prior = np.diag((1, 1)) * tau**2

    nObslist = [0, 1, 10, 50, 201]
    pidx = 1
    for nObs in nObslist:
        X_obs = X[:,0:nObs]
        y_obs = y[0:nObs]

        ##
        sigma_posterior = np.linalg.inv(
            (1/sigma**2) * np.dot(X_obs, X_obs.transpose()) + (1/tau**2) * np.diag((1,1))
            )
        mu_posterior = np.dot(sigma_posterior,
                              (1/sigma**2) * np.dot(X_obs, y_obs))

        ## Plot it
        vcm = sigma_posterior
        mu = mu_posterior
        Z = mlab.bivariate_normal(Xx, Yy,
                                  np.sqrt(vcm[0,0]), np.sqrt(vcm[1,1]),
                                  mu[0], mu[1], vcm[0,1])

        plt.subplot(len(nObslist), 2, pidx)
        pidx += 1
        plt.contour(Xx,Yy,Z)
        plt.xlabel('w0')
        plt.ylabel('w1')

        plt.subplot(len(nObslist), 2, pidx)
        pidx += 1
        for _ in range(4):
            coeffs = np.random.multivariate_normal(mu_posterior, sigma_posterior)
            line = np.sum(X.transpose() * coeffs, axis=1)
            plt.plot(X[0], line)
            plt.plot(X_obs[0], y_obs, '+')
            plt.title('observations: {}'.format(nObs))
