""" Question 21 """
# TODO: restructure this to make it cleaner
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.optimize import fmin_cg
from numpy import log, dot, eye, trace
from numpy.linalg import inv, det
np.random.seed(123)

def run():
    ## Our data
    sigma = .1
    N = 100

    latent = np.linspace(1, 4*np.pi, N) # True latent variable
    A = np.random.normal(0, 1, (10, 2))
    fnonlin = np.column_stack((latent*np.sin(latent), latent*np.cos(latent)))
    Y = np.dot(A, fnonlin.transpose()) + np.random.normal(0, sigma, (10, N))

    # Center Y
    Y = Y - Y.mean(axis=1)[:, None]

    ## The cov matrix we need for our goal function
    S = np.cov(Y, bias=1) 

    # fmin_cg expects a vector, not a matrix..., and it HAS to be a 1-D arr
    def loglik(W):
        W = W.reshape(10, 2)
        C = dot(W, W.transpose()) + sigma**2 * eye(10)
        return N * ( log(det(C)) + trace(dot(inv(C), S)) )

    def dloglik(W):
        W = W.reshape(10, 2)
        C = dot(W, W.transpose()) + sigma**2 * eye(10)

        t1 = dot(inv(C), S)
        t2 = dot(inv(C), W)
        left = dot(t1, t2)
        right = dot(inv(C), W)
        grad =  N * (-left + right) # Sanity check: check if dloglik(W_star) ~= 0, i.e. I correctly specified
                                    # the gradients and we are at a stationary point...
        return grad.reshape(20)

    # No noise   
    Winit = np.random.normal(0, 1, 20)
    W_star = fmin_cg(loglik, Winit, fprime=dloglik, disp=0)
    W_star = W_star.reshape(10, 2)
    ## Recover our latent factors based on the estimated W
    X_hat = np.zeros((N, 2))
    for n in range(N):
        X_hat[n]= dot(
            inv(dot(W_star.transpose(), W_star)), dot(W_star.transpose(), Y[:,n]))

    X_hat1 = np.copy(X_hat)

    ## Run some experiments ##

    # Some noise
    sigma = 1

    Y = np.dot(A, fnonlin.transpose()) + np.random.normal(0, sigma, (10, N))
    Y = Y - Y.mean(axis=1)[:, None]
    S = np.cov(Y, bias=1) 

    W_star = fmin_cg(loglik, Winit, fprime=dloglik, disp=0)
    W_star = W_star.reshape(10, 2)
    X_hat = np.zeros((N, 2))
    for n in range(N):
        X_hat[n]= dot(
            inv(dot(W_star.transpose(), W_star)), dot(W_star.transpose(), Y[:,n]))

    X_hatNoise = np.copy(X_hat)

    # Plenty of noise
    sigma = 10

    Y = np.dot(A, fnonlin.transpose()) + np.random.normal(0, sigma, (10, N))
    Y = Y - Y.mean(axis=1)[:, None]
    S = np.cov(Y, bias=1) 

    W_star = fmin_cg(loglik, Winit, fprime=dloglik, disp=0)
    W_star = W_star.reshape(10, 2)
    X_hat = np.zeros((N, 2))
    for n in range(N):
        X_hat[n]= dot(
            inv(dot(W_star.transpose(), W_star)), dot(W_star.transpose(), Y[:,n]))

    X_hatNoiseP = np.copy(X_hat)

    ## Plot it
    plt.subplot(2, 2, 1)
    plt.title('The true lower dimensional representation')
    plt.plot(latent, label='True latent variable')
    plt.plot(fnonlin[:,0], fnonlin[:,1], label='Non linear transform')
    plt.legend()
    plt.xlabel('$Xi$')

    plt.subplot(2, 2, 2)
    plt.title('Recovered latent variables, no noise')
    plt.plot(X_hat1[:,0], X_hat1[:,1])
    plt.xlabel('$X1$')
    plt.ylabel('$X2$')

    plt.subplot(2, 2, 3)
    plt.title('sigma=1')
    plt.plot(X_hatNoise[:,0], X_hatNoise[:,1])
    plt.xlabel('$X1$')
    plt.ylabel('$X2$')

    plt.subplot(2, 2, 4)
    plt.title('sigma=10')
    plt.plot(X_hatNoiseP[:,0], X_hatNoiseP[:,1])
    plt.xlabel('$X1$')
    plt.ylabel('$X2$')
