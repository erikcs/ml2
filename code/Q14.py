""" Question 14 """
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def run():
    sigmak = 1.0
    sigmaeps = np.sqrt(0.5)
    N = 7
    X = np.column_stack(
        (np.linspace(-np.pi, np.pi, N), np.zeros(N)))
    y_prf = np.sin(X[:,0])  # Could perhaps use the notation f instead of y_prf, now my notation
                            # is getting a bit sloppy
    epsilon = np.random.normal(0, sigmaeps, N)
    y = y_prf + epsilon

    # Predictive data
    Nobs = 100
    Xobs = np.column_stack(
        (np.linspace(-3*np.pi, 3*np.pi, Nobs), np.zeros(Nobs)))
    y_obs = np.sin(Xobs[:,0]) # The true latent process

    def kernel(x, y, L=1.0, pred=0):
        K = sigmak**2 * np.exp(-1/L**2 * cdist(x, y, 'sqeuclidean'))
        if pred:
            K += sigmaeps**2 * np.diag(np.diag(np.ones(K.shape)))
        return K

    def posterior(f, x, xobs, L=1.0, pred=0):
        Kstar = kernel(x, xobs, L)
        K = kernel(x, x, L, pred)
        K2star = kernel(xobs, xobs, L)

        comfac = np.dot(Kstar.transpose(), 
                        np.linalg.inv(K))

        mup = np.dot(comfac, f)    
        sigmap = K2star - np.dot(comfac, Kstar)

        return mup, sigmap

    ## Interpolation
    pred = 0
    f = y_prf
    x = X
    xobs = Xobs

    List = [0.1, 0.5, 1.0, 5.0]
    pidx = 1
    plt.figure(1)
    plt.suptitle('Interpolation')
    for L in List:
        mup, sigmap = posterior(f, x, xobs, L, pred)
        """ The VCM could in some cases be borderline degenerate, I just assume this is because of numerical imprecision 
        cumuluting from matrix inversions, etc etc, b/c everything should be implemented correctly """
        
        plt.subplot(len(List), 2, pidx)
        plt.title('l: {}'.format(L))
        pidx += 1
        plt.plot(X[:,0], y, '+', markersize=10)
        plt.plot(X[:,0], y_prf, 'ko', markersize=7)
        plt.plot(Xobs[:,0],y_obs, 'k', linewidth=2)
        plt.plot(Xobs[:,0], mup, '--')
        for _ in range(3):
            fhat = np.random.multivariate_normal(mup, sigmap)
            plt.plot(Xobs[:,0], fhat)
        plt.subplot(len(List), 2, pidx)
        pidx += 1
        plt.imshow(sigmap)
        plt.colorbar()

    ## Prediction
    pred = 1
    f = y
    x = X
    xobs = Xobs

    pidx = 1
    plt.figure(2)
    plt.suptitle('Prediction')
    for L in List:
        mup, sigmap = posterior(f, x, xobs, L, pred)


        plt.subplot(len(List), 2, pidx)
        plt.title('l: {}'.format(L))
        pidx += 1
        plt.plot(X[:,0], y, '+', markersize=10)
        plt.plot(X[:,0], y_prf, 'ko', markersize=7)
        plt.plot(Xobs[:,0],y_obs, 'k', linewidth=2)
        plt.plot(Xobs[:,0], mup, '--')
        for _ in range(3):
            fhat = np.random.multivariate_normal(mup, sigmap)
            plt.plot(Xobs[:,0], fhat)
        
        plt.subplot(len(List), 2, pidx)
        pidx += 1
        plt.imshow(sigmap)
        plt.colorbar()
