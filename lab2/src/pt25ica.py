""" Part 2.5 - ICA """
import numpy as np
from numpy import dot
import matplotlib.pyplot as plt
from matplotlib import gridspec

X = np.loadtxt('data/DataICA.txt', skiprows=2)
X = X.T

def whiten(X):
    """ Demean and whiten
    X shape: (N: vector dimension, M: nr of observed mixed vectors)
    Returns: Xt (NxM) whitened matrix, D, E """
    X = X - X.mean(axis=1)[:,None]
    xx = np.cov(X, bias=1)
    D, E = np.linalg.eig(xx) 
    D = np.diag(D)

    Xt = np.dot(
        np.dot(E, np.diag(np.power(np.diag(D), -0.5))),
        np.dot(E.T, X)
    )

    return Xt, D, E

def plotwhitened():
    fig, axarr = plt.subplots(2, 2, figsize=(10,9))
    axarr[0, 0].plot(X[0,:], X[1,:], '.')
    axarr[0, 0].set_title('original')
    axarr[0, 0].set_xlabel('$x_1$')
    axarr[0, 0].set_ylabel('$x_2$')
    axarr[0, 1].plot(Xt[0,:], Xt[1,:], '.')
    axarr[0, 1].set_title('whitened')
    axarr[0, 1].set_xlabel('$x_1$')
    axarr[0, 1].set_ylabel('$x_2$')
    axarr[1, 0].plot((1, 2), (D[0,0], D[1,1]), 'o')
    axarr[1, 0].set_xticks((1, 2))
    axarr[1, 0].set_xticklabels(['$\lambda_1$', '$\lambda_2$']) 
    axarr[1, 0].margins(0.2)
    axarr[1, 0].set_title('eigenvalues')
    axarr[1, 1].plot([0, E[0,0]], [0, E[1,0]])
    axarr[1, 1].plot([0, E[0,1]], [0, E[1,1]])
    axarr[1, 1].legend(['$v_1$', '$v_2$' ])
    axarr[1, 1].set_title('eigenvectors')
    plt.show()

def fastica(Xt):
    # FastICA
    # from the wikipage
    # Returns the unmixing matrix W
    np.random.seed(123)
    a1 = 1
    N, M = Xt.shape
    C = 2

    def G1(u):
        return 1.0 / a1 * np.log(np.cosh(a1*u))
    def dG1(u):
        return np.tanh(a1 * u)
    def G2(u):
        return -np.exp(-u**2/2.0)
    def dG2(u):
        return -u * G2(u)
    g = G1
    dg = dG1

    W = np.zeros((C, N))
    TOL = 1e-5
    MAXITER = 1000

    for p in range(C):
        wp = np.random.rand(N, 1)
        wpp = wp.copy()
        for _ in range(MAXITER):
            wpp = (1.0/M) * (dot(Xt, g(dot(wp.T, Xt)).T)
                             - dot(dg(dot(wp.T, Xt)), np.ones((M, 1))) * wp)
            if p > 0:
                wpp = wpp - dot(
                    dot(W[:,0][:,None], wpp.T),
                    W[:,0][:,None]
                )

            wpp = wpp / np.linalg.norm(wpp)    
            if np.linalg.norm(wpp - wp) < TOL:
                W[:,p][:,None] = wpp
                break
            assert _ != MAXITER-1
            wp = wpp.copy()
    return W.T

def plotfica():
    fig = plt.figure(figsize=(8, 14))
    gs = gridspec.GridSpec(5, 3)

    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(S[0,:], '.-')
    ax1.set_ylabel('s$_1$')
    ax1.set_title('Reconstructed signals')
    ax2 = fig.add_subplot(gs[1,:])
    ax2.plot(S[1,:], '.-')
    ax2.set_ylabel('s$_2$')
    ax3 = fig.add_subplot(gs[2:5,0:2])
    ax3.plot(S[0,:], S[1,:], '.')
    ax3.set_ylabel('s$_2$')
    ax3.set_xlabel('s$_1$')
    ax3.set_title('Reconstructed unmixed distribution')
    ax4 = fig.add_subplot(gs[3:4,2])
    ax4.plot([0, W.T[0,0]], [0, W.T[0,1]])
    ax4.plot([0, W.T[1,0]], [0, W.T[1,1]])
    ax4.legend(['$w\'_1$', '$w\'_2$' ], loc=4)
    ax4.set_title('Estimated mixing \n vectors')

Xt, D, E = whiten(X)
W = fastica(Xt)
S = np.dot(W, Xt)
