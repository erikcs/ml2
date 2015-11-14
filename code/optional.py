import numpy as np
import GPy
import matplotlib.pyplot as plt

sigma = .1
N = 100

latent = np.linspace(1, 4*np.pi, N) # True latent variable
A = np.random.normal(0, 1, (10, 2))
fnonlin = np.column_stack((latent*np.sin(latent), latent*np.cos(latent)))
Y = np.dot(A, fnonlin.transpose()) + np.random.normal(0, sigma, (10, N))
Y = Y.transpose()

Q = 1
m_gplvm = GPy.models.GPLVM(Y, Q, kernel=GPy.kern.Exponential(Q))
m_gplvm.kern.lengthscale = .2
m_gplvm.kern.variance = 1
m_gplvm.likelihood.variance = 1.
m_gplvm.optimize(messages=0, max_iters=5e4)

plt.suptitle('GPy latent variable model (exp. kernel)', fontsize=15)
plt.subplot(1, 2, 1)
plt.plot(m_gplvm.X)
plt.xlabel('observation')
plt.ylabel('latent variable 1')

Q = 2
m_gplvm = GPy.models.GPLVM(Y, Q, kernel=GPy.kern.Exponential(Q))
m_gplvm.kern.lengthscale = .2
m_gplvm.kern.variance = 1
m_gplvm.likelihood.variance = 1.
m_gplvm.optimize(messages=0, max_iters=5e4)

plt.subplot(1, 2, 2)
plt.plot(m_gplvm.X[:,0], m_gplvm.X[:,1])
plt.xlabel('latent variable 1')
plt.ylabel('latent variable 2')
