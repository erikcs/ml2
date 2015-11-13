# %%time
# %%cython
#%load_ext Cython
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.utils.extmath import cartesian
from code import index

# The data, one row per obs
X = np.array([[-1, 1], [0, 1], [1, 1],
             [-1, 0], [0, 0], [1, 0],
             [-1, -1], [0, -1], [1, -1]])

# One row is one dataset
D = cartesian(
    np.column_stack((np.ones(9), -1*np.ones(9)))
    )

def viz(D, idx):
    """
    Red: 1
    Blue: -1
    """
    cmap = colors.ListedColormap(['blue', 'red'])
    bounds = [-1, 1, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(D[idx].reshape(3, 3), interpolation='nearest', cmap=cmap, norm=norm)
    plt.axis('off')
    
def probD(X, Y, model, *args):
    """
    Compute the probability of the entire dataset
    X: x array
    Y: Data array
    model: string 'M0', etc
    *args: theta_i
    Return a 512, Ndarray with the probability of each dataset.
    Sanity check: all these sum to 1
    """
    if model == 'M0':
        arr = np.zeros(512)
        arr.fill(1.0/512.0)
        return arr
    
    elif model == 'M1':
        assert len(args) == 1
        t1 = args
        fcn =  1.0 / (1.0 + np.exp(-Y *t1 * X[:,0]))
        return fcn.prod(axis=1)
    
    elif model == 'M2':
        assert len(args) == 2
        t1, t2 = args
        fcn =  1.0 / (1.0 + np.exp(-Y *(t1 * X[:,0] + t2 * X[:,1])))
        return fcn.prod(axis=1)  

    elif model == 'M3':
        assert len(args) == 3
        t1, t2, t3 = args
        fcn =  1.0 / (1.0 + np.exp(-Y *(t1 * X[:,0] + t2 * X[:,1] + t3)))
        return fcn.prod(axis=1)
        
        
def runMC(S, X, D):
    """
    Run once and save results
    Read with np.load('code/myfile.npy')
    """

    res = np.zeros((512, 5), dtype=np.float64) ## The last column is the index column
    res[:,0] = probD(X, D, 'M0')
    
    ## Prior 0 ##
    sigma = np.sqrt(10**3)    
    m = np.zeros(3)
    sig = sigma**2 * np.eye(3)
    
    for s in range(S):
        theta  = np.random.multivariate_normal(m, sig)
        
        res[:,1] += probD(X, D, 'M1', theta[0])
        res[:,2] += probD(X, D, 'M2', theta[0], theta[1])
        res[:,3] += probD(X, D, 'M3', theta[0], theta[1], theta[2]) 
        
    res[:,1:4] /= S
    ix = index.create_index_set(res[:,0:4].transpose())
    res[:,-1] = ix
    np.save('code/res_prior00', res)
    res[:,1:5].fill(0)
    
    ## Prior 1 - nondiag VCM ##
    sigma = np.sqrt(10**3)    
    m = np.zeros(3)
    t = np.random.rand(3, 3)
    sig = sigma**2 * np.dot(t, t.transpose())
    
    for s in range(S):
        theta  = np.random.multivariate_normal(m, sig)
        
        res[:,1] += probD(X, D, 'M1', theta[0])
        res[:,2] += probD(X, D, 'M2', theta[0], theta[1])
        res[:,3] += probD(X, D, 'M3', theta[0], theta[1], theta[2]) 
        
    res[:,1:4] /= S
    ix = index.create_index_set(res[:,0:4].transpose())
    res[:,-1] = ix
    np.save('code/res_prior11', res)
    res[:,1:5].fill(0)
    
    ## Prior 2 - nonzero mean ##
    sigma = np.sqrt(10**3)    
    m = 5 * np.ones(3)
    sig = sigma**2 * np.eye(3)
    
    for s in range(S):
        theta  = np.random.multivariate_normal(m, sig)
        
        res[:,1] += probD(X, D, 'M1', theta[0])
        res[:,2] += probD(X, D, 'M2', theta[0], theta[1])
        res[:,3] += probD(X, D, 'M3', theta[0], theta[1], theta[2]) 
        
    res[:,1:4] /= S
    ix = index.create_index_set(res[:,0:4].transpose())
    res[:,-1] = ix
    np.save('code/res_prior22', res)
#runMC(10**6, X, D) (fname 0, 1, etc)
# MC run two
#runMC(10**6, X, D) (00, 11, etc)
# MC run three
#runMC(10**7, X, D) (000, etc)
   
""" Plot the stuff"""
def showmain(dset=1):
    if dset == 1:
        fnames = ['code/res_prior0.npy', 'code/res_prior1.npy', 'code/res_prior2.npy']
    elif dset == 3:
        fnames = ['code/res_prior000.npy', 'code/res_prior111.npy', 'code/res_prior222.npy']
    data = []
    idx = []

    for fname in fnames:
        arr = np.load(fname)
        idx.append(arr[:, -1].astype(int))
        data.append(arr[:,0:4])

    for id in range(3):
        arr = data[id]
        index = idx[id]

        plt.figure(id)
        plt.suptitle('Evidence (Prior {})'.format(id) ,fontsize=15)
        plt.subplot(1, 2, 1)
        plt.plot(arr[index]);
        plt.legend(['M0', 'M1', 'M2', 'M4'])
        plt.xlabel('Data sets')
        plt.ylabel('Evidence')
        plt.axis('tight');

        plt.subplot(1, 2, 2)
        plt.plot(arr[index][0:90]);
        plt.legend(['M0', 'M1', 'M2', 'M4'])
        plt.xlabel('Data sets (subset)')
        plt.ylabel('Evidence')
        plt.axis('tight');

    pidx = 1
    plt.figure(4)
    plt.suptitle('Most/least probable data set', fontsize=15)
    for id in range(3):
        NROWS = 3
        NCOLS = 6

        arr = data[id]
        index = idx[id]

        plt.subplot(NROWS, NCOLS, pidx)
        pidx += 1
        plt.title('Prior {}: M1 max'.format(id))

        viz(D, arr[:,1].argmax())
        plt.subplot(NROWS, NCOLS, pidx)
        pidx += 1
        plt.title('M1 min')
        viz(D, arr[:,1].argmin())

        plt.subplot(NROWS, NCOLS, pidx)
        pidx += 1
        plt.title('M2 max')
        viz(D, arr[:,2].argmax())
        plt.subplot(NROWS, NCOLS, pidx)
        pidx += 1
        plt.title('M2 min')
        viz(D, arr[:,2].argmin())

        plt.subplot(NROWS, NCOLS, pidx)
        pidx += 1
        plt.title('M3 max')
        viz(D, arr[:,3].argmax())
        plt.subplot(NROWS, NCOLS, pidx)
        pidx += 1
        plt.title('M3 min')
        viz(D, arr[:,3].argmin())

