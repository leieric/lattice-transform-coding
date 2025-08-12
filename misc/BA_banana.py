"""
Runs Blahut-Arimoto on the Banana source.
"""

import numpy as np
import LTC.data as data
import ot
import tqdm
import scipy.stats

def est_ent(X):
    H, xedges, yedges = np.histogram2d(X[:,0].numpy(), X[:,1].numpy(), bins=50, density=True)
    p_x = H.flatten()
    idx_keep = np.where(p_x > 0)[0]
    p_x = p_x[idx_keep]
    ent = scipy.stats.entropy(p_x, base=2)
    return ent


def run_BA_full_2d(beta, X):
    """
    Runs Blahut-Arimoto on data X, of shape [n, 2].

    beta: rate-distortion tradeoff lagrangian
    X: numpy array of data samples
    """
    # generate 2D empirical distribution from data X
    H, xedges, yedges = np.histogram2d(X[:,0].numpy(), X[:,1].numpy(), bins=300, density=True)
    xlocs = (xedges[:-1] + xedges[1:])/2
    ylocs = (yedges[:-1] + yedges[1:])/2
    p_x = H.flatten()
    idx_keep = np.where(p_x > 0)[0]
    p_x = p_x[idx_keep]
    x1, y1 = np.meshgrid(xlocs, ylocs)
    gridpos = np.stack((x1.flatten(), y1.flatten()), axis=1)[idx_keep]
    M = ot.dist(gridpos)
    
    # run BA
    print(p_x.shape, M.shape)
    R,D = run_BlahutArimoto(M, p_x, beta, max_it=300000, eps=1e-6)
    return R, D

def run_BA_full_1d(beta, X):
    """
    Runs Blahut-Arimoto on 1-d data X.

    beta: rate-distortion tradeoff lagrangian
    X: numpy array of data samples
    """
    H, xedges = np.histogram(X.numpy(), bins=200, density=True)
    xlocs = (xedges[:-1] + xedges[1:])/2
    p_x = H.flatten()
    idx_keep = np.where(p_x > 0)[0]
    p_x = p_x[idx_keep]
    # x1, y1 = np.meshgrid(xlocs, ylocs)
    # gridpos = np.stack((x1.flatten(), y1.flatten()), axis=1)[idx_keep]
    gridpos = np.expand_dims(xlocs[idx_keep], 1)
    M = ot.dist(gridpos)
    
    print(p_x.shape, M.shape)
    R,D = run_BlahutArimoto(M, p_x, beta, max_it=300000, eps=1e-8)
    # R, D = run_BlahutArimoto(x_pos.numpy(), y_pos.numpy(), p_x.numpy(), beta, eps=1e-10)
    return R, D

def run_BA_empirical(beta, X):
    X = X.numpy()
    M = ot.dist(X)
    n = X.shape[0]
    p_x = (1/n)*np.ones(n)
    print(p_x.shape, M.shape)
    R,D = run_BlahutArimoto(M, p_x, beta, max_it=1000, eps=1e-7)
    return R, D

if __name__ == "__main__":
    X = data.Banana(1000, n_samples=1000000).dataset.tensors[0] # shape is: [1000000, 2]
    # X = data.Uniform(1, 1000, n_samples=2000).dataset.tensors[0] # shape is: [2000, 1]
    # X = data.Banana1d(1, 1000, n_samples=500000).dataset.tensors[0] # shape is: [500000, 1]
    rates = []
    dists = []
    # lams = [0.05, 0.2, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]
    # lams = [24, 32, 48, 64, 96, 128, 256]
    lams = [0.05, 0.2, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256]
    # lams = [0.01, 0.05]
    for lam in tqdm.tqdm(lams):
        R,D = run_BA_full_2d(lam, X)
        print(f"R={R:.4f}, D={D:.4f}")
        rates.append(R)
        dists.append(D)
    print(f"rates={rates}")
    print(f"dists={dists}")
    
def run_BlahutArimoto(dist_mat, p_x, beta ,max_it = 500,eps = 1e-10) :
    """Compute the rate-distortion function of an i.i.d distribution
    Original author: Alon Kipnis
    Inputs :
        'dist_mat' -- (numpy matrix) representing the distoriton matrix between the input 
            alphabet and the reconstruction alphabet. dist_mat[i,j] = dist(x[i],x_hat[j])
        'p_x' -- (1D numpy array) representing the probability mass function of the source
        'beta' -- (scalar) the slope of the rate-distoriton function at the point where evaluation is 
                    required
        'max_it' -- (int) maximal number of iterations
        'eps' -- (float) accuracy required by the algorithm: the algorithm stops if there
                is no change in distoriton value of more than 'eps' between consequtive iterations
    Returns :
        'Iu' -- rate (in bits)
        'Du' -- distortion
    """
    import numpy as np

    l,l_hat = dist_mat.shape
    p_cond = np.tile(p_x, (l_hat,1)).T #start with iid conditional distribution

    p_x = p_x / np.sum(p_x) #normalize
    p_cond /= np.sum(p_cond,1,keepdims=True)

    it = 0
    Du_prev = 0
    Du = 2*eps
    while it < max_it and np.abs(Du-Du_prev)> eps :
        
        it+=1
        Du_prev = Du
        p_hat = np.matmul(p_x,p_cond)

        p_cond = np.exp(-beta * dist_mat) * p_hat
        p_cond /= np.expand_dims(np.sum(p_cond,1),1)
        
        zeros = np.where(p_cond == 0)
        term = p_cond*np.log(p_cond / np.expand_dims(p_hat,0))
        term[zeros] = 0
        
        Iu = np.matmul(p_x, term).sum() / np.log(2)
        Du = np.matmul(p_x,(p_cond * dist_mat)).sum()
#         print(it, Iu, Du)
    # print(p_cond, p_hat)
    return Iu, Du