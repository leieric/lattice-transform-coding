"""
Implements algorithm from:
E. Agrell and T. Eriksson, "Optimization of lattices for quantization," in IEEE Transactions on Information Theory, vol. 44, no. 5, pp. 1814-1828, Sept. 1998, doi: 10.1109/18.705561.
"""

import torch
import numpy as np
import tqdm
from argparse import ArgumentParser
from joblib import Parallel, delayed
import os

def gram_schmidt(vv):
	def projection(u, v):
		return (v * u).sum() / (u * u).sum() * u

	nk = vv.size(0)
	uu = torch.zeros_like(vv, device=vv.device)
	# uu[:, 0] = vv[:, 0].clone()
	uu[0,:] = vv[0,:].clone()
	for k in range(1, nk):
		vk = vv[k].clone()
		uk = 0
		for j in range(0, k):
			# uj = uu[:, j].clone()
			uj = uu[j,:].clone()
			uk = uk + projection(uj, vk)
		# uu[:, k] = vk - uk
		uu[k, :] = vk - uk
	for k in range(nk):
		# uk = uu[:, k].clone()
		uk = uu[k, :].clone()
		# uu[:, k] = uk / uk.norm()
		uu[k, :] = uk / uk.norm()
	return uu


def LLL(G, delta=0.75):
    """
    Lenstra, A.K., Lenstra, H.W. & Lovász, L. Factoring polynomials with rational coefficients. Math. Ann. 261, 515-534 (1982).
    """
    G = torch.clone(G)
    n = G.shape[0]
    O = gram_schmidt(G)
	# print(O)
    mu = lambda i, j: (O[j, :]*G[i, :]).sum() / (O[j] * O[j]).sum()
    # mu = lambda i, j: (O[:, j]*G[:, i]).sum() / (O[:, j] * O[:, j]).sum()
    k = 1
    while k < n:
        for j in range(k-1, -1, -1):
            mu_kj = mu(k, j)
            if abs(mu_kj) > 0.5:
                G[k] = G[k] - G[j] * torch.round(mu_kj)				
                O = gram_schmidt(G)
        # if O[k].dot(O[k]) >= (delta - mu(k, k-1)**2) * O[k-1].dot(O[k-1]):
        # 	k += 1
        if (O[k]*O[k]).sum() >= (delta - mu(k, k-1)**2) * (O[k-1]*O[k-1]).sum():
            k += 1
        else:
            idx = torch.arange(n)
            idx[k] = k-1
            idx[k-1] = k
            G = G.index_select(0, idx)
            # G[k], G[k - 1] = G[k - 1], G[k]
            O = gram_schmidt(G)
            k = max(k - 1, 1)
    return G

def sgn(x):
        return 2*(x > 0).float() - 1


def quantize_schnorr_euchner(H, x):
    """
    The Schnorr-Euchner sphere decoder with Pohst strategy. Implements the decode function in 

    E. Agrell, T. Eriksson, A. Vardy and K. Zeger, "Closest point search in lattices," in IEEE Transactions on Information Theory, vol. 48, no. 8, pp. 2201-2214, Aug. 2002, doi: 10.1109/TIT.2002.800499.
    """

    # lattice dimensionality
    dim = H.shape[0]

    # squared distance of current closest point
    best_dist = np.inf

    # dimension currently considered
    k = dim - 1

    dist = torch.zeros(dim, device=x.device, dtype=x.dtype)

    # transform x
    E = torch.zeros([dim, dim], device=x.device, dtype=x.dtype)
    E[k] = x @ H #np.dot(x, H)

    # lattice point (integer coordinates)
    u = torch.zeros(dim, device=x.device, dtype=x.dtype)
    u[k] = torch.round(E[k, k])

    # distance of x to sublattice indicated by u[k]
    y = (E[k, k] - u[k]) / H[k, k]

    # indicates which sublattice to consider next
    step = torch.zeros(dim, device=x.device, dtype=x.dtype)
    step[k] = sgn(y)

    while True:
        # lower bound on distance of all points in current sublattice
        new_dist = dist[k] + y ** 2

        if new_dist < best_dist:
            if k > 0:
                E[k - 1, :k] = E[k, :k] - y * H[k, :k]

                # move down
                k -= 1
                dist[k] = new_dist
                u[k] = torch.round(E[k, k])
                y = (E[k, k] - u[k]) / H[k, k]
                step[k] = sgn(y)
            else:
                # found closer point
                u_best = torch.clone(u) # u.copy()
                best_dist = new_dist

                # move up
                k += 1
                u[k] += step[k]
                y = (E[k, k] - u[k]) / H[k, k]

                # change direction
                step[k] = -step[k] - sgn(step[k])
        else:
        # no point in sublattice is better than current best point 
            # return u_best
            if k == dim - 1:
                return u_best
            else:
                # move up
                k += 1
                u[k] += step[k]
                y = (E[k, k] - u[k]) / H[k, k]

                # change direction
                step[k] = -step[k] - sgn(step[k])

def closest_point(G, x, time=False, parallel=False):
    """
    E. Agrell, T. Eriksson, A. Vardy and K. Zeger, "Closest point search in lattices," in IEEE Transactions on Information Theory, vol. 48, no. 8, pp. 2201-2214, Aug. 2002, doi: 10.1109/TIT.2002.800499.
    """
    # preprocess G
    G2 = G
    # G2 = self.reduce_LLL(G).float()
    # print(G2)
    Qt, Gt = torch.linalg.qr(G2.T, mode='complete')
    Q, G3 = Qt.T, Gt.T
    # print(Q, G3)
    # make sure diagonals of G are positive
    v = torch.ones(G3.shape[0], device=x.device)
    v[torch.diag(G3) == -1] = -1
    S = torch.diag(v)
    # print(S)
    G3 = G3 @ S
    Q = S @ Q

    # print(G3)

    H = torch.linalg.inv(G3)
    x = x @ Qt

    # find closest point in lattice
    U = []
    itertr = tqdm.trange(x.shape[0]) if time else range(x.shape[0])
    if parallel:
        U = Parallel(n_jobs=64)(delayed(quantize_schnorr_euchner)(H, x[i,:]) for i in itertr)
    else:
        for i in itertr:
            U.append(quantize_schnorr_euchner(H, x[i,:]))
    U = torch.stack(U)

    return U @ G2, U #np.dot(np.dot(Q, G), U)

def NSM(G, n_samples=1000, parallel=False): 
    n = G.shape[0]
    z = torch.rand((n_samples, n), device=G.device)
    x = z @ G
    xq, _ = closest_point(G, x, time=True, parallel=parallel)
    # xq = quantizer(x)
    x = x - xq
    norms = torch.linalg.norm(x, dim=1)
    NSM_estimate = torch.mean(norms**2) / n
    var_estimate = (1 / (n_samples - 1)) * (torch.mean(norms**4) - NSM_estimate**2)
    std = var_estimate ** 0.5
    return NSM_estimate, std

def grad(y, G):
    # d \|e\|^2 / db_{i,j}
    e = y @ G
    n = e.shape[1]
    # de2 = y.T @ e
    de2 = torch.einsum('bp,bq->bpq', y, e) # [batch_size, n, n], batched outer product
    de2[:, torch.arange(n), torch.arange(n)] = 2*e*y - 2*e[:,n-1][:, None] * y[:,n-1][:, None] * G[n-1,n-1] / torch.diag(G)[None, :]
    return torch.tril(de2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--n', default=4, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--M', default=1000000, type=int)
    parser.add_argument('--Mr', default=10000, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--parallel', action='store_true') # best used if batch_size > 1
    args = parser.parse_args()

    if args.use_gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'
        torch.set_num_threads(1) # for some reason this is faster than default
    batch_size = args.batch_size
    lr = args.lr
    n = args.n
    G = torch.eye(n).float().to(device)
    M = args.M
    Mr = args.Mr

    # Begin steepest descent
    for m in tqdm.trange(1, M+1):
        # print(f"iter={m}, G={G}")
        z = torch.rand((batch_size, n)).to(device)
        x = z @ G
        _, u = closest_point(G, x, parallel=args.parallel)
        y = z - u
        de2 = torch.mean(grad(y, G), dim=0)
        G = G - lr * (1 - (m / M)) * de2
        G[n-1, n-1] = (torch.prod(G[torch.arange(n-1), torch.arange(n-1)])) ** (-1)
        if m % Mr == 0:
            # perform reduction
            G_red = LLL(G)
            # rotate
            _, Gt = torch.linalg.qr(G_red.T, mode='complete')
            G_rot = Gt.T
            try:
                _ = torch.linalg.inv(G_rot)
                G = G_rot
            except RuntimeError:
                pass # if not invertible, skip the reduction/rotation  
        if m % (M // 10) == 0:
            nsm, std = NSM(G, 100000, args.parallel)
            print(f"iter={m}, NSM={nsm:.8f}, std={std:.8f}")
    os.makedirs("saved_lattices", exist_ok=True)
    torch.save(G, f'saved_lattices/{args.n}.pt')
    print(f"G = {G}")
    nsm, std = NSM(G, 1000000, args.parallel)
    print(f"NSM = {nsm:.8f}, std={std:.8f}")