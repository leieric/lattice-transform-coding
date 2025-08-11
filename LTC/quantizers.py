import torch.nn as nn
import torch
import numpy as np
import itertools

def get_lattice(name, d):
    if name == 'Hexagonal':
        return HexagonalQuantizer(d)
    elif name == 'HexagonalUnitVol':
        return HexagonalQuantizerUnitVol(d)
    elif name == 'Square':
        return SquareQuantizer(d)
    elif name == 'BrickWall':
        return BrickWallQuantizer(d)
    elif name == 'An':
        return AnQuantizer(d)
    elif name == 'Dn':
        return DnQuantizer(d)
    elif name == 'DnDual':
        return DnDualQuantizer(d)
    elif name == 'DnDualUnitVol':
        return DnDualQuantizerUnitVol(d)
    elif name == 'E8':
        return E8Quantizer(d)
    elif name == 'E8_2':
        return E8ProductQuantizer(d, 2)
    elif name == 'E8Product':
        n_product = d // 8
        print(f"n_product={n_product}")
        return E8ProductQuantizer(d, n_product)
    elif name == 'BarnesWallUnitVol':
        return BarnesWallQuantizerUnitVol(d)
    elif name == 'LeechUnitVol':
        return LeechQuantizerUnitVol(d)
    elif name == 'Leech2UnitVol':
        return Leech2QuantizerUnitVol(d)
    elif name == 'Leech2ProductUnitVol':
        n_product = d // 24
        print(f"n_product={n_product}")
        return Leech2ProductQuantizerUnitVol(d, n_product)
    elif name == 'Leech128ProductUnitVol':
        d = 128
        n_product = 5
        return Leech128ProductQuantizerUnitVol(d, n_product)
    elif name == 'L32UnitVol':
        return L32QuantizerUnitVol(d)
    elif name == 'L32ProductUnitVol':
        n_product = d // 32
        print(f"n_product={n_product}")
        return L32ProductQuantizerUnitVol(d, n_product)
    elif 'General' in name:
        lattice_name = name.split('_')[1]
        return GeneralQuantizer(d, lattice_name)
    else:
        raise Exception("Invalid Lattice name")

class DiamondQuantizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        G = torch.zeros(dim, dim)
        G[0,0] = 1
        for i in range(1, dim):
            G[i, i-1] = 0.5
            G[i, i] = -0.5
        return G

    def forward(self, x):
        # [batch_size, dim]
        y1 = torch.round(x)
        y2 = torch.round(x - 0.5) + 0.5
        # print(x.shape, y1.shape, y2.shape)
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        return y

class HexagonalQuantizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim != 2:
            raise Exception("Hexagonal Quantizer must be in dimension 2")
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(self.G, requires_grad=False)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        return torch.tensor([[1., 0.],
                             [0.5, 0.5*(3**0.5)]])

    def forward(self, x, retidx=False):
        # [batch_size, dim]
        x_scaled = torch.clone(x)
        x_scaled[:,1] /= 3**0.5

        y1 = torch.round(x_scaled)
        y2 = torch.round(x_scaled - 0.5) + 0.5
        y1[:,1] *= 3**0.5
        y2[:,1] *= 3**0.5
        # print(x.shape, y1.shape, y2.shape)
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        if retidx:
            return y, idx
        return y
    
class HexagonalQuantizerUnitVol(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim != 2:
            raise Exception("Hexagonal Quantizer must be in dimension 2")
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.a, self.G = self.make_unit_vol(self.G)

        self.G = nn.Parameter(self.G, requires_grad=False)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        M = torch.tensor([[1., 0.],
                          [0.5, 0.5*(3**0.5)]])
        return M
    
    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M

    def forward(self, x_in, retidx=False):
        # [batch_size, dim]
        x = torch.clone(x_in) / self.a
        x_scaled = torch.clone(x)
        x_scaled[:,1] /= 3**0.5

        y1 = torch.round(x_scaled)
        y2 = torch.round(x_scaled - 0.5) + 0.5
        y1[:,1] *= 3**0.5
        y2[:,1] *= 3**0.5
        # print(x.shape, y1.shape, y2.shape)
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :] * self.a
        if retidx:
            return y, idx
        return y

class AnQuantizer(nn.Module):
    """
    A_n quantizer, where n=dim. 
    Quantization occurs in the n-dimensional hyperplane \sum_{i=1}^{n+1} x_i = 0, in n+1-dimensional space. 
    Assumes x is n-d, projects to the (n+1)-d hyperplane, quantizes to nearest lattice point, and projects back to n-d space.
    Implementation follows Algorithm 3 in Ch. 20 of Conway and Sloane, "Sphere Packings, Lattices and Groups."

    We iterate over the samples in the batch to perform the sorting (not sure if there is a better way).
    We have 2 methods for the sorting of the delta values: 
    - Pre-sorting the sorted delta values across entire batch in parallel (better for small n)
    - using topk (introselect) for each sample (probably better for large n, not accounting for parallelization of argsort).
    """
    def __init__(self, dim, sort_method='presort'):
        super().__init__()
        self.dim = dim
        if sort_method == 'presort':
            self.project_int_to_plane = self._project_int_to_plane_presort
        elif sort_method == 'topk':
            self.project_int_to_plane = self._project_int_to_plane_topk
        else:
            raise Exception("sort_method invalid")
        G = self.init_generator_matrix(dim)
        _, _, Vh = torch.linalg.svd(G, full_matrices=False)
        self.transform =  Vh
        self.G = G @ Vh.T # n-d to n-d generator matrix is given by this

        self.G = nn.Parameter(self.G, requires_grad=False)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        M = torch.zeros((dim, dim+1))
        M[:, :dim] = -torch.eye(dim)
        M[:, 1:] += torch.eye(dim)
        return M

    def _project_int_to_plane_presort(self, xp, f_xp):
        """
        pre-sort delta values
        """
        Delta = torch.sum(f_xp, dim=1).int()
        delta_xp = xp - f_xp
        idx = torch.argsort(delta_xp, dim=1)
        for batch in range(xp.shape[0]):
            if Delta[batch] > 0:
                f_xp[batch, idx[batch, 0:Delta[batch]]] -= 1
            elif Delta[batch] < 0:
                f_xp[batch, idx[batch, Delta[batch]:]] += 1
        return f_xp
    
    def _project_int_to_plane_topk(self, xp, f_xp):
        """
        use top-k (introselect) to compute ordering of delta values
        """
        Delta = torch.sum(f_xp, dim=1).int() # batch_size 
        delta_xp = xp - f_xp #[batch_size, dim+1]
        for batch in range(xp.shape[0]):
            if Delta[batch] > 0:
                idx = torch.topk(-delta_xp[batch], Delta[batch]).indices
                f_xp[batch, idx] -= 1
            elif Delta[batch] < 0:
                idx = torch.topk(delta_xp[batch], Delta[batch].abs()).indices
                f_xp[batch, idx] += 1
        return f_xp
    
    def forward(self, x):
        # [batch_size, dim]
        self.transform = self.transform.to(x.device)
        xp = x @ self.transform
        f_xp = torch.round(xp) # [batch_size, dim+1]
        f_xp = self.project_int_to_plane(xp, f_xp)
        f_xp = f_xp @ self.transform.T
        return f_xp
    # def forward(self, x):
    #     # [batch_size, dim+1]
    #     s = torch.sum(x, dim=1) # [batch_size]
    #     xp = x - (1/(self.dim+1)) * s[:,None] * torch.ones((x.shape[0], self.dim+1)) #[batch_size, dim+1]
    #     f_xp = torch.round(xp) # [batch_size, dim+1]
    #     f_xp = self.project_int_to_plane(xp, f_xp)
    #     return f_xp, xp

class DnQuantizer(nn.Module):
    """
    D_n quantizer.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        M = -torch.eye(dim)
        M[1:, :dim-1] += torch.eye(dim-1)
        M[0, 1] = -1
        return M
        # return torch.tensor([[-1., 0.], [1., -1.]])

    def forward(self, x):
        # [batch_size, dim]
        # Compute f(x), g(x)
        f_x = torch.round(x)
        delta_x = x - f_x
        delta_x_abs = torch.abs(delta_x)
        k = torch.argmax(delta_x_abs, dim=1)
        g_x = torch.clone(f_x)
        fix = torch.ones(delta_x.shape[0], device=delta_x.device) # default: if rounded down, round up
        fix[torch.where((torch.gather(delta_x, dim=1, index=k.unsqueeze(1)) < 0))[0]] = -1
        g_x.scatter_add_(index=k.unsqueeze(1), dim=1, src=fix[:, None]) 

        # return whichever has even component sum
        g_comp_sum = torch.sum(g_x, dim=1)
        idx_g_even = torch.where(g_comp_sum % 2 == 0)[0]
        f_x[idx_g_even, :] = g_x[idx_g_even, :]
        return f_x

class E8Quantizer(nn.Module):
    """
    E_8 quantizer, equivalent to E_8^*.
    """
    def __init__(self, dim):
        super().__init__()
        assert dim == 8
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(self.G, requires_grad=False)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        M = torch.eye(dim)
        M[0, 0] = 2.
        M[1:, 0:dim-1] += -torch.eye(dim-1)
        M[dim-1, :] = 0.5
        return M
        # return torch.tensor([[-1., 0.], [1., -1.]])
    
    def NSM(self, n_samples=1000): 
        n = self.G.shape[0]
        z = torch.rand((n_samples, n), device=self.G.device)
        x = z @ self.G
        # xq, _ = closest_point(G, x, time=True, parallel=parallel)
        xq = self.forward(x)
        # xq = quantizer(x)
        x = x - xq
        norms = torch.linalg.norm(x, dim=1)
        NSM_estimate = torch.mean(norms**2) / n
        var_estimate = (1 / (n_samples - 1)) * (torch.mean(norms**4) - NSM_estimate**2)
        std = var_estimate ** 0.5
        return NSM_estimate, std


    def _D8_quantize(self, x):
        # [batch_size, dim]
        # Compute f(x), g(x)
        f_x = torch.round(x)
        delta_x = x - f_x
        delta_x_abs = torch.abs(delta_x)
        k = torch.argmax(delta_x_abs, dim=1)
        g_x = torch.clone(f_x)
        fix = torch.ones(delta_x.shape[0], device=delta_x.device) # default: if rounded down, round up
        fix[torch.where((torch.gather(delta_x, dim=1, index=k.unsqueeze(1)) < 0))[0]] = -1
        g_x.scatter_add_(index=k.unsqueeze(1), dim=1, src=fix[:, None]) 

        # return whichever has even component sum
        g_comp_sum = torch.sum(g_x, dim=1)
        idx_g_even = torch.where(g_comp_sum % 2 == 0)[0]
        f_x[idx_g_even, :] = g_x[idx_g_even, :]
        return f_x
    
    def forward(self, x):
        # D_8 quantize x and coset 1/2, choose the closest
        # [batch_size, dim]
        y1 = self._D8_quantize(x)
        y2 = self._D8_quantize(x - 0.5) + 0.5
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        return y

class E8ProductQuantizer(E8Quantizer):
    """
    E_8 product quantizer.
    """
    def __init__(self, dim, n_product=2):
        super().__init__(8)
        self.n_product = n_product
        assert dim == 8 * n_product
        self.dim = dim
        Gs = [self.init_generator_matrix(8) for _ in range(n_product)]
        G = torch.block_diag(*Gs)
        # self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(G, requires_grad=False)

    def _E8_quantize(self, x):
        # D_8 quantize x and coset 1/2, choose the closest
        # [batch_size, dim]
        y1 = self._D8_quantize(x)
        y2 = self._D8_quantize(x - 0.5) + 0.5
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        return y
    
    def forward(self, x):
        # x: [bsize, 8*n_product]
        bsize, dim = x.shape
        x = x.reshape(bsize, self.n_product, 8)
        x = x.reshape(bsize*self.n_product, 8)
        x_q = self._E8_quantize(x)
        x_q = x_q.reshape(bsize, self.n_product, 8)
        x_q = x_q.reshape(bsize, 8*self.n_product)
        return x_q

        # ys = []
        # for i in range(self.n_product):
        #     ys.append(self._E8_quantize(x[:, 8*i:8*(i+1)]))
        # y = torch.cat(ys, dim=1)
        # return y

    
class BarnesWallQuantizerUnitVol(nn.Module):
    """
    Barnes Wall quantizer
    """
    def __init__(self, dim):
        super().__init__()
        assert dim == 16
        self.dim = dim
        self.G = get_generator_matrix("BarnesWall")
        self.a, self.G = self.make_unit_vol(self.G)

        C = self._RM14_codewords() # [32, 16]

        self.G = nn.Parameter(self.G, requires_grad=False)
        self.C_rep = nn.Parameter(C, requires_grad=False) # coset representatives

    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M
    
    def NSM(self, n_samples=1000): 
        n = self.G.shape[0]
        z = torch.rand((n_samples, n), device=self.G.device)
        x = z @ self.G
        # xq, _ = closest_point(G, x, time=True, parallel=parallel)
        xq = self.forward(x)
        # xq = quantizer(x)
        x = x - xq
        norms = torch.linalg.norm(x, dim=1)
        NSM_estimate = torch.mean(norms**2) / n
        var_estimate = (1 / (n_samples - 1)) * (torch.mean(norms**4) - NSM_estimate**2)
        std = var_estimate ** 0.5
        return NSM_estimate, std
    
    def _RM14_codewords(self):
        # codewords of the [16, 5, 8] Reed-Muller(1, 4) code, using generator matrix from last 5 rows of BarnesWall generator matrix
        G = torch.tensor([
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
         ])
        binary_inputs = torch.tensor([list(i) for i in itertools.product([0, 1], repeat=5)])
        codewords = binary_inputs @ G
        return codewords

    def _2D16_quantize(self, x_in):
        x = torch.div(x_in, 2)
        f_x = torch.round(x)
        mask = torch.sum(f_x, dim=1) % 2 == 1
        col = torch.argmax(torch.abs(x - f_x), dim=1)[mask]
        row = torch.arange(x.shape[0], device=x_in.device)[mask]
        x_ = x[row, col]
        up = torch.ceil(x_)
        down = torch.floor(x_)
        f_x[row, col] = up + down - f_x[row, col]
        f_x = torch.mul(f_x, 2)
        return f_x
    
    def forward(self, x_in):
        # [batch_size, 24]
        x = torch.div(x_in, self.a)
        # x = x_in
        inputs = x[:, None, :] - self.C_rep[None, :, :] # [bsize, 32, 16]
        X = self._2D16_quantize(inputs.reshape(-1, 16)).reshape(-1, 32, 16) + self.C_rep[None, :, :]

        D = torch.linalg.norm(X - x[:, None, :], dim=2)**2 # [bsize, 32]
        idx_keep = torch.argmin(D, dim=1) # [bsize]
        # print(idx_keep)
        y = X[torch.arange(X.shape[0]), idx_keep, :]
        y = torch.mul(y, self.a)
        return y


    
class L32QuantizerUnitVol(nn.Module):
    """
    L_32 quantizer
    """
    def __init__(self, dim):
        super().__init__()
        assert dim == 32
        self.dim = dim
        self.G = get_generator_matrix("L32")
        self.a, self.G = self.make_unit_vol(self.G)

        C = self._RM15_codewords() # [32, 16]

        self.G = nn.Parameter(self.G, requires_grad=False)
        self.C_rep = nn.Parameter(C, requires_grad=False) # coset representatives

    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M
    
    def NSM(self, n_samples=1000): 
        n = self.G.shape[0]
        z = torch.rand((n_samples, n), device=self.G.device)
        x = z @ self.G
        # xq, _ = closest_point(G, x, time=True, parallel=parallel)
        xq = self.forward(x)
        # xq = quantizer(x)
        x = x - xq
        norms = torch.linalg.norm(x, dim=1)
        NSM_estimate = torch.mean(norms**2) / n
        var_estimate = (1 / (n_samples - 1)) * (torch.mean(norms**4) - NSM_estimate**2)
        std = var_estimate ** 0.5
        return NSM_estimate, std
    
    def _RM15_codewords(self):
        # codewords of the [32, 6, 16] Reed-Muller(1, 5) code
        G = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
         ])
        binary_inputs = torch.tensor([list(i) for i in itertools.product([0, 1], repeat=6)])
        codewords = binary_inputs @ G
        return codewords

    def _2D32_quantize(self, x_in):
        x = torch.div(x_in, 2)
        f_x = torch.round(x)
        mask = torch.sum(f_x, dim=1) % 2 == 1
        col = torch.argmax(torch.abs(x - f_x), dim=1)[mask]
        row = torch.arange(x.shape[0], device=x_in.device)[mask]
        x_ = x[row, col]
        up = torch.ceil(x_)
        down = torch.floor(x_)
        f_x[row, col] = up + down - f_x[row, col]
        f_x = torch.mul(f_x, 2)
        return f_x
    
    def forward(self, x_in):
        # [batch_size, 24]
        x = torch.div(x_in, self.a)
        # x = x_in
        inputs = x[:, None, :] - self.C_rep[None, :, :] # [bsize, 64, 32]
        X = self._2D32_quantize(inputs.reshape(-1, 32)).reshape(-1, 64, 32) + self.C_rep[None, :, :]

        D = torch.linalg.norm(X - x[:, None, :], dim=2)**2 # [bsize, 64]
        idx_keep = torch.argmin(D, dim=1) # [bsize]
        # print(idx_keep)
        y = X[torch.arange(X.shape[0]), idx_keep, :]
        # print(y)
        y = torch.mul(y, self.a)
        # print(y)
        return y

class L32ProductQuantizerUnitVol(L32QuantizerUnitVol):
    """
    L32 product quantizer
    """
    def __init__(self, dim, n_product=2):
        super().__init__(32)
        self.n_product = n_product
        assert dim == 32 * n_product
        self.dim = dim
        Gs = [self.G.data for _ in range(n_product)]
        G = torch.block_diag(*Gs)
        
        # self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(G, requires_grad=False)

    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M

    def _L32_quantize(self, x_in):
        # [batch_size, 24]
        x = torch.div(x_in, self.a)
        # x = x_in
        inputs = x[:, None, :] - self.C_rep[None, :, :] # [bsize, 64, 32]
        X = self._2D32_quantize(inputs.reshape(-1, 32)).reshape(-1, 64, 32) + self.C_rep[None, :, :]

        D = torch.linalg.norm(X - x[:, None, :], dim=2)**2 # [bsize, 64]
        idx_keep = torch.argmin(D, dim=1) # [bsize]
        # print(idx_keep)
        y = X[torch.arange(X.shape[0]), idx_keep, :]
        # print(y)
        y = torch.mul(y, self.a)
        # print(y)
        return y
    
    def forward(self, x):
        # x: [bsize, 32*n_product]
        # ys = []
        # for i in range(self.n_product):
        #     ys.append(self._L32_quantize(x[:, 32*i:32*(i+1)]))
        # y = torch.cat(ys, dim=1)
        # return y
        bsize, dim = x.shape
        x = x.reshape(bsize, self.n_product, 32)
        x = x.reshape(bsize*self.n_product, 32)
        x_q = self._L32_quantize(x)
        x_q = x_q.reshape(bsize, self.n_product, 32)
        x_q = x_q.reshape(bsize, 32*self.n_product)
        return x_q
    
# class LnQuantizerUnitVol(nn.Module):
#     """
#     L_n quantizer
#     """
#     def __init__(self, dim):
#         super().__init__()
#         assert dim == 32
#         self.dim = dim
#         self.G = get_generator_matrix("L32")
#         self.a, self.G = self.make_unit_vol(self.G)

#         self.m = np.log2(dim)
#         C = self._RM_codewords(self.m) # [32, 16]

#         self.G = nn.Parameter(self.G, requires_grad=False)
#         self.C_rep = nn.Parameter(C, requires_grad=False) # coset representatives

#     def make_unit_vol(self, M):
#         n = M.shape[0]
#         vol = torch.sqrt(torch.linalg.det(M @ M.T))
#         a = 1 / (vol ** (1/n))
#         M = a * M
#         return a, M
    
#     def NSM(self, n_samples=1000): 
#         n = self.G.shape[0]
#         z = torch.rand((n_samples, n), device=self.G.device)
#         x = z @ self.G
#         # xq, _ = closest_point(G, x, time=True, parallel=parallel)
#         xq = self.forward(x)
#         # xq = quantizer(x)
#         x = x - xq
#         norms = torch.linalg.norm(x, dim=1)
#         NSM_estimate = torch.mean(norms**2) / n
#         var_estimate = (1 / (n_samples - 1)) * (torch.mean(norms**4) - NSM_estimate**2)
#         std = var_estimate ** 0.5
#         return NSM_estimate, std
    
#     def _RM_codewords(self, m):
#         # codewords of the [2^m, m+1, 2^(m-1)] Reed-Muller(1, m) code
#         G = torch.tensor([
#             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#             [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
#             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
#             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
#             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#          ])
#         binary_inputs = torch.tensor([list(i) for i in itertools.product([0, 1], repeat=6)])
#         codewords = binary_inputs @ G
#         return codewords

#     def _2D32_quantize(self, x_in):
#         x = torch.div(x_in, 2)
#         f_x = torch.round(x)
#         mask = torch.sum(f_x, dim=1) % 2 == 1
#         col = torch.argmax(torch.abs(x - f_x), dim=1)[mask]
#         row = torch.arange(x.shape[0], device=x_in.device)[mask]
#         x_ = x[row, col]
#         up = torch.ceil(x_)
#         down = torch.floor(x_)
#         f_x[row, col] = up + down - f_x[row, col]
#         f_x = torch.mul(f_x, 2)
#         return f_x
    
#     def forward(self, x_in):
#         # [batch_size, 24]
#         x = torch.div(x_in, self.a)
#         # x = x_in
#         inputs = x[:, None, :] - self.C_rep[None, :, :] # [bsize, 64, 32]
#         X = self._2D32_quantize(inputs.reshape(-1, 32)).reshape(-1, 64, 32) + self.C_rep[None, :, :]

#         D = torch.linalg.norm(X - x[:, None, :], dim=2)**2 # [bsize, 64]
#         idx_keep = torch.argmin(D, dim=1) # [bsize]
#         # print(idx_keep)
#         y = X[torch.arange(X.shape[0]), idx_keep, :]
#         # print(y)
#         y = torch.mul(y, self.a)
#         # print(y)
#         return y
    
class LeechQuantizerUnitVol(nn.Module):
    """
    Leech quantizer
    """
    def __init__(self, dim):
        super().__init__()
        assert dim == 24
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.a = 1 / np.sqrt(8)
        # self.G = self.a * self.G

        C = self._Golay_codewords() # [4096, 24]
        u = torch.tensor([-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        C_rep = torch.cat((2*C, 2*C + u[None, :]), dim=0) # [8192, 24]

        self.G = nn.Parameter(self.G, requires_grad=False)
        self.C_rep = nn.Parameter(C_rep, requires_grad=False)

    def _Golay_codewords(self):
        G = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        binary_inputs = torch.tensor([list(i) for i in itertools.product([0, 1], repeat=12)])
        codewords = binary_inputs @ G
        return codewords
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        return torch.tensor([
            [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
            [-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) / np.sqrt(8)

    def _4D24_quantize(self, x_in):
        x = torch.div(x_in, 4)
        f_x = torch.round(x)
        mask = torch.sum(f_x, dim=1) % 2 == 1
        col = torch.argmax(torch.abs(x - f_x), dim=1)[mask]
        row = torch.arange(x.shape[0], device=x_in.device)[mask]
        x_ = x[row, col]
        up = torch.ceil(x_)
        down = torch.floor(x_)
        f_x[row, col] = up + down - f_x[row, col]
        f_x = torch.mul(f_x, 4)
        return f_x
    
    def forward(self, x_in):
        # [batch_size, 24]
        x = torch.div(x_in, self.a)
        inputs = x[:, None, :] - self.C_rep[None, :, :] # [bsize, 8192, 24]
        X = self._4D24_quantize(inputs.reshape(-1, 24)).reshape(-1, 8192, 24) + self.C_rep[None, :, :]

        D = torch.linalg.norm(X - x[:, None, :], dim=2)**2 # [bsize, 8192]
        idx_keep = torch.argmin(D, dim=1) # [bsize]
        # print(idx_keep)
        y = X[torch.arange(X.shape[0]), idx_keep, :]
        y = torch.mul(y, self.a)
        return y

# from pykeops.torch import LazyTensor
# def compute_D_argmin(X, x):
#     # X: [batch_size, 4096, 24]
#     # x: [batch_size, 24]
#     bsize = X.shape[0]
#     # X = X.reshape(bsize*4096, 24)
#     # x = x[:, None, :].reshape(bsize*4096, 24)
#     X_i = LazyTensor(X.view(bsize, 4096, 1, 24))
#     x_j = LazyTensor(x.view(bsize, 1, 1, 24))
#     idx = ((X_i - x_j)**2).sum(dim=3).argmin(dim=1)
#     return idx.squeeze()

class Leech2QuantizerUnitVol(nn.Module):
    """
    Leech quantizer
    """
    def __init__(self, dim):
        super().__init__()
        assert dim == 24
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.a = 1 / np.sqrt(8)

        A = torch.tensor([[0,0,0,0,0,0,0,0],[4,0,0,0,0,0,0,0],[2,2,2,2,0,0,0,0],[-2,2,2,2,0,0,0,0],[2,2,0,0,2,2,0,0],[-2,2,0,0,2,2,0,0],[2,2,0,0,0,0,2,2],[-2,2,0,0,0,0,2,2],[2,0,2,0,2,0,2,0],[-2,0,2,0,2,0,2,0],[2,0,2,0,0,2,0,2],[-2,0,2,0,0,2,0,2],[2,0,0,2,2,0,0,2],[-2,0,0,2,2,0,0,2],[2,0,0,2,0,2,2,0],[-2,0,0,2,0,2,2,0]])
        T = torch.tensor([[0,0,0,0,0,0,0,0],[2,2,2,0,0,2,0,0],[2,2,0,2,0,0,0,2],[2,0,2,2,0,0,2,0],[0,2,2,2,2,0,0,0],[2,2,0,0,2,0,2,0],[2,0,2,0,2,0,0,2],[2,0,0,2,2,2,0,0],[-3,1,1,1,1,1,1,1],[3,-1,-1,1,1,-1,1,1],[3,-1,1,-1,1,1,1,-1],[3,1,-1,-1,1,1,-1,1],[3,1,1,1,1,-1,-1,-1],[3,-1,1,1,-1,1,-1,1],[3,1,-1,1,-1,1,1,-1],[3,1,1,-1,-1,-1,1,1]])
        table = torch.tensor([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],[1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14],[2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13],[3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12],[4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11],[5,4,7,6,1,0,3,2,13,12,15,14,9,8,11,10],[6,7,4,5,2,3,0,1,14,15,12,13,10,11,8,9],[7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8],[8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7],[9,8,11,10,13,12,15,14,1,0,3,2,5,4,7,6],[10,11,8,9,14,15,12,13,2,3,0,1,6,7,4,5],[11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4],[12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3],[13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2],[14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1],[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]).int()
        at_pairs = torch.cartesian_prod(torch.arange(16), torch.arange(16)) # [256, 2]
        AT = A[at_pairs[:,0]] + T[at_pairs[:,1]] # [256, 8]
        a = at_pairs[:,0].repeat_interleave(16) #[4096]
        b = at_pairs[:,1].repeat_interleave(16)
        c = table[at_pairs[:,0], at_pairs[:,1]].repeat_interleave(16)
        t = torch.arange(16).repeat(256)

        self.G = nn.Parameter(self.G, requires_grad=False)
        self.AT = nn.Parameter(AT, requires_grad=False)
        self.idx_a = nn.Parameter(a, requires_grad=False)
        self.idx_b = nn.Parameter(b, requires_grad=False)
        self.idx_c = nn.Parameter(c, requires_grad=False)
        self.idx_t = nn.Parameter(t, requires_grad=False)

        # self.A = nn.Parameter(A, requires_grad=False)
        # self.T = nn.Parameter(T, requires_grad=False)
        # self.table = nn.Parameter(table, requires_grad=False)

    def NSM(self, n_samples=1000): 
        n = self.G.shape[0]
        z = torch.rand((n_samples, n), device=self.G.device)
        x = z @ self.G
        # xq, _ = closest_point(G, x, time=True, parallel=parallel)
        xq = self.forward(x)
        # xq = quantizer(x)
        x = x - xq
        norms = torch.linalg.norm(x, dim=1)
        NSM_estimate = torch.mean(norms**2) / n
        var_estimate = (1 / (n_samples - 1)) * (torch.mean(norms**4) - NSM_estimate**2)
        std = var_estimate ** 0.5
        return NSM_estimate, std
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        return torch.tensor([[8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
        [2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
        [0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
        [-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) / np.sqrt(8)

    def _D8_quantize(self, x):
        # [batch_size, dim]
        # Compute f(x), g(x)
        f_x = torch.round(x)
        delta_x = x - f_x
        delta_x_abs = torch.abs(delta_x)
        k = torch.argmax(delta_x_abs, dim=1)
        g_x = torch.clone(f_x)
        fix = torch.ones(delta_x.shape[0], device=delta_x.device, dtype=x.dtype) # default: if rounded down, round up
        fix[torch.where((torch.gather(delta_x, dim=1, index=k.unsqueeze(1)) < 0))[0]] = -1
        g_x.scatter_add_(index=k.unsqueeze(1), dim=1, src=fix[:, None]) 

        # return whichever has even component sum
        f_comp_sum = torch.sum(f_x, dim=1)
        idx_f_even = torch.where(f_comp_sum % 2 == 0)[0]
        g_x[idx_f_even, :] = f_x[idx_f_even, :]
        return g_x
    
    def _4E8_quantize(self, x_in):
        # D_8 quantize x and coset 1/2, choose the closest
        # [batch_size, dim]
        x = torch.div(x_in, 4)
        y1 = self._D8_quantize(x)
        y2 = self._D8_quantize(x - 0.5) + 0.5
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        y = torch.mul(y, 4)
        return y
    
    def forward(self, x_in):
        # [batch_size, 24]
        x = torch.div(x_in, self.a)
        # x = x.double()
        # 1. Precomputation stage
        bsize = x.shape[0]
        # at_pairs = torch.cartesian_prod(torch.arange(16), torch.arange(16)) # [256, 2]
        # AT = self.A[at_pairs[:,0]] + self.T[at_pairs[:,1]] # [256, 8]
        # AT = AT.repeat(1, 3) # [256, 24]
        x1 = x[:, None, 0:8] - self.AT[None, :, :] # [batch_size, 256, 8]
        x2 = x[:, None, 8:16] - self.AT[None, :, :] # [batch_size, 256, 8]
        x3 = x[:, None, 16:24] - self.AT[None, :, :] # [batch_size, 256, 8]
        xx = torch.cat((x1, x2, x3), dim=0) #[3*batch_size, 256, 8]
        xxq = self._4E8_quantize(xx.reshape(-1, 8)).reshape(-1, 256, 8) + self.AT[None, :, :]
        # xq1 = self._4E8_quantize(x1.reshape(-1, 8)).reshape(-1, 256, 8) + self.AT[None, :, :]
        # xq2 = self._4E8_quantize(x2.reshape(-1, 8)).reshape(-1, 256, 8) + self.AT[None, :, :]
        # xq3 = self._4E8_quantize(x3.reshape(-1, 8)).reshape(-1, 256, 8) + self.AT[None, :, :]
        xxq = xxq.reshape(3, bsize, 256, 8)
        pre = torch.cat((xxq[0], xxq[1], xxq[2]), dim=2) # [batch_size, 256, 24]
        pre = pre.reshape((-1, 16, 16, 24))

        # 1. Precomputation stage
        # AT = self.A[:, None, :] + self.T[None, :, :] # [16, 16, 8]


        # 2. Main stage
        # a = at_pairs[:,0].repeat_interleave(16) #[4096]
        # b = at_pairs[:,1].repeat_interleave(16)
        # c = self.table[at_pairs[:,0], at_pairs[:,1]].repeat_interleave(16)
        # t = torch.arange(16).repeat(256)
        # print(a, b, c, t)
        x1 = pre[:, self.idx_a, self.idx_t, 0:8] # [batch_size, 4096, 8]
        x2 = pre[:, self.idx_b, self.idx_t, 8:16] # [batch_size, 4096, 8]
        x3 = pre[:, self.idx_c, self.idx_t, 16:24] # [batch_size, 4096, 8]
        X = torch.cat((x1, x2, x3), dim=2) # [batch_size, 4096, 24]
        D = torch.linalg.norm(X - x[:, None, :], dim=2) # [batch_size, 4096]
        idx_keep = torch.argmin(D, dim=1) # [batch_size]
        # idx_keep = compute_D_argmin(X, x)
        y = X[torch.arange(bsize, device=x.device), idx_keep, :]
        y = torch.mul(y, self.a)
        return y
    
class Leech128ProductQuantizerUnitVol(Leech2QuantizerUnitVol):
    """
    Leech quantizer x 5 with one E8
    """
    def __init__(self, dim, n_product=5):
        super().__init__(24)
        self.n_product = n_product
        # assert dim == 24 * n_product
        assert n_product == 5
        assert dim == 128
        self.dim = dim
        Gs = [self.G.data for _ in range(n_product)]
        Gs.append(get_generator_matrix("E8"))
        G = torch.block_diag(*Gs)
        # self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(G, requires_grad=False)

    def _L24_quantize(self, x_in):
        # [batch_size, 24]
        x = torch.div(x_in, self.a)
        bsize = x.shape[0]
        x1 = x[:, None, 0:8] - self.AT[None, :, :] # [batch_size, 256, 8]
        x2 = x[:, None, 8:16] - self.AT[None, :, :] # [batch_size, 256, 8]
        x3 = x[:, None, 16:24] - self.AT[None, :, :] # [batch_size, 256, 8]
        xx = torch.cat((x1, x2, x3), dim=0) #[3*batch_size, 256, 8]
        xxq = self._4E8_quantize(xx.reshape(-1, 8)).reshape(-1, 256, 8) + self.AT[None, :, :]
        xxq = xxq.reshape(3, bsize, 256, 8)
        pre = torch.cat((xxq[0], xxq[1], xxq[2]), dim=2) # [batch_size, 256, 24]
        pre = pre.reshape((-1, 16, 16, 24))
        x1 = pre[:, self.idx_a, self.idx_t, 0:8] # [batch_size, 4096, 8]
        x2 = pre[:, self.idx_b, self.idx_t, 8:16] # [batch_size, 4096, 8]
        x3 = pre[:, self.idx_c, self.idx_t, 16:24] # [batch_size, 4096, 8]
        X = torch.cat((x1, x2, x3), dim=2) # [batch_size, 4096, 24]
        D = torch.linalg.norm(X - x[:, None, :], dim=2) # [batch_size, 4096]
        idx_keep = torch.argmin(D, dim=1) # [batch_size]
        y = X[torch.arange(bsize, device=x.device), idx_keep, :]
        y = torch.mul(y, self.a)
        return y
    
    def _E8_quantize(self, x):
        # D_8 quantize x and coset 1/2, choose the closest
        # [batch_size, dim]
        y1 = self._D8_quantize(x)
        y2 = self._D8_quantize(x - 0.5) + 0.5
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        return y
    
    def forward(self, x_in):
        # x_in: [bsize, 128]
        # ys = []
        # for i in range(self.n_product):
        #     ys.append(self._L24_quantize(x[:, 24*i:24*(i+1)]))
        # y = torch.cat(ys, dim=1)
        # return y
        bsize, dim = x_in.shape
        x = x_in[:, 0:120].reshape(bsize, self.n_product, 24)
        x = x.reshape(bsize*self.n_product, 24)
        x_q = self._L24_quantize(x)
        x_q = x_q.reshape(bsize, self.n_product, 24)
        x_q = x_q.reshape(bsize, 24*self.n_product) # [bsize, 120]

        x_last = self._E8_quantize(x_in[:, 120:128]) # [bsize, 8]
        x_q = torch.cat((x_q, x_last), dim=1)
        return x_q




class Leech2ProductQuantizerUnitVol(Leech2QuantizerUnitVol):
    """
    Leech quantizer
    """
    def __init__(self, dim, n_product=2):
        super().__init__(24)
        self.n_product = n_product
        assert dim == 24 * n_product
        self.dim = dim
        Gs = [self.G.data for _ in range(n_product)]
        G = torch.block_diag(*Gs)
        # self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(G, requires_grad=False)

    def _L24_quantize(self, x_in):
        # [batch_size, 24]
        x = torch.div(x_in, self.a)
        bsize = x.shape[0]
        x1 = x[:, None, 0:8] - self.AT[None, :, :] # [batch_size, 256, 8]
        x2 = x[:, None, 8:16] - self.AT[None, :, :] # [batch_size, 256, 8]
        x3 = x[:, None, 16:24] - self.AT[None, :, :] # [batch_size, 256, 8]
        xx = torch.cat((x1, x2, x3), dim=0) #[3*batch_size, 256, 8]
        xxq = self._4E8_quantize(xx.reshape(-1, 8)).reshape(-1, 256, 8) + self.AT[None, :, :]
        xxq = xxq.reshape(3, bsize, 256, 8)
        pre = torch.cat((xxq[0], xxq[1], xxq[2]), dim=2) # [batch_size, 256, 24]
        pre = pre.reshape((-1, 16, 16, 24))
        x1 = pre[:, self.idx_a, self.idx_t, 0:8] # [batch_size, 4096, 8]
        x2 = pre[:, self.idx_b, self.idx_t, 8:16] # [batch_size, 4096, 8]
        x3 = pre[:, self.idx_c, self.idx_t, 16:24] # [batch_size, 4096, 8]
        X = torch.cat((x1, x2, x3), dim=2) # [batch_size, 4096, 24]
        D = torch.linalg.norm(X - x[:, None, :], dim=2) # [batch_size, 4096]
        idx_keep = torch.argmin(D, dim=1) # [batch_size]
        y = X[torch.arange(bsize, device=x.device), idx_keep, :]
        y = torch.mul(y, self.a)
        return y
    
    def forward(self, x):
        # x: [bsize, 24*n_product]
        # ys = []
        # for i in range(self.n_product):
        #     ys.append(self._L24_quantize(x[:, 24*i:24*(i+1)]))
        # y = torch.cat(ys, dim=1)
        # return y
        bsize, dim = x.shape
        x = x.reshape(bsize, self.n_product, 24)
        x = x.reshape(bsize*self.n_product, 24)
        x_q = self._L24_quantize(x)
        x_q = x_q.reshape(bsize, self.n_product, 24)
        x_q = x_q.reshape(bsize, 24*self.n_product)
        return x_q

def get_generator_matrix(name, dim=2):
    if name == 'E8':
        dim = 8
        M = torch.eye(dim)
        M[0, 0] = 2.
        M[1:, 0:dim-1] += -torch.eye(dim-1)
        M[dim-1, :] = 0.5
        return M
    elif name == 'Square':
        return torch.eye(dim)
    elif name == "Hexagonal":
        return torch.tensor([[1., 0.],
                             [0.5, 0.5*(3**0.5)]])
    elif name == "DnDual":
        M = torch.eye(dim)
        M[dim-1, :] = 0.5
        return M
    elif name == "Dn":
        M = -torch.eye(dim)
        M[1:, :dim-1] += torch.eye(dim-1)
        M[0, 1] = -1
        return M
    elif name == 'BarnesWall':
        return torch.tensor([
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]) * 1.
    elif name == 'Leech':
        return torch.tensor([
            [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
            [-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]) * 1.#/ np.sqrt(8)
    elif name == 'L32':
        return torch.tensor([
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
            [2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]) * 1.
        # return torch.tensor([
        #     [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
        #     [2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
        #     [2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
        #     [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        #     [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
        #     [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
        #     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # ]) * 1.
    else:
        raise Exception("Invalid lattice name")

class GeneralQuantizer(nn.Module):
    """
    General quantizer. 
    """
    def __init__(self, dim, lattice_name='E8', G=None):
        super().__init__()
        self.dim = dim
        if G is None:
            self.G = get_generator_matrix(lattice_name)
        else:
            self.G = G
        
        # self.G = self.init_generator_matrix(dim)
        # self.a, self.G = self.make_unit_vol(self.G)

        # preprocess G
        self.G2 = self.G
        self.Q, self.G_diag = torch.linalg.qr(self.G2.T, mode='complete')

        # make sure diagonals of G are positive
        self.S = torch.diag(torch.sign(torch.diag(self.G_diag)))
        self.Q = self.Q @ self.S
        self.G_diag = self.S @ self.G_diag

        self.H = torch.linalg.inv(self.G_diag)

        # self.Q = nn.Parameter(self.Q, requires_grad=False)
        # self.H = nn.Parameter(self.H, requires_grad=False)
        # self.G_diag = nn.Parameter(self.G_diag, requires_grad=False)
        self.G = nn.Parameter(self.G, requires_grad=True)

    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M

        # return torch.tensor(np.asarray([[0.5, 1], [0.5, 0]], dtype=np.float32)).float()

    def sgn(self, x):
        return 2*(x > 0).to(torch.get_default_dtype()) - 1

    def quantize_schnorr_euchner(self, H, x):
        """
        The Schnorr-Euchner closest point search. An improved variant of the Pohst (1981) strategy.
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
        step[k] = self.sgn(y)

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
                    step[k] = self.sgn(y)
                else:
                    # found closer point
                    u_best = torch.clone(u) # u.copy()
                    best_dist = new_dist

                    # move up
                    k += 1
                    u[k] += step[k]
                    y = (E[k, k] - u[k]) / H[k, k]

                    # change direction
                    step[k] = -step[k] - self.sgn(step[k])
            else:
            # no point in sublattice is better than current best point 
            # Babai point will just return the first time this case happens?
                # return u_best
                if k == dim - 1:
                    return u_best
                else:
                    # move up
                    k += 1
                    u[k] += step[k]
                    y = (E[k, k] - u[k]) / H[k, k]

                    # change direction
                    step[k] = -step[k] - self.sgn(step[k])
    
    def forward2(self, x):
        # x: [batch_size, dim]
        X = x.T
        X = self.Q.T @ X

        # find closest point in lattice
        U = []
        for i in range(X.shape[1]):
            U.append(self.quantize_schnorr_euchner(self.H.T, X[:, i]))
        # print(U)
        U = torch.stack(U).T
        # print(U.shape)
        xq = (self.Q @ self.G_diag @ U).T
        return xq
    
    def forward(self, x):
        # preprocess G
        G2 = self.G
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
        with torch.no_grad():
            for i in range(x.shape[0]):
                U.append(self.quantize_schnorr_euchner(H, x[i,:]))
        U = torch.stack(U)

        return U @ G2#, U #np.dot(np.dot(Q, G), U)

class DnDualQuantizer(nn.Module):
    """
    D_n^* quantizer.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        M = torch.eye(dim)
        M[dim-1, :] = 0.5
        return M

    def forward(self, x):
        # [batch_size, dim]
        # Compute f(x), g(x)
        y1 = torch.round(x)
        y2 = torch.round(x - 0.5) + 0.5
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        return y
    
class DnDualQuantizerUnitVol(nn.Module):
    """
    D_n^* quantizer.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.a, self.G = self.make_unit_vol(self.G)
        self.G = nn.Parameter(self.G, requires_grad=False)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        M = torch.eye(dim)
        M[dim-1, :] = 0.5
        return M
    
    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M

    def forward(self, x_in):
        # [batch_size, dim]
        # Compute f(x), g(x)
        x = torch.div(x_in, self.a)
        y1 = torch.round(x)
        y2 = torch.round(x - 0.5) + 0.5
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        y = torch.mul(y, self.a)
        return y
    
class GeneralQuantizerUnitVol(nn.Module):
    """
   General quantizer.
    """
    def __init__(self, dim, G):
        super().__init__()
        self.dim = dim
        self.G = G
        self.a, self.G = self.make_unit_vol(self.G)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        M = torch.eye(dim)
        M[dim-1, :] = 0.5
        return M
    
    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M

    def forward(self, x_in):
        # [batch_size, dim]
        # Compute f(x), g(x)
        x = torch.div(x_in, self.a)
        y1 = torch.round(x)
        y2 = torch.round(x - 0.5) + 0.5
        codebooks = torch.stack((y1, y2), dim=1) #[batch_size, 2, dim]
        scores = torch.stack((torch.linalg.norm(x-y1, dim=1), torch.linalg.norm(x-y2, dim=1)), dim=1) #[batch_size, 2]
        idx = torch.argmin(scores, dim=1) # [batch_size]
        y = codebooks[torch.arange(codebooks.shape[0]), idx, :]
        y = torch.mul(y, self.a)
        return y
        
 
class SquareQuantizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(self.G, requires_grad=False)
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        return torch.eye(dim)

    def forward(self, x):
        # [batch_size, dim]
        return torch.round(x)
    
class BrickWallQuantizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim == 2
        self.dim = dim
        self.G = self.init_generator_matrix(dim)
        self.G = nn.Parameter(self.G, requires_grad=False)
        self.scale = np.sqrt(3) / 2
    
    def init_generator_matrix(self, dim):
        """
        Initialize lattice quantizer's generator matrix.
        """
        # return torch.eye(dim)
        return torch.tensor([[1, 0],
                              [0, np.sqrt(3) / 2]]).float()

    def forward(self, x):
        # [batch_size, dim]
        y = torch.clone(x)
        y_1_int = torch.round(y[:, 1] / self.scale)
        y[:, 1] = y_1_int * self.scale
        idx_odd = torch.where(y_1_int % 2 == 1, True, False)
        # print(idx_odd, y[:, 1])
        y[idx_odd, 0] = torch.round(y[idx_odd, 0] + 0.5) - 0.5
        # print(y[idx_odd])
        y[idx_odd.logical_not(), 0] = torch.round(y[idx_odd.logical_not(), 0])
        # print(y[1 - idx_odd, 0])
        return y