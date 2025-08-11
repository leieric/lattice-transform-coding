import torch.nn as nn
import torch
from LTC.entropy_models import EntropyBottleneck, EntropyBottleneckLattice, EntropyBottleneckLatticeFlow
from typing import cast
import numpy as np
from compressai.models.utils import conv, deconv
from LTC.quantizers import get_lattice, get_generator_matrix

def get_model(args):
    if args.model_name == 'NTC':
        return NTC(args.d, args.dy, args.d_hidden)
    elif args.model_name == 'LTC':
        return LTC(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LTCDither':
        return LTCDither(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'BlockLTC':
        return BlockLTC(args.n, args.d, dy=args.dy, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'BlockLTCZeroMean':
        return BlockLTCZeroMean(args.n, args.d, dy=args.dy, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'ECLQ':
        return ECLQ(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method, scale=args.lam)
    else:
        raise Exception("invalid model_name")

class NTC(nn.Module):
    def __init__(self, d, d_quant=2, d_hidden=100):
        super().__init__()
        activation = nn.Softplus()
        self.g_a = nn.Sequential(nn.Linear(d,d_hidden), 
                                 activation, 
                                 nn.Linear(d_hidden,d_hidden), 
                                 activation, 
                                 nn.Linear(d_hidden, d_quant))
        self.g_s = nn.Sequential(nn.Linear(d_quant,d_hidden), 
                                 activation, 
                                 nn.Linear(d_hidden,d_hidden), 
                                 activation, 
                                 nn.Linear(d_hidden,d))
        self.entropy_bottleneck = EntropyBottleneck(channels=d_quant)

    def forward(self, x):
        # x : [B, d]
        y= self.g_a(x) # [B, d]
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        return x_hat, y_likelihoods
    
    def eval(self, x, return_y=False, N=2048):
        with torch.no_grad():
            y = self.g_a(x)
            y_hat, y_lik = self.entropy_bottleneck(y, training=False)
            x_hat = self.g_s(y_hat)
        if return_y:
            return x_hat, y_lik, y, y_hat
        return x_hat, y_lik

    def aux_loss(self):
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(torch.Tensor, loss)

    
class LTC(nn.Module):
    def __init__(self, d, dy, d_hidden=100, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__()
        # d_hidden = 100

        self.d = d # dimension of quantizer
        self.dy = dy
        self.g_a = get_transform(tname, d, d_hidden, dy)
        self.g_s = get_transform(tname, dy, d_hidden, d)
        self.quantizer = get_lattice(lattice, dy)
        # self.entropy_bottleneck = EntropyBottleneckLattice(channels=d)
        self.entropy_bottleneck = get_entropy_bottleneck(eb_name, dy)

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled", "fixed"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=dy)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=dy, scramble=True)

        # self.voronoi_volume = torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T))).item()

    def _voronoi_volume(self):
        return torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T)))

    def _sample_from_voronoi(self, device, N=2048):
        # returns [N, d] samples drawn from Voronoi region of quantizer
        if self.MC_method == "standard":
            u = torch.rand((N, self.dy), device=device)
        elif self.MC_method == "antithetic":
            N = N // 2
            u = torch.rand((N, self.dy), device=device)
            u = torch.cat((u, -u), dim=0)
        elif self.MC_method.startswith("sobol"):
            u = self.sobol_eng.draw(N).to(device)
        else:
            raise Exception("MC method invalid")
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        return u2
    
    def _quantize(self, y, training=True):
        # Use STE no matter what
        y_q = self.quantizer(y)
        y_hat = y + (y_q - y).detach()
        return y_hat
    
    def _compute_likelihoods(self, y_hat):
        u2 = self._sample_from_voronoi(device=y_hat.device, N=self.N)
        lik = self._voronoi_volume()*self.entropy_bottleneck._likelihood(y_hat, u2)
        return lik

    def forward(self, x):
        # x : [B, d]
        y = self.g_a(x) # [B, d]
        y_hat = self._quantize(y, training=True)
        lik = self._compute_likelihoods(y_hat)
        x_hat = self.g_s(y_hat)
        return x_hat, lik
    
    def eval(self, x, return_y=False, N=2048):
        # get reconstructions.
        with torch.no_grad():
            y = self.g_a(x) # [B, d]
            y_hat = self._quantize(y, training=False)
            lik = self._compute_likelihoods(y_hat)
            x_hat = self.g_s(y_hat)
        if return_y:
            return x_hat, lik, y, y_hat
        return x_hat, lik#, y, y_hat

    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)

class LTCDither(LTC):
    def __init__(self, d, dy, d_hidden=100, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__(d, dy, d_hidden, lattice, tname, eb_name, N, MC_method)

    def _quantize(self, y, training=True):
        if training:
            # add dither noise
            u = torch.rand(y.shape, device=y.device) @ self.quantizer.G.to(y.device)
            u = u - self.quantizer(u)
            y_hat = y + u
        else:
            # use hard quantization
            u = torch.rand(y.shape, device=y.device) @ self.quantizer.G.to(y.device)
            u = u - self.quantizer(u)
            y_hat = self.quantizer(y - u) + u
        return y_hat
    
    def _compute_likelihoods(self, y_hat):
        u2 = self._sample_from_voronoi(device=y_hat.device, N=self.N)
        lik = self._voronoi_volume()*self.entropy_bottleneck._likelihood(y_hat, u2)
        return lik
    


    
class BlockLTC(nn.Module):
    def __init__(self, n, d, dy, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__()
        d_hidden = 100

        self.d = d # dimension of source input
        self.dy = dy # dimension after sample-wise transforms
        self.n = n
        self.g_a = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, dy)
        )

        self.g_s = nn.Sequential(
            nn.Linear(dy, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d)
        )
        self.g_a_c = nn.ModuleList([get_transform(tname, n, d_hidden, n) for _ in range(dy)]) # use different compander for each dimension in dy
        self.g_s_c = nn.ModuleList([get_transform(tname, n, d_hidden, n) for _ in range(dy)])
        # self.g_a_c = get_transform(tname, n, d_hidden, n)
        # self.g_s_c = get_transform(tname, n, d_hidden, n)
        self.quantizer = get_lattice(lattice, n)
        # self.entropy_bottleneck = get_entropy_bottleneck(eb_name, n)
        self.entropy_bottleneck = nn.ModuleList([get_entropy_bottleneck(eb_name, n) for _ in range(dy)]) # use different density model for each dimensions in dy

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n, scramble=True)
        self.voronoi_volume = torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T))).item()

    def _sample_from_voronoi(self, device, N=2048):
        # returns [N, d] samples drawn from Voronoi region of quantizer
        if self.MC_method == "standard":
            u = torch.rand((N, self.n), device=device)
        elif self.MC_method == "antithetic":
            N = N // 2
            u = torch.rand((N, self.d), device=device)
            u = torch.cat((u, -u), dim=0)
        elif self.MC_method.startswith("sobol"):
            u = self.sobol_eng.draw(N).to(device)
        else:
            raise Exception("MC method invalid")
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        return u2

    def forward(self, x):
        # x : [B, n, d]
        # Get latent y
        # print(x.device, self.g_a.device)
        y1 = self.g_a(x) # [B, n, dy]
        u2 = self._sample_from_voronoi(device=x.device)
        u = self._sample_from_voronoi(device=x.device, N=y1.shape[0])
        y_hat1 = []
        lik = []
        for i in range(self.dy):
            y_i = self.g_a_c[i](y1[:, :, i]) # [B, n]
            y_q_i = self.quantizer(y_i)
            # print(f"y_q_i={y_q_i}", f"y1={y1}")
            y_hat_i = y_i + (y_q_i - y_i).detach()
            y_tilde_i = y_i + u
            lik_i = self.entropy_bottleneck[i]._likelihood(y_tilde_i, u2)
            # lik_i = self.entropy_bottleneck._log_likelihood(y_hat_i, u2)
            y_hat1_i = self.g_s_c[i](y_hat_i)
            y_hat1.append(y_hat1_i)
            lik.append(lik_i)
        y_hat1 = torch.stack(y_hat1, dim=2) # [B, n, dy]
        lik = torch.stack(lik, dim=1) # [B, dy]
        x_hat = self.g_s(y_hat1) # [B, n, d]
        return x_hat, lik

    def eval(self, x, N=2048):
        with torch.no_grad():
            y1 = self.g_a(x)
            y_hat1 = []
            lik = []
            u2 = self._sample_from_voronoi(device=x.device, N=N)
            for i in range(self.dy):
                y_i = self.g_a_c[i](y1[:, :, i]) # [B, n]
                y_hat_i = self.quantizer(y_i)
                lik_i = self.entropy_bottleneck[i]._likelihood(y_hat_i, u2)
                y_hat1_i = self.g_s_c[i](y_hat_i)
                y_hat1.append(y_hat1_i)
                lik.append(lik_i)
            y_hat1 = torch.stack(y_hat1, dim=2) # [B, n, dy]
            lik = torch.stack(lik, dim=1) # [B, dy]
            x_hat = self.g_s(y_hat1)
            return x_hat, lik
    
    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)
    
class BlockLTCZeroMean(nn.Module):
    def __init__(self, n, d, dy=1, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__()
        d_hidden = 100

        self.d = d # dimension of source input
        self.dy = dy # dimension after sample-wise transforms
        self.n = n
        self.y_mean = nn.Parameter(torch.randn(n, dy), requires_grad=False)
        self.g_a = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, dy)
        )

        self.g_s = nn.Sequential(
            nn.Linear(dy, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d)
        )
        self.g_a_c = nn.ModuleList([get_transform(tname, n, d_hidden, n) for _ in range(dy)]) # use different compander for each dimension in dy
        self.g_s_c = nn.ModuleList([get_transform(tname, n, d_hidden, n) for _ in range(dy)])
        # self.g_a_c = get_transform(tname, n, d_hidden, n)
        # self.g_s_c = get_transform(tname, n, d_hidden, n)
        self.quantizer = get_lattice(lattice, n)
        self.entropy_bottleneck = nn.ModuleList([get_entropy_bottleneck(eb_name, n) for _ in range(dy)]) # use different density model for each dimensions in dy
        # self.entropy_bottleneck = get_entropy_bottleneck(eb_name, n)

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n, scramble=True)
        self.voronoi_volume = torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T))).item()

    def _sample_from_voronoi(self, device, N=2048):
        # returns [N, d] samples drawn from Voronoi region of quantizer
        if self.MC_method == "standard":
            u = torch.rand((N, self.n), device=device)
        elif self.MC_method == "antithetic":
            N = N // 2
            u = torch.rand((N, self.d), device=device)
            u = torch.cat((u, -u), dim=0)
        elif self.MC_method.startswith("sobol"):
            u = self.sobol_eng.draw(N).to(device)
        else:
            raise Exception("MC method invalid")
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        return u2

    def forward(self, x):
        # x : [B, n, d]
        # Get latent y
        # print(x.device, self.g_a.device)
        y1 = self.g_a(x) # [B, n, dy]
        # print(f"x={x}", f"y1={y1}, g_a={self.g_a[0].weight}")
        y_mean = torch.mean(y1, dim=0) #[n, dy]
        self.y_mean.data = 0.05*self.y_mean.data + 0.95*y_mean
        # # print(f'y1:{y1.shape}')
        
        y1 = y1 - self.y_mean[None, :, :] # [B, n, dy]
        # print(f"y_min={y1.min()}, y_max={y1.max()}")
        u2 = self._sample_from_voronoi(device=x.device, N=self.N)
        y_hat1 = []
        lik = []
        for i in range(self.dy):
            y_i = self.g_a_c[i](y1[:, :, i]) # [B, n]
            # print(f"y_i_min={y_i.min()}, y_i_max={y_i.max()}")
            y_q_i = self.quantizer(y_i)
            # print(f"y_q_i={y_q_i}", f"y1={y1}")
            y_hat_i = y_i + (y_q_i - y_i).detach()
            lik_i = self.entropy_bottleneck[i]._likelihood(y_hat_i, u2)
            # lik_i = self.entropy_bottleneck[i]._log_likelihood(y_hat_i, u2)
            y_hat1_i = self.g_s_c[i](y_hat_i) + self.y_mean[None, :, i]
            y_hat1.append(y_hat1_i)
            lik.append(lik_i)
        y_hat1 = torch.stack(y_hat1, dim=2) # [B, n, dy]
        lik = torch.stack(lik, dim=1) # [B, dy]
        x_hat = self.g_s(y_hat1) # [B, n, d]
        return x_hat, lik

    def eval(self, x, N=2048):
        with torch.no_grad():
            y1 = self.g_a(x)
            y1 = y1 - self.y_mean[None, :, :]
            y_hat1 = []
            lik = []
            u2 = self._sample_from_voronoi(device=x.device, N=N)
            for i in range(self.dy):
                y_i = self.g_a_c[i](y1[:, :, i]) # [B, n]
                y_hat_i = self.quantizer(y_i)
                lik_i = self.entropy_bottleneck[i]._likelihood(y_hat_i, u2)
                y_hat1_i = self.g_s_c[i](y_hat_i) + self.y_mean[None, :, i]
                y_hat1.append(y_hat1_i)
                lik.append(lik_i)
            y_hat1 = torch.stack(y_hat1, dim=2) # [B, n, dy]
            lik = torch.stack(lik, dim=1) # [B, dy]
            x_hat = self.g_s(y_hat1)
            return x_hat, lik

    
    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)
    
class ECLQ(nn.Module):
    def __init__(self, d, dy, d_hidden=100, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard", scale=1):
        super().__init__()
        # d_hidden = 100

        self.d = d # dimension of quantizer
        self.dy = dy
        self.g_a = get_transform(tname, d, d_hidden, dy)
        self.g_s = get_transform(tname, dy, d_hidden, d)
        self.quantizer = get_lattice(lattice, dy)
        self.scale = scale
        # print(self._voronoi_volume())
        self.quantizer.G *= self.scale
        # print(self._voronoi_volume())
        # self.entropy_bottleneck = EntropyBottleneckLattice(channels=d)
        self.entropy_bottleneck = get_entropy_bottleneck(eb_name, dy)

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled", "fixed"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=dy)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=dy, scramble=True)

        # self.voronoi_volume = torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T))).item()

    def _voronoi_volume(self):
        return torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T)))

    def _sample_from_voronoi(self, device, N=2048):
        # returns [N, d] samples drawn from Voronoi region of quantizer
        if self.MC_method == "standard":
            u = torch.rand((N, self.dy), device=device)
        elif self.MC_method == "antithetic":
            N = N // 2
            u = torch.rand((N, self.dy), device=device)
            u = torch.cat((u, -u), dim=0)
        elif self.MC_method.startswith("sobol"):
            u = self.sobol_eng.draw(N).to(device)
        else:
            raise Exception("MC method invalid")
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        return u2
    
    def _quantize(self, y, training=True):
        # Use STE no matter what
        y_q = self.quantizer(y)
        y_hat = y + (y_q - y).detach()
        return y_hat
    
    def _compute_likelihoods(self, y_hat):
        u2 = self._sample_from_voronoi(device=y_hat.device, N=self.N)
        lik = self._voronoi_volume()*self.entropy_bottleneck._likelihood(y_hat, u2)
        # lik = self.entropy_bottleneck._likelihood(y_hat, u2)
        return lik

    def forward(self, x):
        # x : [B, d]
        y = x.clone() / self.scale
        y_hat = self._quantize(y, training=True)
        lik = self._compute_likelihoods(y_hat)
        x_hat = y_hat * self.scale
        return x_hat, lik
    
    def eval(self, x, return_y=False, N=2048):
        # get reconstructions.
        with torch.no_grad():
            y = x.clone() / self.scale
            y_hat = self._quantize(y, training=False)
            lik = self._compute_likelihoods(y_hat)
            x_hat = y_hat * self.scale
        if return_y:
            return x_hat, lik, y, y_hat
        return x_hat, lik#, y, y_hat
    
class ECLQDither(ECLQ):
    def __init__(self, d, dy, d_hidden=100, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard", scale=1):
        super().__init__(d, dy, d_hidden, lattice, tname, eb_name, N, MC_method, scale)

    def _quantize(self, y, training=True):
        if training:
            # add dither noise
            u = torch.rand(y.shape, device=y.device) @ self.quantizer.G.to(y.device)
            u = u - self.quantizer(u)
            y_hat = y + u
        else:
            # use hard quantization
            u = torch.rand(y.shape, device=y.device) @ self.quantizer.G.to(y.device)
            u = u - self.quantizer(u)
            y_hat = self.quantizer(y - u) + u
        return y_hat
    
    def _compute_likelihoods(self, y_hat):
        u2 = self._sample_from_voronoi(device=y_hat.device, N=self.N)
        lik = self.entropy_bottleneck._likelihood(y_hat, u2)
        return lik


def get_transform(name, d_in=2, d_hidden=100, d_out=2, activation=nn.LeakyReLU()):
    if name == 'MLPNoBias':
        # return nn.Sequential(nn.Linear(d_in,d_hidden, bias=False), 
        #                     activation, 
        #                     nn.Linear(d_hidden,d_out, bias=False)
        #                     )
        activation = nn.Softplus()
        return nn.Sequential(nn.Linear(d_in,d_hidden, bias=False), 
                            activation, 
                            nn.Linear(d_hidden,d_hidden, bias=False), 
                            activation, 
                            nn.Linear(d_hidden, d_out, bias=False)
                            )
    elif name == 'MLP':
        # return nn.Sequential(nn.Linear(d_in,d_hidden), 
        #                     activation, 
        #                     nn.Linear(d_hidden,d_out)
        #                     )
        activation = nn.Softplus()
        return nn.Sequential(nn.Linear(d_in,d_hidden), 
                            activation, 
                            nn.Linear(d_hidden,d_hidden), 
                            activation, 
                            # nn.Linear(d_hidden,d_hidden), 
                            # activation, 
                            nn.Linear(d_hidden, d_out)
                            )
    elif name == 'MLP2':
        # return nn.Sequential(nn.Linear(d_in,d_hidden), 
        #                     activation, 
        #                     nn.Linear(d_hidden,d_out)
        #                     )
        activation = nn.CELU()
        return nn.Sequential(nn.Linear(d_in,d_hidden), 
                            activation, 
                            nn.Linear(d_hidden,d_out), 
                            )
    elif name == 'MLP2NoBias':
        # return nn.Sequential(nn.Linear(d_in,d_hidden), 
        #                     activation, 
        #                     nn.Linear(d_hidden,d_out)
        #                     )
        activation = nn.CELU()
        return nn.Sequential(nn.Linear(d_in,d_hidden, bias=False), 
                            activation, 
                            nn.Linear(d_hidden,d_hidden, bias=False), 
                            activation, 
                            nn.Linear(d_hidden,d_out, bias=False), 
                            )
    elif name == 'MLP3':
        # return nn.Sequential(nn.Linear(d_in,d_hidden), 
        #                     activation, 
        #                     nn.Linear(d_hidden,d_out)
        #                     )
        activation = nn.CELU()
        return nn.Sequential(nn.Linear(d_in,1000, bias=True), 
                            activation, 
                            nn.Linear(1000,d_out, bias=True), 
                            )
    elif name == 'MLP3NoBias':
        # return nn.Sequential(nn.Linear(d_in,d_hidden), 
        #                     activation, 
        #                     nn.Linear(d_hidden,d_out)
        #                     )
        activation = nn.CELU()
        return nn.Sequential(nn.Linear(d_in,1000, bias=False), 
                            activation, 
                            nn.Linear(1000,d_out, bias=False), 
                            )
    elif name == 'LinearNoBias':
        # return nn.Sequential(nn.Linear(d_in,d_out, bias=False))
        return nn.Sequential(nn.Linear(d_in,d_hidden, bias=False), 
                            #  nn.Identity(),
                            nn.Linear(d_hidden,d_out, bias=False)
                            )
    elif name == 'Linear':
        return nn.Sequential(nn.Linear(d_in,d_hidden), 
                            nn.Linear(d_hidden,d_out)
                            )
    else:
        return Exception("Invalid transform name")
    
    
def get_entropy_bottleneck(name, d):
    if name == 'FactorizedPrior':
        return EntropyBottleneckLattice(channels=d)
    elif 'Flow' in name:
        name = name.split('_')[1]
        return EntropyBottleneckLatticeFlow(channels=d, flow_name=name)
    else:
        raise Exception("Invalid entropy bottleneck name")