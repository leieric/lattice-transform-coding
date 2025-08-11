import torch.nn as nn
import torch
from entropy_models import EntropyBottleneck, EntropyBottleneckLattice, EntropyBottleneckLatticeFlow
from typing import cast
import numpy as np
from compressai.models.utils import conv, deconv
from quantizers import get_lattice, get_generator_matrix

class SquareCompanderNTC(nn.Module):
    def __init__(self, d, d_quant=2, d_hidden=100):
        super().__init__()
        activation = nn.Softplus()
        # d_hidden = 100
        self.g_a = nn.Sequential(nn.Linear(d,d_hidden), 
                                 activation, 
                                 nn.Linear(d_hidden,d_hidden), 
                                 activation, 
                                #  nn.Linear(d_hidden,d_hidden), 
                                #  activation, 
                                 nn.Linear(d_hidden, d_quant))
        self.g_s = nn.Sequential(nn.Linear(d_quant,d_hidden), 
                                 activation, 
                                #  nn.Linear(d_hidden,d_hidden), 
                                #  activation, 
                                 nn.Linear(d_hidden,d_hidden), 
                                 activation, 
                                 nn.Linear(d_hidden,d))
        self.entropy_bottleneck = EntropyBottleneck(channels=d_quant)

    def forward(self, x):
        # x : [B, d]
        y= self.g_a(x) # [B, d]
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)
        # print(y.shape, y_hat.shape, y_likelihoods.shape, x_hat.shape)
        return x_hat, y_likelihoods
    
    def eval(self, x, return_y=False, N=2048):
        with torch.no_grad():
            y = self.g_a(x)
            # y_hat = torch.round(y)
            y_hat, y_lik = self.entropy_bottleneck(y, training=False)
            # _, y_lik = self.entropy_bottleneck(y, training=False)
            # lik, _, _ = self.entropy_bottleneck._likelihood(torch.round(y_hat.permute(1, 0)), stop_gradient=True)
            # lik, _, _ = self.entropy_bottleneck._likelihood(torch.round(y - (torch.rand_like(y)-0.5)), stop_gradient=True)
            x_hat = self.g_s(y_hat)
        if return_y:
            return x_hat, y_lik, y, y_hat
        return x_hat, y_lik

    def aux_loss(self):
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(torch.Tensor, loss)
    
class SquareCompanderNTC_Conv(SquareCompanderNTC):
    def __init__(self, d, d_quant=2):
        super().__init__(d, d_quant)
        activation = nn.Softplus()
        d_hidden = 64
        self.g_a = nn.Sequential(
            conv(d, d_hidden),
            activation,
            conv(d_hidden, d_hidden),
            activation,
            nn.Conv2d(d_hidden, d_quant, kernel_size=1)
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(d_quant, d_hidden, kernel_size=1),
            activation,
            deconv(d_hidden, d_hidden),
            activation,
            deconv(d_hidden, d)
        )

    
def get_compander(args):
    if args.model_name == 'SquareCompanderNTC':
        return SquareCompanderNTC(args.d, args.dy, args.d_hidden)
    elif args.model_name == 'SquareCompanderNTC_Conv':
        return SquareCompanderNTC_Conv(args.d, args.dy)
    elif args.model_name == 'LatticeCompander':
        return LatticeCompander(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompander_Conv':
        return LatticeCompander_Conv(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderDither':
        return LatticeCompanderDither(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderDither_Conv':
        return LatticeCompanderDither_Conv(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderBRE':
        return LatticeCompanderBRE(args.n, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderBlock':
        return LatticeCompanderBlock(args.n, args.d, dy=args.dy, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderBlockZeroMean':
        return LatticeCompanderBlockZeroMean(args.n, args.d, dy=args.dy, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderBlockZeroMeanBananaSplit':
        return LatticeCompanderBlockZeroMeanBananaSplit(args.n, args.d, dy=args.dy, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderBlock2':
        return LatticeCompanderBlock2(args.n, args.d, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'LatticeCompanderBlock2BananaSplit':
        return LatticeCompanderBlock2BananaSplit(args.n, args.d, dy=args.dy, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method)
    elif args.model_name == 'ECLQ':
        return ECLQ(args.d, args.dy, args.d_hidden, lattice=args.lattice_name, tname=args.transform_name, eb_name=args.eb_name, N=args.N_integral, MC_method=args.MC_method, scale=args.lam)
    else:
        raise Exception("invalid model_name")
    
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
    
class LatticeCompander(nn.Module):
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
    
    # def est_true_rate(self, y_hat):


    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)

class LatticeCompanderDither(LatticeCompander):
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
    

    # def forward(self, x):
    #     # x : [B, d]
    #     # Get latent y
    #     y = self.g_a(x) # [B, d]

    #     # Sample noise
    #     u = torch.rand(y.shape, device=x.device) @ self.quantizer.G.to(x.device)
    #     u = u - self.quantizer(u)

    #     # Compute quantized y
    #     y_q = self.quantizer(y - u) + u
    #     # y_q = self.quantizer(y)
    #     y_hat = y + (y_q - y).detach()

    #     # Compute likelihoods term
    #     u2 = self._sample_from_voronoi(device=x.device, N=self.N)
    #     # lik = self.entropy_bottleneck(y+u, u2) # [B, d] # dithering approach
    #     lik = self._voronoi_volume() * self.entropy_bottleneck._likelihood(y+u, u2)
        
    #     # Compute reconstructions
    #     x_hat = self.g_s(y_hat)
        
    #     return x_hat, lik
    
    # def eval(self, x, return_y=False, N=2048):
    #     # get reconstructions.
    #     with torch.no_grad():
    #         y = self.g_a(x) # [B, d]
    #         u = torch.rand(y.shape, device=x.device) @ self.quantizer.G.to(x.device)
    #         u = u - self.quantizer(u)
    #         y_hat = self.quantizer(y - u) + u
    #         # y_hat = self.quantizer(y)
    #         x_hat = self.g_s(y_hat)

    #         # Rate term, performing dithered quantization.
    #         u2 = self._sample_from_voronoi(device=x.device, N=N)

    #         # lik = self.entropy_bottleneck._likelihood(y_hat, u2) # [B, d]
    #         lik = self._voronoi_volume() * self.entropy_bottleneck._likelihood(y_hat, u2) # [B, d]
    #         # lik = self.voronoi_volume*self.entropy_bottleneck._likelihood(self.quantizer(y - u) + u, u2) # [B, d]
    #     if return_y:
    #         return x_hat, lik, y, y_hat
    #     return x_hat, lik#, y, y_hat

class LatticeCompander_Conv(LatticeCompander):
    def __init__(self, d, dy, d_hidden=100, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__(d, dy, d_hidden, lattice, tname, eb_name, N, MC_method)
        activation = nn.Softplus()
        d_hidden = 64
        self.g_a = nn.Sequential(
            conv(d, d_hidden),
            activation,
            conv(d_hidden, d_hidden),
            activation,
            nn.Conv2d(d_hidden, dy, kernel_size=1)
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(dy, d_hidden, kernel_size=1),
            activation,
            deconv(d_hidden, d_hidden),
            activation,
            deconv(d_hidden, d)
        )

    def _flatten(self, y):
        # y: [B, dy, h, w]
        y = y.permute(0, 2, 3, 1) #[B, h, w, dy]
        y_shape = y.size()
        y = y.reshape(-1, self.dy) # [B*h*w, dy]
        return y, y_shape
    
    def _unflatten(self, y_hat, y_shape):
        y_hat = y_hat.reshape(y_shape) # [B, h, w, dy]
        y_hat = y_hat.permute(0, 3, 1, 2)
        return y_hat
    
    def forward(self, x):
        # x : [B, d]
        y = self.g_a(x) # [B, d]
        y, y_shape = self._flatten(y)
        y_hat = self._quantize(y, training=True)
        lik = self._compute_likelihoods(y_hat)
        y_hat = self._unflatten(y_hat, y_shape)
        x_hat = self.g_s(y_hat)
        return x_hat, lik
    
    def eval(self, x, return_y=False, N=2048):
        # get reconstructions.
        with torch.no_grad():
            y = self.g_a(x) # [B, d]
            y, y_shape = self._flatten(y)
            y_hat = self._quantize(y, training=False)
            lik = self._compute_likelihoods(y_hat)
            y_hat = self._unflatten(y_hat, y_shape)
            x_hat = self.g_s(y_hat)
        if return_y:
            return x_hat, lik, y, y_hat
        return x_hat, lik#, y, y_hat
    
class LatticeCompanderDither_Conv(LatticeCompanderDither):
    def __init__(self, d, dy, d_hidden=100, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__(d, dy, d_hidden, lattice, tname, eb_name, N, MC_method)
        activation = nn.Softplus()
        d_hidden = 64
        self.g_a = nn.Sequential(
            conv(d, d_hidden),
            activation,
            conv(d_hidden, d_hidden),
            activation,
            nn.Conv2d(d_hidden, dy, kernel_size=1)
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(dy, d_hidden, kernel_size=1),
            activation,
            deconv(d_hidden, d_hidden),
            activation,
            deconv(d_hidden, d)
        )

    def _flatten(self, y):
        # y: [B, dy, h, w]
        y = y.permute(0, 2, 3, 1) #[B, h, w, dy]
        y_shape = y.size()
        y = y.reshape(-1, self.dy) # [B*h*w, dy]
        return y, y_shape
    
    def _unflatten(self, y_hat, y_shape):
        y_hat = y_hat.reshape(y_shape) # [B, h, w, dy]
        y_hat = y_hat.permute(0, 3, 1, 2)
        return y_hat
    
    def forward(self, x):
        # x : [B, d]
        y = self.g_a(x) # [B, d]
        y, y_shape = self._flatten(y)
        y_hat = self._quantize(y, training=True)
        lik = self._compute_likelihoods(y_hat)
        y_hat = self._unflatten(y_hat, y_shape)
        x_hat = self.g_s(y_hat)
        return x_hat, lik
    
    def eval(self, x, return_y=False, N=2048):
        # get reconstructions.
        with torch.no_grad():
            y = self.g_a(x) # [B, d]
            y, y_shape = self._flatten(y)
            y_hat = self._quantize(y, training=False)
            lik = self._compute_likelihoods(y_hat)
            y_hat = self._unflatten(y_hat, y_shape)
            x_hat = self.g_s(y_hat)
        if return_y:
            return x_hat, lik, y, y_hat
        return x_hat, lik#, y, y_hat
    
    
class LatticeCompanderBRE(nn.Module):
    def __init__(self, d, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__()
        d_hidden = 100

        self.d = d # dimension of quantizer
        self.g_a = get_transform(tname, d, d_hidden, d)
        self.g_s = get_transform(tname, d, d_hidden, d)
        # self.quantizer = get_lattice(lattice, d)
        if d == 2:
            G = get_generator_matrix("Hexagonal")
        elif d == 4:
            G = get_generator_matrix("DnDual", d)
        elif d == 8:
            G = get_generator_matrix('E8')
        elif d == 24:
            G = get_generator_matrix('Leech')
        self.G = nn.Parameter(G)
        self.entropy_bottleneck = get_entropy_bottleneck(eb_name, d)

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=d)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        self.voronoi_volume = torch.sqrt(torch.linalg.det((self.G @ self.G.T))).item()

    def make_unit_vol(self, M):
        n = M.shape[0]
        vol = torch.sqrt(torch.linalg.det(M @ M.T))
        a = 1 / (vol ** (1/n))
        M = a * M
        return a, M
    
    def _voronoi_volume(self):
        return torch.sqrt(torch.linalg.det((self.G @ self.G.T)))
    
    def _quantize(self, x, training=False):
        """
        Babai rounding estimate.
        """
        v = x @ torch.linalg.inv(self.G)
        # v_q = torch.round(v)
        # v_q = v + (v_q - v).detach()
        v_q = v + torch.rand_like(v) - 0.5
        x_hat = v_q @ self.G
        return x_hat


    def forward(self, x):
        # x : [B, d]
        # Get latent y
        # with torch.no_grad():
        #     _, self.G.data = self.make_unit_vol(self.G.data)
        # print(self.G.norm(2))
        y = self.g_a(x) # [B, d]
        y_hat = self._quantize(y)
        u2 = torch.rand((self.N, self.d), device=x.device) @ self.G
        u2 = u2 - self._quantize(u2)

        lik = self._voronoi_volume()*self.entropy_bottleneck._likelihood(y_hat, u2)
        x_hat = self.g_s(y_hat)
        return x_hat, lik
    
    def eval(self, x, return_y=False, N=2048):
        # get reconstructions.
        with torch.no_grad():
            y = self.g_a(x) # [B, d]
            y_hat = self._quantize(y)
            u2 = torch.rand((N, self.d), device=x.device) @ self.G
            u2 = u2 - self._quantize(u2)
            x_hat = self.g_s(y_hat)
            lik = self._voronoi_volume()*self.entropy_bottleneck._likelihood(y_hat, u2) # [B, d]
        if return_y:
            return x_hat, lik, y, y_hat
        return x_hat, lik

    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)

    
class LatticeCompanderBlock(nn.Module):
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
    
class LatticeCompanderBlockZeroMean(nn.Module):
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
        # # y = torch.stack(y, dim=2) # [B, n, dy]
        # # print(f'y:{y.shape}')
        # y_q = self.quantizer(y)
        # y_hat = y + (y_q - y).detach()
        # u2 = self._sample_from_voronoi(device=x.device)
        # lik = self.entropy_bottleneck._likelihood(y_hat, u2) #[B]
        # # print(f'lik:{lik.shape}')
        # y_hat1 = self.g_s_c(y_hat) + self.y_mean[None, :] # [B, n]
        # # # print(f'y_hat1:{y_hat1.shape}')
        # x_hat = self.g_s(y_hat1.unsqueeze(2))
        # # print(f'x_hat:{x_hat.shape}')
        # return x_hat, lik
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
    # def eval(self, x):
    #     with torch.no_grad():
    #         y1 = self.g_a(x).squeeze() # [B, n, 1]
    #         y = self.g_a_c(y1 - self.y_mean[None, :]) # [B, n]
    #         y_hat = self.quantizer(y)
    #         y_hat1 = self.g_s_c(y_hat) + self.y_mean[None, :] # [B, n]
    #         x_hat = self.g_s(y_hat1.unsqueeze(2))
    #         u2 = self._sample_from_voronoi(device=x.device)
    #         lik = self.voronoi_volume*self.entropy_bottleneck._likelihood(y_hat, u2) # [B, d]
    #     return x_hat, lik
    
    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)
    
class LatticeCompanderBlockZeroMeanBananaSplit(nn.Module):
    def __init__(self, n, d, dy=1, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__()
        d_hidden = 100

        if n not in [12, 4]:
            raise Exception("Invalid blocklength")
        n = 2*n

        self.d = d # dimension of source input
        self.dy = dy # dimension after sample-wise transforms
        self.n = n
        self.y_mean = nn.Parameter(torch.randn(n // 2, dy), requires_grad=False)
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
        # self.g_a_c = nn.ModuleList([get_transform(tname, n, d_hidden, n) for _ in range(dy)]) # use different compander for each dimension in dy
        # self.g_s_c = nn.ModuleList([get_transform(tname, n, d_hidden, n) for _ in range(dy)])
        self.g_a_c = get_transform(tname, n, d_hidden, n)
        self.g_s_c = get_transform(tname, n, d_hidden, n)
        
        self.quantizer = get_lattice(lattice, n)
        self.entropy_bottleneck = get_entropy_bottleneck(eb_name, n)

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n, scramble=True)
        self.voronoi_volume = torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T))).item()

    def _sample_from_voronoi(self, device):
        # returns [N, d] samples drawn from Voronoi region of quantizer
        if self.MC_method == "standard":
            u = torch.rand((self.N, self.n), device=device)
        elif self.MC_method == "antithetic":
            N = self.N // 2
            u = torch.rand((N, self.d), device=device)
            u = torch.cat((u, -u), dim=0)
        elif self.MC_method.startswith("sobol"):
            u = self.sobol_eng.draw(self.N).to(device)
        else:
            raise Exception("MC method invalid")
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        return u2

    def forward(self, x):
        # x : [B, n, d]
        B = x.shape[0]
        y1 = self.g_a(x) # [B, n / 2, dy]
        y_mean = torch.mean(y1, dim=0) #[n, dy]
        self.y_mean.data = 0.05*self.y_mean.data + 0.95*y_mean
        
        y1 = y1 - self.y_mean[None, :, :] # [B, n, dy]
        y1 = y1.reshape(B, self.n)
        y1 = self.g_a_c(y1)
        u2 = self._sample_from_voronoi(device=x.device)
        y_q = self.quantizer(y1)
        y_hat = y1 + (y_q - y1).detach()
        # lik = self.entropy_bottleneck._likelihood(y_hat, u2) #[B]
        lik = self.entropy_bottleneck._log_likelihood(y_hat, u2)
        y_hat1 = self.g_s_c(y_hat).reshape(B, self.n // self.dy, self.dy) + self.y_mean[None, :, :] # [B, n/2, dy]
        x_hat = self.g_s(y_hat1)
        return x_hat, lik

    def eval(self, x):
        B = x.shape[0]
        with torch.no_grad():
            y1 = self.g_a(x)
            y1 = y1 - self.y_mean[None, :, :]
            y1 = self.g_a_c(y1)
            y_hat = self.quantizer(y1)
            u2 = self._sample_from_voronoi(device=x.device)
            lik = self.entropy_bottleneck._likelihood(y_hat, u2)
            y_hat1 = self.g_s_c(y_hat).reshape(B, self.n // self.dy, self.dy) + self.y_mean[None, :, :] # [B, n/2, dy]
            x_hat = self.g_s(y_hat1)
            return x_hat, lik
    
    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)
    
class LatticeCompanderBlock2BananaSplit(nn.Module):
    def __init__(self, n, d, dy=1, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__()
        d_hidden = 500

        if n not in [12, 4]:
            raise Exception("Invalid blocklength")
        n = 2*n

        self.d = d # dimension of source input
        self.dy = dy # dimension after sample-wise transforms
        self.n = n
        self.g_a = nn.Sequential(
            nn.Linear(n, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, n)
        )

        self.g_s = nn.Sequential(
            nn.Linear(n, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Linear(d_hidden, n)
        )
        
        self.quantizer = get_lattice(lattice, n)
        self.entropy_bottleneck = get_entropy_bottleneck(eb_name, n)

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=n, scramble=True)
        self.voronoi_volume = torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T))).item()

    def _sample_from_voronoi(self, device):
        # returns [N, d] samples drawn from Voronoi region of quantizer
        if self.MC_method == "standard":
            u = torch.rand((self.N, self.n), device=device)
        elif self.MC_method == "antithetic":
            N = self.N // 2
            u = torch.rand((N, self.d), device=device)
            u = torch.cat((u, -u), dim=0)
        elif self.MC_method.startswith("sobol"):
            u = self.sobol_eng.draw(self.N).to(device)
        else:
            raise Exception("MC method invalid")
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        return u2

    def forward(self, x):
        # x : [B, n, d]
        B = x.shape[0]
        x = x.reshape(B, self.n)
        y1 = self.g_a(x) # [B, n / 2, dy]
        u2 = self._sample_from_voronoi(device=x.device)
        y_q = self.quantizer(y1)
        y_hat = y1 + (y_q - y1).detach()
        lik = self.entropy_bottleneck._likelihood(y_hat, u2) #[B]
        # lik = self.entropy_bottleneck._log_likelihood(y_hat, u2)
        x_hat = self.g_s(y_hat).reshape(B, self.n // 2, self.dy)
        return x_hat, lik

    def eval(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.n)
        with torch.no_grad():
            y1 = self.g_a(x)
            y_hat = self.quantizer(y1)
            u2 = self._sample_from_voronoi(device=x.device)
            lik = self.entropy_bottleneck._likelihood(y_hat, u2)
            x_hat = self.g_s(y_hat).reshape(B, self.n // 2, self.dy)
            return x_hat, lik
    
    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)
    
class LatticeCompanderBlock2(nn.Module):
    def __init__(self, n, d, lattice='Hexagonal', tname='MLPNoBias', eb_name='FactorizedPrior', N=256, MC_method="standard"):
        super().__init__()
        d_hidden = 100

        self.d = d # dimension of quantizer
        self.n = n
        self.g_a = nn.Sequential(
            nn.Linear(n*d, n*2*d_hidden),
            nn.LeakyReLU(),
            nn.Linear(n*2*d_hidden, n*d_hidden),
            nn.LeakyReLU(),
            nn.Linear(n*d_hidden, n*d_hidden),
            nn.LeakyReLU(),
            nn.Linear(n*d_hidden, n)
        )

        self.g_s = nn.Sequential(
            nn.Linear(n, n*d_hidden),
            nn.LeakyReLU(),
            nn.Linear(n*d_hidden, n*d_hidden),
            nn.LeakyReLU(),
            nn.Linear(n*d_hidden, n*2*d_hidden),
            nn.LeakyReLU(),
            nn.Linear(n*2*d_hidden, n*d)
        )
        self.g_a_c = get_transform(tname, n, d_hidden, n)
        self.g_s_c = get_transform(tname, n, d_hidden, n)
        self.quantizer = get_lattice(lattice, n)
        self.entropy_bottleneck = get_entropy_bottleneck(eb_name, n)

        self.lattice_name = lattice
        self.transform_name = tname
        self.N = N
        self.MC_method = MC_method # can be ["standard", "antithetic", "sobol", "sobol_scrambled"]
        if self.MC_method == "sobol":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=d)
        elif self.MC_method == "sobol_scrambled":
            self.sobol_eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        self.voronoi_volume = torch.sqrt(torch.linalg.det((self.quantizer.G @ self.quantizer.G.T))).item()

    def _sample_from_voronoi(self, device):
        # returns [N, d] samples drawn from Voronoi region of quantizer
        if self.MC_method == "standard":
            u = torch.rand((self.N, self.n), device=device)
        elif self.MC_method == "antithetic":
            N = self.N // 2
            u = torch.rand((N, self.d), device=device)
            u = torch.cat((u, -u), dim=0)
        elif self.MC_method.startswith("sobol"):
            u = self.sobol_eng.draw(self.N).to(device)
        else:
            raise Exception("MC method invalid")
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        return u2

    def forward(self, x):
        # x : [B, n, d]
        y = self.g_a(x.reshape(-1, self.n*self.d)).reshape(-1, self.n)
        y_q = self.quantizer(y)
        y_hat = y + (y_q - y).detach()
        u2 = self._sample_from_voronoi(device=x.device)
        lik = self.entropy_bottleneck._likelihood(y_hat, u2)
        x_hat = self.g_s(y_hat).reshape(-1, self.n, self.d)
        return x_hat, lik
    
    def eval(self, x):
        with torch.no_grad():
            y = self.g_a(x.reshape(-1, self.n*self.d)).reshape(-1, self.n)
            y_hat = self.quantizer(y)
            x_hat = self.g_s(y_hat).reshape(-1, self.n, self.d)
            u2 = self._sample_from_voronoi(device=x.device)
            lik = self.voronoi_volume*self.entropy_bottleneck._likelihood(y_hat, u2) # [B, d]
        return x_hat, lik, y, y_hat
    
    def aux_loss(self):
        loss = 0.
        return cast(torch.Tensor, loss)
    
# class LatticeCompanderSimple(nn.Module):
#     def __init__(self, d, lattice='Hexagonal', N=2048):
#         super().__init__()
#         d_latent = 100
#         activation = nn.CELU()
#         self.a = nn.Parameter(torch.randn(1))
#         self.b = nn.Parameter(torch.randn(1))
#         if lattice == 'Hexagonal':
#             self.quantizer = HexagonalQuantizer(d)
#         elif lattice == 'Square':
#             self.quantizer = SquareQuantizer(d)
#         elif lattice == 'An':
#             self.quantizer = AnQuantizer(d)
#         elif lattice == 'Dn':
#             self.quantizer = DnQuantizer(d)
#         else:
#             raise Exception("lattice type invalid")
#         self.entropy_bottleneck = EntropyBottleneckLattice(channels=d)
#         self.N = N

#         self.voronoi_volume = np.sqrt(np.linalg.det((self.quantizer.G @ self.quantizer.G.T).numpy()))

#     def forward(self, x):
#         # x : [B, d]    
#         # Get latent y
#         y = self.a * x # [B, d]


#         # Compute likelihoods term
#         u = torch.rand(y.shape, device=x.device) @ self.quantizer.G.to(x.device)
#         u = u - self.quantizer(u)
#         u2 = torch.rand((self.N, y.shape[1]), device=x.device) @ self.quantizer.G.to(x.device)
#         u2 = u2 - self.quantizer(u2) # u2 is uniformly sampled from Voronoi region centered at origin.
#         lik = self.entropy_bottleneck(y+u, u2) # [B, d]

#         # Compute reconstructions
#         y_hat = y + (self.quantizer(y) - y).detach()
#         # y_hat = y + u
#         x_hat = (y_hat / self.a) * 0.7
        
#         return x_hat, lik
    
#     def eval(self, x):
#         # get reconstructions.
#         with torch.no_grad():
#             y = self.a * x # [B, d]
#             y_hat = self.quantizer(y)
#             x_hat = self.b * y_hat

#         # Rate term, performing dithered quantization.
#         u = torch.rand(y.shape, device=x.device) @ self.quantizer.G.to(x.device)
#         u = u - self.quantizer(u)
#         u2 = torch.rand((self.N, y.shape[1]), device=x.device) @ self.quantizer.G.to(x.device)
#         u2 = u2 - self.quantizer(u2)
#         lik = self.voronoi_volume*self.entropy_bottleneck._likelihood(y_hat, u2) # [B, d]
#         # lik = self.voronoi_volume*self.entropy_bottleneck._likelihood(self.quantizer(y - u) + u, u2) # [B, d]
#         return x_hat, lik, y, y_hat

#     def aux_loss(self):
#         loss = 0.
#         return cast(torch.Tensor, loss)




class GaussianScalarCompander(nn.Module):
    """Optimal scalar compander for Gaussian under high rate assumption."""
    def __init__(self, N):
        super().__init__()
        self.centers = torch.arange(N) / N + (1/(2*N))
        self.N = N

    def CDF_Y(self, y):
        return 0.5*torch.erf(np.sqrt(3)*torch.erfinv(2*y-1)) + 0.5
    
    def likelihood(self, k):
        return self.CDF_Y((k+1)/self.N) - self.CDF_Y(k/self.N)

    def forward(self, x):
        # x : [B, 1]
        y = 0.5*torch.erf(x / np.sqrt(6)) + 0.5
        mse = (y - self.centers[None,:])**2 # [B, num_centers]
        indices = torch.argmin(mse, dim=1) # [B]
        y_hat = self.centers[indices].unsqueeze(1)
        x_hat = np.sqrt(6)*torch.erfinv(2*y_hat - 1)
        return x_hat, self.likelihood(indices), y_hat
    
    