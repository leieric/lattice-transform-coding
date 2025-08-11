import warnings

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai.entropy_models import EntropyModel
from compressai.ops import LowerBound
# from MAF.maf import MAF
from flows.models import NormalizingFlowModel
from flows.flows import RealNVP, RealNVP2, MAF, NSF_CL, NSF_AR

class EntropyBottleneck(EntropyModel):
    r"""
    Original Entropy bottleneck layer from CompressAI library. 
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _get_medians(self) -> Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)
        samples = samples[None, :] + pmf_start[:, None, None]

        pmf, lower, upper = self._likelihood(samples, stop_gradient=True)
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(
        self, inputs: Tensor, stop_gradient: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        half = float(0.5)
        lower = self._logits_cumulative(inputs - half, stop_gradient=stop_gradient)
        upper = self._logits_cumulative(inputs + half, stop_gradient=stop_gradient)
        likelihood = torch.sigmoid(upper) - torch.sigmoid(lower)
        return likelihood, lower, upper


    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize

        outputs = self.quantize(
            values, "noise" if training else "dequantize", self._get_medians()
        )

        if not torch.jit.is_scripting():
            likelihood, _, _ = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)

class EntropyBottleneckLattice(nn.Module):
    """
    Parameterizes a p_y density model (same as Balle18) via its CDF, factorized over its components. 
    Since an integral over non-rectangular regions cannot always be written in terms of CDF evals, 
    this computes integral via Monte-Carlo averaging over the region.
    """
    def __init__(
            self,
            channels: int,
            tail_mass: float = 1e-9,
            init_scale: float = 10,
            filters: Tuple[int, ...] = (3, 3, 3, 3),
            likelihood_bound=1e-10,
            **kwargs: Any,
        ):
        super().__init__()
        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        # self.use_likelihood_bound = likelihood_bound > 0
        # if self.use_likelihood_bound:
        #     self.likelihood_lower_bound = LowerBound(likelihood_bound)
                
        target = np.log(2 / self.tail_mass - 1)
        self.target = torch.tensor([-target, 0, target])
        # self.target = torch.tensor([self.tail_mass/2, 0.5, 1-self.tail_mass/2])
    
    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            # print(matrix.shape, logits.shape)
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    # @torch.compile
    def compute_cdf(self, inputs, stop_gradient=False):
        return torch.sigmoid(self._logits_cumulative(inputs, stop_gradient=stop_gradient))
    
    # @torch.compile
    # def compute_pdf(self, inputs, train=True):
    #     if inputs.is_leaf:
    #         inputs.requires_grad=True
    #     cdf_inputs = self.compute_cdf(inputs)
    #     pdf_inputs = torch.autograd.grad(cdf_inputs.sum(), inputs, create_graph=True, retain_graph=True)[0] 
    #     if train:
    #         return pdf_inputs
    #     return pdf_inputs.detach()

    # @torch.compile

    # @torch.compile
    def compute_pdf(self, inputs, train=True):
        inputs = inputs.unsqueeze(2).unsqueeze(2)
        if inputs.is_leaf:
            inputs.requires_grad=True
        sum_cdf_func = lambda x : self.compute_cdf(x).sum()
        pdf_func = torch.func.grad(sum_cdf_func)
        pdf_inputs = pdf_func(inputs)
        if train:
            return pdf_inputs.squeeze()
        return pdf_inputs.detach().squeeze()



    def forward(self, inputs, noise, train=True):
        """
        Computes likelihood of convolved y+u distribution.
        """
        # inputs: [batch_size, channels] y+u (dithered output of analysis trasnsform)
        # noise: [N, channels] sampled from Voronoi region
        # note: channels should be equivalent to dimension of vectors to be quantized
        batch_size, dim = inputs.shape
        N,_ = noise.shape
        z_u = inputs[:,None,:] + noise[None,:,:] # [batch_size, N, channels]
        # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        z_u = z_u.reshape(batch_size*N, dim, 1, 1) #[batch_size*N, dim, 1, 1]
        p_y = self.compute_pdf(z_u, train)
        p_y = p_y.reshape(batch_size, N, dim)
        lik = torch.mean(p_y, dim=1) #[batch_size, channels]
        # if lik.max() > 1:
        #     print(lik.max())
        return lik

    def _likelihood(self, inputs, noise):
        batch_size, dim = inputs.shape
        N,_ = noise.shape
        z_u = inputs[:,None,:] + noise[None,:,:] # [batch_size, N, channels]
        # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        z_u = z_u.reshape(batch_size*N, dim) #[batch_size*N, dim, 1, 1]
        p_y = self.compute_pdf(z_u, True)
        p_y = p_y.reshape(batch_size, N, dim).contiguous()
        lik = torch.mean(p_y, dim=1) #[batch_size, channels]
        # if self.use_likelihood_bound:
        #     lik = self.likelihood_lower_bound(lik)
        # if lik.max() > 1:
        #     print(lik.max())
        return lik
    
    @torch.no_grad()
    def _update_quantiles(self, search_radius=1e5, rtol=1e-6, atol=1e-5, device='cpu'):
        """Fast quantile update via bisection search.

        Often faster and much more precise than minimizing aux loss.
        """
        self.quantiles = torch.Tensor(self.channels, 1, 3)
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)
        print(self.quantiles)
        # device = self.quantiles.device
        shape = (self.channels, 1, 1)
        low = torch.full(shape, -search_radius, device=device)
        high = torch.full(shape, search_radius, device=device)

        def f(y, self=self):
            return self._logits_cumulative(y, stop_gradient=True)

        for i in range(len(self.target)):
            q_i = self._search_target(f, self.target[i], low, high, rtol, atol)
            self.quantiles[:, :, i] = q_i[:, :, 0]

    @staticmethod
    def _search_target(f, target, low, high, rtol=1e-4, atol=1e-3, strict=False):
        assert (low <= high).all()
        if strict:
            assert ((f(low) <= target) & (target <= f(high))).all()
        else:
            low = torch.where(target <= f(high), low, high)
            high = torch.where(f(low) <= target, high, low)
        while not torch.isclose(low, high, rtol=rtol, atol=atol).all():
            mid = (low + high) / 2
            f_mid = f(mid)
            low = torch.where(f_mid <= target, mid, low)
            high = torch.where(f_mid >= target, mid, high)
        return (low + high) / 2
    
class EntropyBottleneckLatticeFlow(nn.Module):
    """
    Parameterizes a p_y density model via a normalizing flow model.
    Since an integral over non-rectangular regions cannot always be written in terms of CDF evals, 
    this computes integral via Monte-Carlo averaging over the region.
    """
    def __init__(
            self,
            channels: int,
            flow_name: str,
            likelihood_bound=1e-12,
        ):
        super().__init__()
        # self.maf = MAF(dim=channels, n_layers=5, hidden_dims=[channels])
        n_flows = 5
        if flow_name == 'RealNVP':
            flow = RealNVP
        elif flow_name == 'RealNVP2':
            flow = RealNVP2
            n_flows=5
        elif flow_name == 'MAF':
            flow = MAF
        elif flow_name == 'NSF-CL':
            flow = NSF_CL
        elif flow_name == 'NSF-AR':
            flow = NSF_AR
        else:
            raise Exception("Invalid flow name")
        flows = [flow(channels, hidden_dim=channels) for _ in range(n_flows)]
        self.flow = NormalizingFlowModel(channels, flows)

        # self.use_likelihood_bound = likelihood_bound > 0
        # if self.use_likelihood_bound:
        #     self.likelihood_lower_bound = LowerBound(likelihood_bound)

    def _llhd(self, inputs):
        zs, prior_logprob, log_det = self.flow(inputs)
        logprob = prior_logprob + log_det
        # print(f"prior_logprob={prior_logprob}, min={prior_logprob.min()}, max={prior_logprob.max()}")
        # print(f"logprob={logprob}, logdet={log_det}, z_u={inputs}")
        return logprob
    
    def compute_pdf(self, inputs):
        # print(inputs.shape)
        logprob = self._llhd(inputs)
        # print(f"logprob={logprob}, min={logprob.min()}, max={logprob.max()}")
        return torch.exp(logprob)


    def forward(self, inputs, noise):
        """     
        Computes likelihood of convolved y+u distribution.
        """
        # inputs: [batch_size, channels] y+u (dithered output of analysis trasnsform)
        # noise: [N, channels] sampled from Voronoi region
        # note: channels should be equivalent to dimension of vectors to be quantized
        batch_size, dim = inputs.shape
        N,_ = noise.shape
        z_u = inputs[:,None,:] - noise[None,:,:] # [batch_size, N, channels]
        # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        z_u = z_u.reshape(batch_size*N, dim) #[batch_size*N, dim]
        p_y = self.compute_pdf(z_u) #[batch_size*N, 1]
        # print(p_y[0:10])
        p_y = p_y.reshape(batch_size, N)
        lik = torch.mean(p_y, dim=1) #[batch_size]
        # if self.use_likelihood_bound:
        #     lik = self.likelihood_lower_bound(lik)
        # print(lik)
        # if lik.max() > 1:
        #     print(lik.max())
        return lik

    def _likelihood(self, inputs, noise):
        batch_size, dim = inputs.shape
        N,_ = noise.shape
        z_u = inputs[:,None,:] + noise[None,:,:] # [batch_size, N, channels]
        # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        z_u = z_u.reshape(batch_size*N, dim).contiguous() #[batch_size*N, dim, 1, 1]
        # print(inputs, z_u)
        p_y = self.compute_pdf(z_u)
        # print(f"p_y={p_y}, min={p_y.min()}, max={p_y.max()}")
        p_y = p_y.reshape(batch_size, N).contiguous()
        lik = torch.mean(p_y, dim=1) #[batch_size]
        # if self.use_likelihood_bound:
        #     lik = self.likelihood_lower_bound(lik)
        # if lik.max() > 1:
        #     print(lik.max())
        return lik
    
    def _log_likelihood(self, inputs, noise):
        batch_size, dim = inputs.shape
        N,_ = noise.shape
        z_u = inputs[:,None,:] + noise[None,:,:] # [batch_size, N, channels]
        # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        z_u = z_u.reshape(batch_size*N, dim).contiguous() #[batch_size*N, dim, 1, 1]
        # print(inputs, z_u)
        log_p_y = self._llhd(z_u)
        log_p_y = log_p_y.reshape(batch_size, N).contiguous()
        log_lik = torch.mean(log_p_y, dim=1) #[batch_size]
        # if lik.max() > 1:
        #     print(lik.max())
        return log_lik 
    

class EntropyBottleneckLatticeDecomposed(nn.Module):
    """
    Parameterizes a p_y density model (same as Balle18) via its CDF, factorized over its components. 
    Assumes Lattice can be decomposed into 2 square lattices, e.g. Diamond or Hexagonal.
    """
    def __init__(
            self,
            channels: int,
            tail_mass: float = 1e-9,
            init_scale: float = 10,
            filters: Tuple[int, ...] = (3, 3, 3, 3),
            **kwargs: Any,
        ):
        super().__init__()
        self.eb1 = EntropyBottleneck(2)
        self.eb2 = EntropyBottleneck(2)
    
    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            # print(matrix.shape, logits.shape)
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    def compute_cdf(self, inputs, stop_gradient=False):
        return torch.sigmoid(self._logits_cumulative(inputs, stop_gradient=stop_gradient))
    
    def compute_pdf(self, inputs, train=True):
        if inputs.is_leaf:
            inputs.requires_grad=True
        cdf_inputs = self.compute_cdf(inputs)
        pdf_inputs = torch.autograd.grad(cdf_inputs.sum(), inputs, create_graph=True, retain_graph=True)[0] 
        if train:
            return pdf_inputs
        return pdf_inputs.detach()

    def forward(self, inputs, train=True):
        # inputs are not dithered
        x_scaled = inputs
        x_scaled[:,1] /= 3**0.5

        y1 = x_scaled + torch.rand_like(x_scaled) - 0.5
        y2 = x_scaled - 0.5 + torch.rand_like(x_scaled) - 0.5
        y1[:,1] *= 3**0.5
        y2[:,1] *= 3**0.5

        """ 
        TODO: Compute logits of both y1 and y2. Need to figure out if we use LVQAC loss (parametrize a distribution over the 2-level VQ.)
        """

        # """
        # Computes likelihood of convolved y+u distribution.
        # """
        # # inputs: [batch_size, channels] y+u (dithered output of analysis trasnsform)
        # # noise: [N, channels] sampled from Voronoi region
        # # note: channels should be equivalent to dimension of vectors to be quantized
        # batch_size, dim = inputs.shape
        # N,_ = noise.shape
        # z_u = inputs[:,None,:] - noise[None,:,:] # [batch_size, N, channels]
        # # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        # z_u = z_u.reshape(batch_size*N, dim, 1, 1) #[batch_size*N, dim, 1, 1]
        # p_y = self.compute_pdf(z_u, train)
        # p_y = p_y.reshape(batch_size, N, dim)
        # lik = torch.mean(p_y, dim=1) #[batch_size, channels]
        # # if lik.max() > 1:
        # #     print(lik.max())
        return lik

    def _likelihood(self, inputs, noise):
        batch_size, dim = inputs.shape
        N,_ = noise.shape
        z_u = inputs[:,None,:] + noise[None,:,:] # [batch_size, N, channels]
        # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        z_u = z_u.reshape(batch_size*N, dim, 1, 1) #[batch_size*N, dim, 1, 1]
        p_y = self.compute_pdf(z_u, False)
        p_y = p_y.reshape(batch_size, N, dim)
        lik = torch.mean(p_y, dim=1) #[batch_size, channels]
        # if lik.max() > 1:
        #     print(lik.max())
        return lik