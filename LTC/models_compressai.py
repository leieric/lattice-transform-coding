from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from torch import Tensor
from compressai.models import Cheng2020Attention
from compressai.entropy_models import EntropyModel
from compressai.ops import LowerBound
import scipy

from LTC.quantizers import get_lattice

class GaussianConditionalLattice(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    Modified to use lattice quantization + MC integration.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://github.com/tensorflow/compression/blob/v1.3/docs/api_docs/python/tfc/GaussianConditional.md>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        channels=192,
        N_integral=2048,
        lattice_name="E8Product",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )
        self.channels = channels
        self.N_integral = N_integral
        # self.quantizer = E8ProductQuantizer(dim=channels, n_product=channels//8)
        self.quantizer = get_lattice(lattice_name, channels)
        self.sobol_eng = torch.quasirandom.SobolEngine(dimension=channels, scramble=True)

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)
    
    def _normal_pdf(self, x):
        return (1/torch.sqrt(2*np.pi))*torch.exp(-0.5 * x**2)
    
    def normal_pdf(self, x, mean, scale):
        return (1/torch.sqrt(2*np.pi*scale**2))*torch.exp(-0.5*((x-mean)/scale)**2)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _flatten(self, y):
        # y: [B, dy, h, w]
        y = y.permute(0, 2, 3, 1) #[B, h, w, dy]
        y_shape = y.size()
        y = y.reshape(-1, self.channels) # [B*h*w, dy]
        return y, y_shape
    
    def _unflatten(self, y_hat, y_shape):
        y_hat = y_hat.reshape(y_shape) # [B, h, w, dy]
        y_hat = y_hat.permute(0, 3, 1, 2)
        return y_hat

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            y, y_shape = self._flatten(inputs)
            u = self.sobol_eng.draw(y.shape[0]).to(y.device)
            u = u @ self.quantizer.G.to(y.device)
            u = u - self.quantizer(u)
            y_tilde = y + u
            y_hat = self.quantizer(y)
            y_hat = y + (y_hat - y).detach()
            y_tilde = self._unflatten(y_tilde, y_shape)
            y_hat = self._unflatten(y_hat, y_shape)
            return y_tilde, y_hat
            # half = float(0.5)
            # noise = torch.empty_like(inputs).uniform_(-half, half)
            # inputs = inputs + noise
            # return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        y, y_shape = self._flatten(outputs)
        y_hat = self.quantizer(y)
        outputs = self._unflatten(y_hat, y_shape)
        # outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs, outputs

        assert mode == "symbols", mode
        # outputs = outputs.int()
        return outputs
    
    def _lik_MC_est(self, inputs, noise, scales, means):
        # inputs, scales, means: [bsize, channels]
        # noise: [N, channels]
        # print(inputs.shape, noise.shape, scales.shape, means.shape)
        batch_size, dim = inputs.shape
        N,_ = noise.shape
        # if means is not None:
        #     values = inputs - means
        # else:
        #     values = inputs
        z_u = inputs[:,None,:] + noise[None,:,:] # [batch_size, N, channels]
        # z_u = z_u.permute(0,2,1).unsqueeze(3) # [batch_size, channels, N]
        # z_u = z_u.reshape(batch_size*N, dim).contiguous() #[batch_size*N, dim, 1, 1]
        # print(inputs, z_u)
        # p_y = (1 / scales[:,None,:]) * self._normal_pdf(z_u / scales[:,None,:]) # [batch_size, N, channels]
        p_y = self.normal_pdf(z_u, means[:,None,:], scales[:,None,:])
        # print(f"p_y={p_y}, min={p_y.min()}, max={p_y.max()}")
        # p_y = p_y.reshape(batch_size, N).contiguous()
        lik = torch.mean(p_y, dim=1) #[batch_size, channels]
        # if lik.max() > 1:
        #     print(lik.max())
        return lik

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        scales = self.lower_bound_scale(scales)
        # u = torch.rand((self.N_integral, self.channels), device=inputs.device)
        u = self.sobol_eng.draw(self.N_integral).to(inputs.device)
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        y, y_shape = self._flatten(inputs)
        scales_flatten, _ = self._flatten(scales)
        means_flatten, _ = self._flatten(means)
        lik = self._lik_MC_est(y, u2, scales_flatten, means_flatten)
        lik = self._unflatten(lik, y_shape) # [same as inputs]
        # print(inputs.shape, lik.shape)
        # print(lik)
        return lik

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        outputs_noise = self.quantize(inputs, "noise" if training else "dequantize", means) # to adjust
        likelihood = self._likelihood(outputs_noise, scales, means) # to adjust
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs_noise, likelihood

    def build_indexes(self, scales: Tensor) -> Tensor:
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes

class Cheng2020AttentionLattice(Cheng2020Attention):
    """
    Modified to use the GaussianConditionalLattice (Gaussian mean/scale hyperprior, with lattice).

    """

    def __init__(self, N=192, N_integral=2048, lattice_name="E8Product", **kwargs):
        super().__init__(N=N, **kwargs)
        self.gaussian_conditional = GaussianConditionalLattice(None, channels=N, N_integral=N_integral, lattice_name=lattice_name)
    
    def forward(self, x):
        """
        Modified to use STE for reconstruction, noise for entropy modelling.
        """
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_tilde, y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_tilde)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_likelihoods = self.gaussian_conditional._likelihood(y_tilde, scales_hat, means=means_hat)
        if self.gaussian_conditional.use_likelihood_bound:
            y_likelihoods = self.gaussian_conditional.likelihood_lower_bound(y_likelihoods)
        # _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
        x_hat = self.g_s(y_hat)
        # x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    
