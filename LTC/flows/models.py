import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import math

def mvnlogprob(dist: torch.distributions.multivariate_normal.MultivariateNormal, 
               inputs: torch.tensor):
    p = dist.loc.size(0)
    diff = inputs - dist.loc
    
    batch_shape = diff.shape[:-1]
    
    scale_shape = dist.scale_tril.size()
    
    _scale_tril = dist.scale_tril.expand(batch_shape+scale_shape)
    z = torch.linalg.solve_triangular(_scale_tril,
                                      diff.unsqueeze(-1), 
                                      upper=False).squeeze()
    
    out=  -0.5*p*torch.tensor(2*math.pi).log() - dist.scale_tril.logdet() -0.5*(z**2).sum(dim=-1)
    return out.squeeze()

class MVN_custom(MultivariateNormal):
    """
    Multivariate normal with 0-mean, identity covariance. Used because torch MVN fails on large inputs. 
    """
    def __init__(
        self,
        loc,
        covariance_matrix=None,
        precision_matrix=None,
        scale_tril=None,
        validate_args=None,
    ):
        super().__init__(loc, covariance_matrix, precision_matrix, scale_tril, validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = torch.sum(value**2, dim=1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M)

class NormalizingFlowModel(nn.Module):

    def __init__(self, dim, flows):
        super().__init__()
        device = 'cuda:0'
        # self.prior = MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device))
        self.prior = MVN_custom(torch.zeros(dim, device=device), torch.eye(dim, device=device))
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        bsz, _ = x.shape
        # print(f"input={x}")
        log_det = torch.zeros(bsz, device=x.device)
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
        z = x
        # print(f"x_min={x.min()}, x_max={x.max()}")
        prior_logprob = self.prior.log_prob(x)
        # prior_logprob = mvnlogprob(self.prior, x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        bsz, _ = z.shape
        log_det = torch.zeros(bsz, device=x.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        return x
