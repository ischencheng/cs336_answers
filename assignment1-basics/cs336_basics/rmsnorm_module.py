import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

class RMSNorm(nn.Module):
    def __init__(self,
        d_model:int,
        eps:float=1e-5,
        device:torch.device | None=None,
        dtype: torch.dtype | None=None
    ):
        super().__init__()
        self.gain=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.eps=eps

    def forward(self, 
        x:Float[Tensor, " ... d_model"]
    )->Float[Tensor, " ... d_model"]:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        x_norm=x*torch.rsqrt((x**2).mean(dim=-1,keepdim=True)+self.eps)
        return x_norm.to(in_dtype)*self.gain

