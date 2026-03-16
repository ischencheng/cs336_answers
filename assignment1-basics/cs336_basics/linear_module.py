import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int
from einops import einsum
import math
class LinearModule(nn.Module):
    def __init__(self,
                in_features:int, 
                out_features:int, 
                device:torch.device | None=None, 
                dtype:torch.dtype | None=None
    )->None:
        super().__init__()
        self.W=nn.Parameter(torch.empty(
            *(out_features, in_features),
            device=device,
            dtype=dtype
        ))
        sigma=math.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(tensor=self.W,std=sigma,a=-3.0*sigma,b=3.0*sigma)
        return 

    def forward(self,
        x:Float[Tensor, " ... d_in"]
    )->Float[Tensor, " ... d_out"]:
        return einsum(x,self.W, "... d_in, d_out d_in -> ... d_out")