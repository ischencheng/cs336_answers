import torch
from torch import Tensor
import torch.nn as nn
import math
from einops import einsum
from jaxtyping import Float, Int
class RoPE(nn.Module):
    def __init__(self,
        theta:float,
        d_k:int,
        max_seq_len:int,
        device: torch.device | None=None    
    ) -> None:
        super().__init__()
        freqs=torch.exp(-math.log(theta)*(torch.arange(0,d_k,2,device=device)/d_k))
        positions=torch.arange(0,max_seq_len,1,device=device)
        freqs_positions=einsum(freqs,positions,"d,l->l d")
        freqs_positions_full=freqs_positions.repeat_interleave(2,dim=-1)
        self.register_buffer('cached_cos',torch.cos(freqs_positions_full),persistent=False)
        self.register_buffer('cached_sin',torch.sin(freqs_positions_full),persistent=False)

    def forward(self, 
        x: Float[Tensor, " ... sequence_length d_k"], 
        token_positions: Int[Tensor, " ... sequence_length"]
    ) -> torch.Tensor:
        cos=self.cached_cos[token_positions].to(x.dtype)
        sin=self.cached_sin[token_positions].to(x.dtype)
        # x: [..., d_k]
        # we need to interleave: x_rotated = [-x[..., 1::2], x[..., 0::2]]
        x_rotated = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape(x.shape)
        
        return cos * x + sin * x_rotated