import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.msa_module import MSA
from cs336_basics.swiglu_module import FFN_SwiGLU
from cs336_basics.rmsnorm_module import RMSNorm
from jaxtyping import Float,Int

class TransformerBlock(nn.Module):
    def __init__(self,
        d_model:int,
        num_heads:int,
        d_ff:int,
        max_seq_len: int,
        theta: float,
        eps:float=1e-5,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None
    ) -> None:
        super().__init__()
        self.ffn=FFN_SwiGLU(d_model,d_ff,device,dtype)
        self.msa=MSA(d_model,num_heads,device=device, dtype=dtype,theta=theta, max_seq_len=max_seq_len,causal=True)
        self.rmsnorm1=RMSNorm(d_model,eps,device,dtype)
        self.rmsnorm2=RMSNorm(d_model,eps,device,dtype)

    def forward(self,
        x: Float[Tensor, " batch sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None
    )->Float[Tensor, " batch sequence_length d_model"]:
        if token_positions is None:
            seq_len= x.size(-2)
            token_positions = torch.arange(seq_len, device=x.device)
        x = x + self.msa(self.rmsnorm1(x), token_positions=token_positions)
        x = x + self.ffn(self.rmsnorm2(x))
        return x