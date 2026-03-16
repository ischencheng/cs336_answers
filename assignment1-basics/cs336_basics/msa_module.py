import torch
import torch.nn as nn
from torch import Tensor
from cs336_basics.sdpa_func import sdpa
from cs336_basics.linear_module import LinearModule
from cs336_basics.rope_module import RoPE
from jaxtyping import Float,Int
from einops import rearrange
class MSA(nn.Module):
    def __init__(self, 
        d_model:int,
        num_heads:int,
        d_in:int | None=None,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None,
        theta:float | None=None,
        max_seq_len:int | None=None,
        causal:bool=True
    ) -> None:
        super().__init__()
        if d_in is None:
            d_in=d_model
        self.qkv_proj=LinearModule(d_in,3*d_model,device=device,dtype=dtype)
        self.o_proj=LinearModule(d_model,d_model,device=device,dtype=dtype)
        self.d_k=d_model//num_heads
        self.num_heads=num_heads
        self.rope=None
        self.causal=causal
        if theta is not None and max_seq_len is not None:
            self.rope=RoPE(theta,self.d_k,max_seq_len,device=device)
        if max_seq_len is not None:
            self.register_buffer(
                'causal_mask',
                ~torch.triu(torch.ones(max_seq_len,max_seq_len,dtype=torch.bool, device=device),diagonal=1),
                persistent=False
            )
        else:
            self.causal_mask = None
        

    def forward(self,
        x:Float[Tensor, " ... sequence_length d_in"],
        token_positions:Int[Tensor, " ... sequence_length"] | None=None
    )->Float[Tensor, " ... sequence_length d_out"]:
        s_len=x.size(-2)
        Q,K,V=self.qkv_proj(x).chunk(3,dim=-1)
        Q=rearrange(Q,'... s (n d_k) -> ... n s d_k',n=self.num_heads)
        K=rearrange(K,'... s (n d_k) -> ... n s d_k',n=self.num_heads)
        V=rearrange(V,'... s (n d_k) -> ... n s d_k',n=self.num_heads)
        causal_mask=None
        if self.causal_mask is not None and self.causal:
            causal_mask=self.causal_mask[:s_len,:s_len]
        elif self.causal:
            causal_mask=~torch.triu(torch.ones(s_len,s_len,dtype=torch.bool, device=x.device),diagonal=1)
        
        if self.rope is not None and token_positions is not None:
            Q=self.rope(Q,token_positions)
            K=self.rope(K,token_positions)
        V_out=rearrange(sdpa(Q,K,V,causal_mask),'... n s d_k -> ... s (n d_k)', n=self.num_heads)
        return self.o_proj(V_out)

        