import torch
from torch import Tensor
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding_module import Embedding
from cs336_basics.rmsnorm_module import RMSNorm
from cs336_basics.linear_module import LinearModule
from jaxtyping import Int,Float

class TransformerLM(nn.Module):
    def __init__(self,
        num_layers:int,
        d_model:int,
        num_heads:int,
        d_ff:int,
        context_length: int,
        theta: float,
        vocab_size:int,
        eps:float=1e-5,
        device: torch.device | None=None,
        dtype: torch.dtype | None=None
    ) -> None:
        super().__init__()
        self.embedding=Embedding(vocab_size,d_model,device, dtype)
        self.blocks=nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,theta, eps, device, dtype) for _ in range(num_layers)])
        self.norm=RMSNorm(d_model,eps,device,dtype)
        self.linear=LinearModule(d_model,vocab_size,device,dtype)

    
    def forward(self, 
        token_ids:Int[Tensor, " batch_size sequence_length"],
        token_positions: Int[Tensor, " batch_size sequence_length"] | None = None
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x=self.embedding(token_ids)
        
        if token_positions is None:
            s_len = token_ids.size(-1)
            token_positions = torch.arange(s_len, device=x.device)
            
        for block in self.blocks:
            x=block(x,token_positions)
        x=self.norm(x)
        x=self.linear(x)
        return x