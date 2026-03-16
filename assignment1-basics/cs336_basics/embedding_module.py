import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(self,
        num_embeddings:int,
        embedding_dim:int,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None,
    ):
        super().__init__()
        self.embedding=nn.Parameter(torch.empty(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype
        ))
        nn.init.trunc_normal_(tensor=self.embedding,a=-3,b=3)
    def forward(self, 
        token_ids:Int[Tensor, " ..."]
    )->Float[Tensor, " ... d_model"]:
        return self.embedding[token_ids]
