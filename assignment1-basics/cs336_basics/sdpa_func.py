import math
import torch
from torch import Tensor
from jaxtyping import Float,Bool
from cs336_basics.softmax_func import softmax
from einops import einsum
def sdpa(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... kv_len d_k"],
        V: Float[Tensor, " ... kv_len d_v"],
        mask: Bool[Tensor, " ... queries kv_len"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k=Q.shape[-1]
        pre_softmax_values= einsum(Q,K," ... queries d_k, ... kv_len d_k -> ... queries kv_len")/math.sqrt(d_k)
        if mask is not None:
            pre_softmax_values = pre_softmax_values.masked_fill(~mask, -torch.inf)
        attn_map=softmax(pre_softmax_values,-1)

        return einsum(attn_map, V, "... queries kv_len, ... kv_len d_v -> ... queries d_v")