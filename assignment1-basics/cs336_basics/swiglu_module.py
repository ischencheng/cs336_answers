import torch
from torch import Tensor
from jaxtyping import Float
import torch.nn as nn
from cs336_basics.linear_module import LinearModule
class FFN_SwiGLU(nn.Module):
    def __init__(self, 
        d_model:int,
        d_ff:int | None=None,
        device:torch.device | None=None,
        dtype:torch.dtype | None=None
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff=(int(8/3*d_model+64-1)//64+1)*64
        self.W1=LinearModule(d_model,d_ff,device=device,dtype=dtype)
        self.W2=LinearModule(d_ff,d_model,device=device,dtype=dtype)
        self.W3=LinearModule(d_model,d_ff,device=device,dtype=dtype)

    def _SiLU(self,
        x:Float[Tensor, " ... d_model"]
    )->Float[Tensor, " ... d_model"]:
        return x*torch.sigmoid(x)

    def forward(self,
        x:Float[Tensor, " ... d_model"]
    )->Float[Tensor, " ... d_model"]:
        return self.W2(self._SiLU(self.W1(x))*self.W3(x))