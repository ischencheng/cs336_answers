import torch
from torch import Tensor
from jaxtyping import Float
def softmax(x:Float[Tensor, " ..."],i:int)->torch.Tensor:
    x_norm=x-torch.max(x,dim=i,keepdim=True).values
    x_norm_exp=torch.exp(x_norm)
    x_norm_sum=torch.sum(x_norm_exp,dim=i,keepdim=True)
    return x_norm_exp/x_norm_sum


if __name__=="__main__":
    test_tensor=torch.tensor([[ 1.0, 2.0, 3.0 ],[ 4.0, 5.0, 6.0 ]])
    print(softmax(test_tensor,0))