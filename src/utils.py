import torch
import torch.nn
eps = 1e-7

def l2(x,y):
    return torch.nn.MSELoss()(x,y)
    
def kl_div(x,y):
    x = x+eps
    y = y+eps
    return (x * torch.log(x/y) - x + y).mean()

def l2s(x,y):
    return torch.nn.MSELoss(reduction='sum')(x,y)

def l2_sparse(x,y):
    return torch.nn.MSELoss()(torch.masked_select(x,y!=0),torch.masked_select(y,y!=0))