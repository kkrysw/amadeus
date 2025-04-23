import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from nnAudio import Spectrogram
from .constants import *
from model.utils import Normalization

class stepwise_VAT(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, XI, epsilon, n_power):
        super().__init__()
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon

    def forward(self, model, x):  
        with torch.no_grad():
            y_ref, _ = model(x) # This will be used as a label, therefore no need grad()
            
        # generate_virtual_adversarial_perturbation
        d = torch.randn_like(x, requires_grad=True) # Need gradient
        for _ in range(self.n_power):
            r = self.XI * _l2_normalize(d)
            y_pred, _ = model(x + r)
            dist =F.binary_cross_entropy(y_pred, y_ref)
            dist.backward() # Calculate gradient wrt d
            d = d.grad.detach()
            model.zero_grad() # prevent gradient change in the model    

        # generating virtual labels and calculate VAT    
        r_adv = self.epsilon * _l2_normalize(d)
#         logit_p = logit.detach()
        y_pred, _ = model(x + r_adv)
        vat_loss = F.binary_cross_entropy(y_pred, y_ref)              
            
        return vat_loss, r_adv  # already averaged
    
def _l2_normalize(d):
    '''
    d = d/torch.norm(d, dim=2, keepdim=True)
    return d
    '''
    eps=1e-8
    if torch.isnan(d).any():
        print("WARNING: d has NaNs before normalization")
        d = torch.nan_to_num(d, nan=0.0)

    norm = torch.norm(d, dim=2, keepdim=True)
    if torch.isnan(norm).any() or (norm == 0).any():
        print(f"WARNING: norm has NaNs or zeros. norm min={norm.min()} max={norm.max()}")

    norm = norm + eps  # Prevent division by zero
    d = d / norm

    if torch.isnan(d).any():
        print("WARNING: d has NaNs after normalization")
        d = torch.nan_to_num(d, nan=0.0)
    return d
