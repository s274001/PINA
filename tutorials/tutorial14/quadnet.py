import torch
import torch.nn as nn
from pina.model import FeedForward

class QuadNet(nn.Module):
    def __init__(self, modes,coordinates):
        super().__init__()
        self.scaler = None #might want to get rid of this

        self.modes = modes
        self.coords = coordinates
        r = modes.shape[1]
        self.b = FeedForward(input_dimensions=r,
                              output_dimensions=r*(r+1)//2,
                              layers = [20,20,20,20,20,20,20],#,20,20,20],
                              func = nn.Tanh)
        self.t = FeedForward(input_dimensions=2,
                             output_dimensions=r*(r+1)//2,
                             layers = [20,20,20,20,20,20,20],
                             func = nn.Tanh)
        # self.bmu = FeedForward(input_dimensions=1,
        #                      output_dimensions=r*(r+1)//2,
        #                      layers = [20,20,20,20,20,20,20],
        #                      func = nn.Tanh)
        self.red = FeedForward(input_dimensions=r*(r+1)//2,
                             output_dimensions=r*(r+1)//2,
                             n_layers = 1,
                             func = nn.Tanh)

    def forward(self,par,coef):
        #c1 = torch.reshape(self.b1(self.modes),(-1,coef.shape[1],coef.shape[1]))
        z1 = self.b(self.modes)
        z2 = self.t(self.coords)
        # z3 = self.bmu(par)
        # z = torch.einsum('bi,Ni,Ni->bNi',z3,z2,z1)
        c = self.red(z1*z2) #case without mu
        # c = self.red(z)

        r = self.modes.shape[1]
        indices = torch.triu_indices(r,r)
        indices = r * indices[0] + indices[1]
        ai = torch.einsum("ni,nj->nij",coef,coef).flatten(start_dim=1)[:,indices]
        return torch.einsum("ni,Ni->nN",ai,c) #case without mu
        # return torch.einsum("ni,nNi->nN",ai,c)

    def C(self,par):
        # c1 = torch.reshape(self.b1(self.modes),(-1,3,3))
        z1 = self.b(self.modes)
        z2 = self.t(self.coords)
        # z3 = self.bmu(par)
        # z = torch.einsum('bi,Ni,Ni->bNi',z3,z2,z1)
        c = self.red(z1*z2) #case without mu
        # c = self.red(z)
        return c

