import torch
from pina.model.layers import RBFBlock
from  abc import ABC, abstractmethod

class BaseCorrNet(ABC, torch.nn.Module):
    '''
    Base class for the correction neural network, that will approximate the
    correction term in the ROM. The correction term is the difference between
    the solution approximated with a large number of modes and the solution
    approximated with the reduced number of modes considered.
    '''
    def __init__(self, pod, interp=RBFBlock(), scaler=None):
        super().__init__()
        self.pod = pod
        self.reduced_dim = pod.rank
        self.modes = pod.basis.T
        self.scaler = scaler
        self.params2correction = interp

    @abstractmethod
    def fit(self, params, corrections):
        '''
        Fit the scaler (if not None).
        '''
        if self.scaler is not None:
            corrections = self.scaler.fit_transform(corrections)

    @abstractmethod
    def forward(self, params):
        pass


