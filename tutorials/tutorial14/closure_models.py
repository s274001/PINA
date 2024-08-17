import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
from pina.solvers import ReducedOrderModelSolver as ROMSolver
from pina.solvers import SupervisedSolver
from pina.model.layers import PODBlock, RBFLayer, ResidualBlock
from pina.model import FeedForward, DeepONet
from pina.geometry import CartesianDomain
from pina.problem import AbstractProblem, ParametricProblem
from pina import Condition, LabelTensor, Trainer
from pina.callbacks import MetricTracker
from smithers.dataset import NavierStokesDataset
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseCorrNet(ABC, torch.nn.Module):
    '''
    Base class for the correction neural network, that will approximate the
    correction term in the ROM. The correction term is the difference between
    the solution approximated with a large number of modes and the solution
    approximated with the reduced number of modes considered.
    '''
    def __init__(self, pod, interp=RBFLayer(), scaler=None):
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


class LinearCorrNet(BaseCorrNet):
    '''
    Simple neural network to approximate the correction term, where
    approx_correction=corr_coeff*pod_modes, where corr_coeff are
    correction reduced coefficients, multiplied by the POD modes.
    The coefficients are found with the least squares method (no NN here).
    '''
    def __init__(self, pod, interp=RBFLayer(), scaler=None):
        super().__init__(pod, interp=interp, scaler=scaler)
        self.fictitious_params = torch.nn.Parameter(torch.randn(1))

    def fit(self, params, corrections):
        '''
        Fit the scaler, perform the least squares method and fit the
        params2correction to find the coefficients for unseen parameters.
        '''
        if self.scaler is not None:
            corrections = self.scaler.fit_transform(corrections)
        coeff_corr = torch.linalg.lstsq(self.modes, corrections.T).solution.T
        self.params2correction.fit(params, coeff_corr)
        return coeff_corr

    def forward(self, param_test):
        '''
        Compute the correction term for param_test.
        '''
        coeff_corr_test = self.params2correction.forward(param_test)
        approx_correction = torch.matmul(self.modes, coeff_corr_test.T).T
        if approx_correction.dim() == 1:
            approx_correction = approx_correction.unsqueeze(0)
        if self.scaler is not None:
            approx_correction = self.scaler.inverse_transform(approx_correction)
        return approx_correction


class PrototypeNet(torch.nn.Module):
    '''
    Build a neural network with residual connections. This is supposed to be used as branch and trunk for a DeepONet to mitigate possible vanishing gradients.
    '''
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 activation = torch.nn.ReLU(),
                 ):
        super().__init__()
        self.rb1 = ResidualBlock(input_dim=input_dim,
                                 hidden_dim=hidden_dim,
                                 output_dim=hidden_dim,
                                 activation=activation)
        self.rb2 = ResidualBlock(input_dim=hidden_dim,
                                 hidden_dim=hidden_dim,
                                 output_dim=output_dim,
                                 activation=activation)
        self.activation = activation

    def forward(self, x):
        out = self.rb1(x)
        out = self.activation(out)
        out = self.rb2(out)
        
        return out


class DeepONetCorrNet(BaseCorrNet):
    '''
    Build a DeepONet model that transform the POD modes, taking as input the
    POD modes and the x coordinates, and output the exact correction, built as
    (learnable coefficients)@(transformed_modes), where the transformed modes
    are the actual output of the DeepONet.
    '''
    def __init__(self, pod, coords, interp=RBFLayer(), scaler=None):
        super().__init__(pod, interp=interp, scaler=scaler)

        self.coords = coords
        self.modes = LabelTensor(self.modes, [f'mode_{i}' for i in range(self.reduced_dim)])

        # build the mode network
        out_size = 10
        #branch_net = FeedForward(
        #        input_dimensions=self.reduced_dim,
        #        output_dimensions=out_size,
        #        layers=[20, 20, 20],
        #        func=torch.nn.ReLU,
        #        )
        #trunk_net = FeedForward(
        #        input_dimensions=self.coords.shape[1],
        #        output_dimensions=out_size,
        #        layers=[20, 20, 20],
        #        func=torch.nn.ReLU,
        #        )
        branch_net = PrototypeNet(input_dim=self.reduced_dim,
                                  hidden_dim=20,
                                  output_dim=out_size,
                                  activation=torch.nn.Softplus()
                                  )
        trunk_net = PrototypeNet(input_dim=self.coords.shape[1],
                                  hidden_dim=20,
                                  output_dim=out_size,
                                  activation=torch.nn.Softplus()
                                  )
        reduction_layer = torch.nn.Linear(out_size,self.reduced_dim)

        self.mode_net = DeepONet(branch_net, 
                                 trunk_net, 
                                 [f'mode_{i}' for i in range(self.reduced_dim)],
                                 ['x','y'],
                                 reduction=reduction_layer
                                 )

        # build the coefficient network
        out_size = 5
        #branch_net = FeedForward(
        #        input_dimensions=self.reduced_dim,
        #        output_dimensions=out_size,
        #        layers=[10, 10, 10],
        #        func=torch.nn.ReLU,
        #        )
        #trunk_net = FeedForward(
        #        input_dimensions=1,
        #        output_dimensions=out_size,
        #        layers=[10, 10, 10],
        #        func=torch.nn.ReLU,
        #        )
       # branch_net = PrototypeNet(input_dim=self.reduced_dim,
       #                           hidden_dim=10,
       #                           output_dim=out_size,
       #                           activation=torch.nn.Softplus()
       #                           )
       # trunk_net = PrototypeNet(input_dim=1,
       #                           hidden_dim=10,
       #                           output_dim=out_size,
       #                           activation=torch.nn.Softplus()
       #                           )
       # reduction_layer = torch.nn.Linear(out_size,self.reduced_dim)

       # self.coef_net = DeepONet(branch_net,
       #                          trunk_net,
       #                          [f'coef_{i}' for i in range(self.reduced_dim)],
       #                          ['mu'],
       #                          reduction=reduction_layer
       #                          )
        self.coef_net = FeedForward(input_dimensions=self.reduced_dim+1,
                                    layers=[10, 10, 10],
                                    output_dimensions=self.reduced_dim,
                                    func=torch.nn.Softplus
                                    )


    def fit(self, params, corrections):
        '''
        Fit the scaler if not None.
        '''
        if self.scaler is not None:
            corrections = self.scaler.fit_transform(corrections)

    @property
    def transformed_modes(self):
        '''
        Compute the transformed modes from the trunk net.
        '''
        input = self.coords.append(self.modes)
        transf_modes = self.mode_net(input) 
        return transf_modes

    def transformed_coefficients(self, coef_orig, param):
        '''
        Compute the transformed coefficients from the branch net.
        '''
        coef_orig = LabelTensor(coef_orig, [f'coef_{i}' for i in range(self.reduced_dim)])
        input = param.append(coef_orig)
        transf_coef = self.coef_net(input)
        return transf_coef


    def forward(self, param, coef_orig):
        '''
        Compute the correction term as the output of the DeepONet.
        '''        
        coef = self.transformed_coefficients(coef_orig, param)
        modes = self.transformed_modes

        approx_correction = torch.matmul(coef,modes.T)

        #approx_correction = self.deeponet(input)

        if self.scaler is not None:
            approx_correction = self.scaler.inverse_transform(approx_correction)

        return approx_correction

