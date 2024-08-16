import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import torch
from pina.solvers import ReducedOrderModelSolver as ROMSolver
from pina.solvers import SupervisedSolver
from pina.model.layers import PODBlock, RBFLayer
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


class DeepONetCorrNet(BaseCorrNet):
    '''
    Build a DeepONet model that transform the POD modes, taking as input the
    POD modes and the x coordinates, and output the exact correction, built as
    (learnable coefficients)@(transformed_modes), where the transformed modes
    are the actual output of the DeepONet.
    '''
    def __init__(self, pod, coords, num_params=400, interp=RBFLayer(),
            scaler=None):
        super().__init__(pod, interp=interp, scaler=scaler)
        self.coords = coords
        branch_net = FeedForward(
                input_dimensions=1,
                output_dimensions=10,
                layers=[20, 20, 20],
                func=torch.nn.Softplus,
                )
        trunk_net = FeedForward(
                input_dimensions=2,
                output_dimensions=10,
                layers=[20, 20, 20],
                func=torch.nn.Softplus,
                )
#        self.nn_coeffs = FeedForward(
#                input_dimensions=1,
#                output_dimensions=self.reduced_dim,
#                layers=[5, 5, 5, 5],
#                func=torch.nn.Softplus,
#                )
        self.deeponet = DeepONet(branch_net, trunk_net, ['modes'], ['x', 'y'])
        self.coeff_corr = torch.nn.Parameter(torch.randn(num_params, self.reduced_dim))

    def fit(self, params, corrections):
        '''
        Fit the scaler if not None.
        '''
        if self.scaler is not None:
            corrections = self.scaler.fit_transform(corrections)

    @property
    def transformed_modes(self):
        '''
        Compute the transformed modes from the DeepONet and reshape them.
        '''
        # build the input for the DeepONet
        ## repeat the coordinates reduced_dim times as columns
        coords = LabelTensor(self.coords.repeat(self.reduced_dim, 1),
                ['x', 'y'])
        ## the modes here are stored in columns, so concatenate those in a long column
        modes = torch.vstack([self.modes[:, i] for i in range(self.reduced_dim)])
        ## append concatenates along the 1st direction. So we have (x,y.mode)
        input_deeponet = coords.append(LabelTensor(modes, 'modes'))
        # compute the transformed modes and reshape them
        transf_modes = (self.deeponet(input_deeponet).reshape(-1,1) +
                input_deeponet.extract(['modes']))
        transf_modes = transf_modes.reshape(
                self.reduced_dim, transf_modes.shape[0]//self.reduced_dim).T
        return transf_modes


    def forward(self, param, coeff_corr):
        '''
        Compute the correction term, by multiplying the transformed modes
        with the learnable coefficients.
        '''
        #coeff_corr = self.nn_coeffs(param)
        transf_modes = self.transformed_modes
        approx_correction = torch.matmul(transf_modes, coeff_corr.T).T
        if approx_correction.dim() == 1:
            approx_correction = approx_correction.unsqueeze(0)
        if self.scaler is not None:
            approx_correction = self.scaler.inverse_transform(approx_correction)
        return approx_correction

