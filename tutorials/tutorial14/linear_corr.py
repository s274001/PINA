import torch
from base_corr import BaseCorrNet
from pina.model.layers import RBFBlock

class LinearCorrNet(BaseCorrNet):
    '''
    Simple neural network to approximate the correction term, where
    approx_correction=corr_coeff*pod_modes, where corr_coeff are
    correction reduced coefficients, multiplied by the POD modes.
    The coefficients are found with the least squares method (no NN here).
    '''
    def __init__(self, pod, interp=RBFBlock(), scaler=None):
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


