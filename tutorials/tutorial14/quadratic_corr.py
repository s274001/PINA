import torch
from base_corr import BaseCorrNet
from pina.model.layers import RBFBlock
import time

class QuadraticCorrNet(BaseCorrNet):
    '''
    Simple neural network to approximate the correction term, where
    approx_correction=coeff^T@C@coeff, where C is an operator rxNxr,
    which is the same for all parameters, and can be used also in test
    setting.
    '''
    def __init__(self, pod, coeffs, interp=RBFBlock(), scaler=None):
        super().__init__(pod, interp=interp, scaler=scaler)
        self.fictitious_params = torch.nn.Parameter(torch.randn(1))
        self.interp = interp
        self.coeffs = coeffs
        self.rank = pod.rank
        self.operator = None

    def D_matrix(self, coeffs):
        coefs_i = []
        for i in range(self.rank):
            coef_i = coeffs[:, :i+1]*coeffs[:, i]
            coefs_i.append(coef_i)
        D = torch.cat([coef.tensor for coef in coefs_i], dim=-1)
        return D

    def fit(self, params, corrections):
        '''
        Fit the scaler, perform the least squares method and fit the
        params2correction to find the coefficients for unseen parameters.
        '''
        R_mat = corrections
        D_mat = self.D_matrix(self.coeffs)
        if torch.linalg.cond(D_mat) >= 1e4:
            driver = 'gelsd'
        else:
            driver = 'gelsy'
        tic = time.time()
        self.operator = torch.linalg.lstsq(D_mat, R_mat,driver=driver).solution
        toc = time.time()
        print("Time for least squares: ", toc-tic)

    def forward(self, param_test, coeff_test):
        '''
        Compute the correction term for param_test.
        '''
        if self.operator is None:
            raise ValueError("The method needs to be fitted.")
        else:
            pass
        D_mat = self.D_matrix(coeff_test)
        pred = D_mat@self.operator
        return pred
    
    def C(self,input_):
        return self.operator.T




