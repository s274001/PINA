import torch
from pina.model.feed_forward import FeedForward
from pina.model.layers import ResidualBlock, OrthogonalBlock, RBFBlock
from pina.model import DeepONet
from pina import LabelTensor
from base_corr import BaseCorrNet

class PrototypeNet(torch.nn.Module):
    '''
    Build a neural network with residual connections. Concatenates two residual blocks and results in a network with three hidden layers.

    :param input_dim: the dimension of the input layer
    :param hidden_dim: dimension of the hidden layers
    :param output_dim: dimension of the output layer
    :param activation: activation function
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
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        out = self.rb1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.rb2(out)
        
        return out


class DeepONetCorrNet(BaseCorrNet):
    '''
    Build a DeepONet model that transform the POD modes and coefficients. Has two DeepONets: one takes as input the parameter and coefficients, the other takes as input the coordinates and modes.
    '''
    def __init__(self, pod, coords, interp=RBFBlock(), scaler=None):
        super().__init__(pod, interp=interp, scaler=scaler)

        self.coords = coords
        self.modes = LabelTensor(self.modes, [f'mode_{i}' for i in range(self.reduced_dim)])

        # build the mode network
        out_size = 10
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
        #branch_net = PrototypeNet(input_dim=self.reduced_dim,
        #                          hidden_dim=10,
        #                          output_dim=out_size,
        #                          activation=torch.nn.Softplus()
        #                          )
        #trunk_net = PrototypeNet(input_dim=1,
        #                          hidden_dim=10,
        #                          output_dim=out_size,
        #                          activation=torch.nn.Softplus()
        #                          )
        #reduction_layer = torch.nn.Linear(out_size,self.reduced_dim)

        #self.coef_net = DeepONet(branch_net,
        #                         trunk_net,
        #                         [f'coef_{i}' for i in range(self.reduced_dim)],
        #                         ['mu'],
        #                         reduction=reduction_layer
        #                         )
        self.coef_net = FeedForward(input_dimensions=1,
                                    output_dimensions=self.reduced_dim,
                                    layers=[10,10])
        self.orth = OrthogonalBlock(dim=-1)

    def fit(self, params, corrections):
        '''
        Fit the scaler if not None.
        '''
        if self.scaler is not None:
            corrections = self.scaler.fit_transform(corrections)
        return corrections

    #@property
    def transformed_modes(self):
        '''
        Compute the transformed modes.
        '''
        input = self.coords.append(self.modes)
        transf_modes = self.mode_net(input) 
        transf_modes = self.orth(transf_modes.tensor)
        #return orth_modes
        return transf_modes 

    def transformed_coefficients(self, coef_orig, param):
        '''
        Compute the transformed coefficients.
        '''
        #coef_orig = LabelTensor(coef_orig, [f'coef_{i}' for i in range(self.reduced_dim)])
        #input = param.append(coef_orig)
        #transf_coef = self.coef_net(input)
        transf_coef = self.coef_net(param)
        return transf_coef


    def forward(self, param, coef_orig):
        '''
        Compute the correction term.
        :param param: parameter
        :param coef_original: the coefficients in the original POD expansion
        '''        
        coef = self.transformed_coefficients(coef_orig, param)
        modes = self.transformed_modes()

        approx_correction = torch.matmul(coef,modes.T)

        #if self.scaler is not None:
        #    approx_correction = self.scaler.inverse_transform(approx_correction)

        return approx_correction

