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
    def __init__(self, pod, coords, interp=RBFBlock(), scaler=None, mode_output_size = 10):
        super().__init__(pod, interp=interp, scaler=scaler)

        self.coords = coords
        self.modes = LabelTensor(self.modes, [f'mode_{i}' for i in range(self.reduced_dim)])

        # build the mode network
        out_size = mode_output_size
        #branch_net = PrototypeNet(input_dim=1,
        #                          hidden_dim=20,
        #                          output_dim=out_size,
        #                          activation=torch.nn.Softplus()
        #                          )
        #trunk_net = PrototypeNet(input_dim=self.coords.shape[1],
        #                          hidden_dim=20,
        #                          output_dim=out_size,
        #                          activation=torch.nn.Softplus()
        #                          )
        #reduction_layer = torch.nn.Linear(out_size,self.reduced_dim)
        branch_net = FeedForward(input_dimensions=1,
                                  inner_size=20,
                                  n_layers=3,
                                  output_dimensions=out_size,
                                  func=torch.nn.Softplus
                                  )
        trunk_net = FeedForward(input_dimensions=self.coords.shape[1],
                                  inner_size=20,
                                  n_layers=3,
                                  output_dimensions=out_size,
                                  func=torch.nn.Softplus
                                  )

        self.mode_net = DeepONet(branch_net, 
                                 trunk_net, 
                                 #[f'mode_{i}' for i in range(self.reduced_dim)],
                                 ['mode'],
                                 ['x','y'],
                                 #reduction=reduction_layer
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
        branch_net = FeedForward(input_dimensions=1,
                                  inner_size=10,
                                  n_layers=2,
                                  output_dimensions=out_size,
                                  func=torch.nn.Softplus
                                  )
        trunk_net = FeedForward(input_dimensions=1,
                                  inner_size=10,
                                  n_layers=2,
                                  output_dimensions=out_size,
                                  func=torch.nn.Softplus
                                  )

        self.coef_net = DeepONet(branch_net,
                                 trunk_net,
                                 ['coef'],
                                 #[f'coef_{i}' for i in range(self.reduced_dim)],
                                 ['mu'],
                                 #reduction=reduction_layer
                                 )
        #self.coef_net = FeedForward(input_dimensions=1,
        #                            output_dimensions=1,
        #                            layers=[10,10])
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
        #input = self.coords.append(self.modes)
        stack_coords = self.coords.tensor.repeat(self.reduced_dim,1)
        #print(stack_coords[:10,:], stack_coords[1639:1649,:])
        stack_coords = LabelTensor(stack_coords, self.coords.labels)
        #print(self.modes.tensor[:10,1])
        stack_modes = self.modes.tensor.T.ravel().unsqueeze(1)
        #print(stack_modes[1639:1649])
        stack_modes = LabelTensor(stack_modes, 'mode')
        input = stack_coords.append(stack_modes)
        #print(stack_coords.shape)
        #print(stack_modes.shape)
        
        transf_modes = self.mode_net(input).squeeze()
        #print(transf_modes.shape,type(transf_modes))
        #print(transf_modes[1639:1649])
        transf_modes = transf_modes.tensor.reshape(self.reduced_dim,-1).T + self.modes 
        #print(transf_modes[:10,1])
        transf_modes = self.orth(transf_modes.tensor)
        return transf_modes 

    def transformed_coefficients(self, coef_orig, param):
        '''
        Compute the transformed coefficients.
        '''
        #print(param[:3])
        param = param.tensor.unsqueeze(2).repeat(1,self.reduced_dim,1)
        #print(param.shape)
        #print(param[:3,:,:])
        param = LabelTensor(param,'mu')
        #print(coef_orig[:3])
        coef_orig = coef_orig.unsqueeze(2)
        #print(coef_orig.shape)
        #print(coef_orig.tensor[:3,:,:])
        coef_orig = LabelTensor(coef_orig,'coef')
        input = torch.cat([param,coef_orig],-1)
        input = input.flatten(start_dim=0,end_dim=1)
        #print(input[:9,:])
        input = LabelTensor(input,['mu','coef'])
        #print(input.shape)
        #coef_orig = LabelTensor(coef_orig, [f'coef_{i}' for i in range(self.reduced_dim)])
        #input = param.append(coef_orig)
        transf_coef = self.coef_net(input)
        #print(transf_coef.shape)
        #print(transf_coef[:9])
        transf_coef = transf_coef.tensor.reshape(-1,self.reduced_dim)
        #print(transf_coef.shape)
        #print(transf_coef[:3,:])
        #transf_coef = self.coef_net(param)
        #transf_coef = self.coef_net(coef_orig.unsqueeze(2)).reshape(self.reduced_dim,-1).T
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

