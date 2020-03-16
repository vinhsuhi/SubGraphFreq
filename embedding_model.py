import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.nn import init


def init_weight(modules, activation):
    """
    Weight initialization
    :param modules: Iterable of modules
    :param activation: Activation function.
    """
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data) #, gain=nn.init.calculate_gain(activation.lower()))
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


def get_act_function(activate_function):
    """
    Get activation function by name
    :param activation_fuction: Name of activation function 
    """
    if activate_function == 'sigmoid':
        activate_function = nn.Sigmoid()
    elif activate_function == 'relu':
        activate_function = nn.ReLU()
    elif activate_function == 'tanh':
        activate_function = nn.Tanh()
    else:
        return None
    return activate_function


class GCN(nn.Module):
    """
    The GCN multistates block
    """
    def __init__(self, activate_function, input_dim, output_dim):
        """
        activate_function: Tanh
        input_dim: input features dimensions
        output_dim: output features dimensions
        """
        super(GCN, self).__init__()
        if activate_function is not None:
            self.activate_function = get_act_function(activate_function)
        else:
            self.activate_function = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        init_weight(self.modules(), activate_function)
    
    def forward(self, input, A_hat):
        output = self.linear(input)
        output = torch.matmul(A_hat, output)
        if self.activate_function is not None:
            output = self.activate_function(output)
        output = F.normalize(output)
        return output


class FA_GCN(nn.Module):
    """
    Training a multilayer GCN model
    """
    def __init__(self, activate_function, num_GCN_blocks, input_dim, output_dim):
        """
        :params activation_fuction: Name of activation function
        :params num_GCN_blocks: Number of GCN layers of model
        :params input_dim: The number of dimensions of input
        :params output_dim: The number of dimensions of output
        :params num_source_nodes: Number of nodes in source graph
        :params num_target_nodes: Number of nodes in target graph
        :params source_feats: Source Initialized Features
        :params target_feats: Target Initialized Features
        """
        super(FA_GCN, self).__init__()
        self.num_GCN_blocks = num_GCN_blocks 
        self.input_dim = input_dim

        # GCN blocks (emb)
        self.GCNs = []
        for i in range(num_GCN_blocks):
            self.GCNs.append(GCN(activate_function, input_dim, output_dim))
            input_dim = self.GCNs[-1].output_dim
        self.GCNs = nn.ModuleList(self.GCNs)
        init_weight(self.modules(), activate_function)


    def forward(self, A_hat, input):
        """
        Do the forward
        :params A_hat: The sparse Normalized Laplacian Matrix 
        :params net: Whether forwarding graph is source or target graph
        """
        
        emb_input = input.clone()
        outputs = []
        for i in range(self.num_GCN_blocks):
            GCN_output_i = self.GCNs[i](emb_input, A_hat)
            outputs.append(GCN_output_i)
            emb_input = GCN_output_i
        return outputs
