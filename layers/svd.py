import numpy as np
import torch
from torch import nn



class SVD_conv_layer(torch.nn.Module):
    def __init__(self, layer, rank_selection, rank):
        super(SVD_conv_layer, self).__init__()
        
        self.cin = layer.in_channels
        self.cout = layer.out_channels
        self.padding = layer.padding
        self.stride = layer.stride[0]
        self.kernel_size = layer.kernel_size
        self.is_bias = layer.bias is not None 
        if self.is_bias:
            self.bias = layer.bias
        
        if rank is None or type(rank) is not int:
            raise AttributeError('Rank should be an integer number')
        else:
            self.rank = rank
                
        
        self.svd_decomposition = self.__replace__(layer)
        
    def __replace__(self, layer):
        """ Gets a conv layer and a target rank, 
            returns a nn.Sequential object with
            each layer representing a decomposed factor"""
        
        U, S, Vt = np.linalg.svd(layer.weight[:, :, 0, 0].data.numpy(), full_matrices=False)
            
        w0 = np.dot(np.diag(np.sqrt(S[0:self.rank])),Vt[0:self.rank, :])
        w1 = np.dot(U[:, 0:self.rank], np.diag(np.sqrt(S[0:self.rank])))


        new_layers = [
            nn.Conv2d(in_channels=self.cin, bias=False, 
                      out_channels=self.rank, kernel_size = self.kernel_size,
                      stride = self.stride),

            nn.Conv2d(in_channels = self.rank, bias=self.is_bias,
                      out_channels = self.cout, kernel_size = self.kernel_size,
                      padding = self.padding, stride = self.stride)]

        new_kernels = [torch.FloatTensor(w0[:,:, np.newaxis, np.newaxis]).contiguous(),
                       torch.FloatTensor(w1[:,:, np.newaxis, np.newaxis]).contiguous()]
        
        with torch.no_grad():
            for i in range(len(new_kernels)):
                new_layers[i].weight = nn.Parameter(new_kernels[i].cpu())
                if i == len(new_kernels)-1 and self.is_bias:
                    new_layers[i].bias = nn.Parameter(self.bias)
        
        return nn.Sequential(*new_layers)
    
    def forward(self, x):
        out = self.svd_decomposition(x)
        return out