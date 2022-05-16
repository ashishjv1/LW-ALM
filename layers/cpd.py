import torch
import torch.nn as nn
import numpy as np
import scipy.io

class CPD3_layer(torch.nn.Module):
    
    def __init__(self, layer, factors, cpd_type = 'epc'):
        
        super(CPD3_layer, self).__init__()
        self.factors = factors
        
        if type(self.factors) is str:
            self.rank = int(self.factors.split('_')[-1][:-4][1:])
        elif type(self.factors) is list:
            self.rank = self.factors[0].shape[1]
            
        self.cin = layer.in_channels
        self.cout = layer.out_channels
        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.kernel_size = layer.kernel_size
        self.is_bias = layer.bias is not None
        if self.is_bias:
            self.bias = layer.bias
        
        self.cpd_type = cpd_type
        self.cp_decomposition = self.__replace__()
        
    def __replace__(self):
        
        if type(self.factors) is str:
            mat = scipy.io.loadmat(self.factors)
            if self.cpd_type == 'epc':
                cpd_key = 'Uepc'
            elif self.cpd_type == 'cp':
                cpd_key = 'Ucp'
            else:
                raise NotImplementedError
            U = mat[cpd_key][0]
        elif type(self.factors) is list:
            U = self.factors
            
       
        # expected order of factors is the following [k*k,rank],[cin,rank],[cout,rank]
        try:
            f_z = np.array(U[0])
            f_cout = np.array(U[1])
            f_cin = np.array(U[2])
            new_layers = [
                nn.Conv2d(in_channels=self.cin, bias=False, 
                          out_channels=self.rank, kernel_size = (1, 1)),
                nn.Conv2d(in_channels = self.rank, 
                                        out_channels=self.rank,
                                        kernel_size = self.kernel_size,
                                        groups = self.rank, 
                                        padding = self.padding,
                                        dilation = self.dilation,
                                        bias=False,
                                        stride = self.stride),
                nn.Conv2d(in_channels = self.rank,
                                        bias=self.is_bias,
                                        out_channels = self.cout, 
                                        kernel_size = (1, 1))
            ]

            new_kernels = [ torch.FloatTensor(np.reshape(f_cin.T, [self.rank, self.cin, 1, 1])),
                            torch.FloatTensor(np.reshape(f_z.T, [self.rank, 1, *self.kernel_size])),
                            torch.FloatTensor(np.reshape(f_cout, [self.cout, self.rank, 1, 1]))#.contiguous()
            ]
        except:
            raise ValueError('Something wrong with order of cpd factors')
        with torch.no_grad():
            for i in range(len(new_kernels)):
                new_layers[i].weight = nn.Parameter(new_kernels[i])
                if i == len(new_kernels)-1 and self.is_bias:
                    new_layers[i].bias = nn.Parameter(self.bias)
        
        return nn.Sequential(*new_layers)
    
    def forward(self, x):
        out = self.cp_decomposition(x)
        return out

    
# need extra tests
class CPD4_layer(torch.nn.Module):
    
    def __init__(self, layer, pretrained, cpd_type = 'epc'):
        
        super(CPD4_layer, self).__init__()
        self.pretrained = pretrained
        self.rank = int(self.pretrained.split('_')[-1][:-4][1:])
        self.cin = layer.in_channels
        self.cout = layer.out_channels
        self.padding = layer.padding
        self.stride = layer.stride
        self.dilation = layer.dilation
        self.kernel_size = layer.kernel_size
        self.is_bias = layer.bias is not None
        if self.is_bias:
            self.bias = layer.bias
        
        self.cpd_type = cpd_type
        self.cp_decomposition = self.__replace__()
        
    def __replace__(self):
        
        mat = scipy.io.loadmat(self.pretrained)
        if self.cpd_type == 'epc':
            cpd_key = 'Uepc'
        elif self.cpd_type == 'cp':
            cpd_key = 'Ucp'
        else:
            raise NotImplementedError
        U = mat[cpd_key][0]
    
        # expected order of factors is the following [k,rank],[k,rank],[cout,rank],[cin,rank] ???
        # need to check new version of cpd-epc-mat code
        f_cout = np.array(U[2])
        f_cin = np.array(U[3])
        f_h = np.array(U[0])
        f_w = np.array(U[1])
        
        new_layers = [
            nn.Conv2d(in_channels=self.cin, bias=False, 
                      out_channels=self.rank, kernel_size = (1, 1)),
            nn.Conv2d(in_channels = self.rank, 
                                    out_channels=self.rank,
                                    kernel_size = (self.kernel_size[0], 1),
                                    groups = self.rank, 
                                    padding = (self.padding[0],0),
                                    stride = (self.stride[0], 1)),
                          
            nn.Conv2d(in_channels = self.rank,
                                    out_channels=self.rank,
                                    kernel_size = (1, self.kernel_size[1]),
                                    groups = self.rank,
                                    padding = (0, self.padding[1]),
                                    stride = (1, self.stride[1])),
            nn.Conv2d(in_channels = self.rank,
                                    bias=self.is_bias,
                                    out_channels = self.cout, 
                                    kernel_size = (1, 1))
        ]
            
        new_kernels = [ torch.FloatTensor(np.reshape(f_cin.T, [self.rank, self.cin, 1, 1])),
                        torch.FloatTensor(np.reshape(f_h.T, (self.rank, 1, self.kernel_size[0], 1))),
                        torch.FloatTensor(np.reshape(f_w.T, [self.rank, 1, 1, self.kernel_size[1]])),
                        torch.FloatTensor(np.reshape(f_cout, [self.cout, self.rank, 1, 1]))#.contiguous()
        ]
        with torch.no_grad():
            for i in range(len(new_kernels)):
                new_layers[i].weight = nn.Parameter(new_kernels[i])
                if i == len(new_kernels)-1 and self.is_bias:
                    new_layers[i].bias = nn.Parameter(self.bias)
        
        return nn.Sequential(*new_layers)
    
    def forward(self, x):
        out = self.cp_decomposition(x)
        return out