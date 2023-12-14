import sys
import torch
from torchvision import datasets, transforms, models
from torch import nn

def batchnorm_callibration(model, train_loader, layer_name = None,
                           n_batches = 200000//512, device="cuda:0"):
    '''
    Update batchnorm statistics for layers after layer_name
    
    Parameters:

    model                   -   Pytorch model
    train_loader            -   Training dataset dataloader, Pytorch Dataloader
    n_callibration_batches  -   Number of batchnorm callibration iterations, int
    layer_name              -   Name of layer after which to update BN statistics, string or None
                                (if None updates statistics for all BN layers)
    device                  -   Device to store the model, string
    '''
    
    # switch batchnorms into the mode, in which its statistics are updated
    model.to(device).train() 

    if layer_name is not None:
        #freeze batchnorms before replaced layer
        for lname, l in model.named_modules():

            if lname == layer_name:
                break
            else:
                if (isinstance(l, nn.BatchNorm2d)):
                    l.eval()

    with torch.no_grad():            

        for i, (data, _) in enumerate(train_loader):
            _ = model(data.to(device))

            if i > n_batches:
                break
            
        del data
        torch.cuda.empty_cache()
        
    model.eval()
    return model

def get_layer_by_name(model, mname):
    '''
    Extract layer using layer name
    '''
    module = model
    mname_list = mname.split('.')
    for mname in mname_list:
        module = module._modules[mname]

    return module


def replace_conv_layer_by_name(model, mname, new_layer):
    '''
    Replace layer using layer name
    '''
    module = model
    mname_list = mname.split('.')
    for mname in mname_list[:-1]:
        module = module._modules[mname]
    module._modules[mname_list[-1]] = new_layer
