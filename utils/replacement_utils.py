import sys
sys.path.append("/beegfs/home/a.jha/scripts/stable_lr_cnn_compression/")
import torch
from torchvision import datasets, transforms, models
from torch import nn
from modeling.dcp.pruned_resnet import PrunedResNet



def load_model(architecture_name, dataset='imagenet', device = 'cuda'):
    if dataset == 'imagenet':
        
        if architecture_name == 'resnet18':
            model = models.resnet18(pretrained=True).to(device)

        elif architecture_name == 'resnet18-dcp-05':
            model = PrunedResNet(18, 0.5, num_classes=1000)
            model.load_state_dict(torch.load('../check_points/resnet18_pruned0.5.pth'))
            
        elif architecture_name == 'resnet34':
            model = models.resnet34(pretrained=True).to(device)
            
        elif architecture_name == 'resnet50':
            model = models.resnet50(pretrained=True).to(device)

        elif architecture_name == 'densenet-161':
            model = models.vgg16(pretrained=True).to(device)
        
        elif architecture_name == 'vgg16':
            model = models.vgg16(pretrained=True).to(device)

        elif architecture_name == 'alexnet':
            model = models.alexnet(pretrained=True).to(device)
            
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    model.eval()
    print('The model is loaded')
    
    return model


def batchnorm_callibration(model, train_loader, n_callibration_batches = 200000//512, 
                           layer_name = None, device="cuda:0"):
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

            if i > n_callibration_batches:
                break
            
        del data
        torch.cuda.empty_cache()
        
    model.eval()
    return model


def disable_other_layers_grads(model, layer_name):
    '''
    Disable gradients for all layers except layer_name
    '''
    for name,param in model.named_parameters():
        if '.'.join(name.split('.')[:-1]) != layer_name:
            param.requires_grad = False


def enable_all_grads(model):
    '''
    Enable gradients for all layers
    '''
    
    for name,param in model.named_parameters():
        param.requires_grad = True


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
