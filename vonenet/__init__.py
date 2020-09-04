import torch
import torch.nn as nn
import os
import requests

from .vonenet import VOneNet
from torch.nn import Module

FILE_WEIGHTS = {'alexnet': 'vonealexnet_e70.pth.tar', 'resnet50': 'voneresnet50_e70.pth.tar',
                'resnet50_at': 'voneresnet50_at_e96.pth.tar', 'cornets': 'vonecornets_e70.pth.tar'}


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def get_model(model_arch=None, pretrained=True, map_location='cpu', **kwargs):
    """
    Returns a VOneNet model.
    Select pretrained=True for returning one of the 3 pretrained models.
    model_arch: string with identifier to choose the architecture of the back-end (resnet50, cornets, alexnet)
    """
    if pretrained:
        url = f'https://vonenet-models.s3.us-east-2.amazonaws.com/{FILE_WEIGHTS[model_arch.lower()]}'
        home_dir = os.environ['HOME']
        vonenet_dir = os.path.join(home_dir, '.vonenet')
        weightsdir_path = os.path.join(vonenet_dir, FILE_WEIGHTS[model_arch.lower()])
        if not os.path.exists(vonenet_dir):
            os.makedirs(vonenet_dir)
        if not os.path.exists(weightsdir_path):
            print('Downloading model weights to ', weightsdir_path)
            r = requests.get(url, allow_redirects=True)
            open(weightsdir_path, 'wb').write(r.content)

        ckpt_data = torch.load(weightsdir_path, map_location=map_location)

        stride = ckpt_data['flags']['stride']
        simple_channels = ckpt_data['flags']['simple_channels']
        complex_channels = ckpt_data['flags']['complex_channels']
        k_exc = ckpt_data['flags']['k_exc']

        noise_mode = ckpt_data['flags']['noise_mode']
        noise_scale = ckpt_data['flags']['noise_scale']
        noise_level = ckpt_data['flags']['noise_level']

        if model_arch.lower() == 'resnet50_at':
            model_id = 'resnet50'
        else:
            model_id = model_arch

        model = globals()[f'VOneNet'](model_arch=model_id, stride=stride, k_exc=k_exc,
                                      simple_channels=simple_channels, complex_channels=complex_channels,
                                      noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level)

        if model_arch.lower() == 'resnet50_at':
            ckpt_data['state_dict'].pop('vone_block.div_u.weight')
            ckpt_data['state_dict'].pop('vone_block.div_t.weight')
            model.load_state_dict(ckpt_data['state_dict'])
        else:
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module

        model = nn.DataParallel(model)
    else:
        model = globals()[f'VOneNet'](model_arch=model_arch, **kwargs)
        nn.DataParallel(model)
    return model

