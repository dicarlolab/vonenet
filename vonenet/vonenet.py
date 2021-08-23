from collections import OrderedDict
from torch import nn
from .modules import VOneBlock, VOneBlockEnsemble, VOneBlockWeightedEnsemble
from .back_ends import ResNetBackEnd, Bottleneck, AlexNetBackEnd, \
    CORnetSBackEnd, BasicBlock
from .params import generate_gabor_param

import numpy as np


def VOneNet(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07,
            poisson_scale=1.0, is_fix_noise=False, noise_batch_size=None,
            noise_seed=None, k_exc=25, model_arch='resnet50', image_size=224,
            num_classes=200, visual_degrees=8, ksize=25, stride=4):
    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed,
                                                    rand_param, sf_corr, sf_max,
                                                    sf_min)

    gabor_params = {'simple_channels': simple_channels,
                    'complex_channels': complex_channels,
                    'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max,
                    'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(),
                    'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize,
                   'stride': stride}

    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta / 180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy,
                           phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode,
                           noise_scale=noise_scale, noise_level=noise_level,
                           poisson_scale=poisson_scale,
                           is_fix_noise=is_fix_noise,
                           noise_batch_size=noise_batch_size,
                           noise_seed=noise_seed,
                           simple_channels=simple_channels,
                           complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size)

    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1,
                               bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out',
                                nonlinearity='relu')

        if model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            model_back_end = ResNetBackEnd(block=Bottleneck,
                                           layers=[3, 4, 6, 3],
                                           num_classes=num_classes)
        elif model_arch.lower() == 'resnet18':
            print('Model: ', 'VOneResnet18')
            model_back_end = ResNetBackEnd(block=BasicBlock,
                                           layers=[2, 2, 2, 2],
                                           num_classes=num_classes)
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd(num_classes=num_classes)
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = CORnetSBackEnd(num_classes=num_classes)

        model = nn.Sequential(OrderedDict([
            ('vone_block', vone_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    return model


def VOneNetEnsemble(
        block_dict, weighted=False, sf_corr=0.75, rand_param=False,
        noise_scale=0.286, noise_level=0.071, is_fix_noise=False,
        noise_batch_size=None, noise_seed=None, out_channels=512,
        k_exc=23.5, model_arch='resnet18', image_size=64,
        num_classes=200, visual_degrees=2, ksize=25, stride=2):
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize,
                   'stride': stride}

    if weighted:
        vone_block = VOneBlockWeightedEnsemble(block_dict, sf_corr=sf_corr,
                                               rand_param=rand_param,
                                               noise_scale=noise_scale,
                                               noise_level=noise_level,
                                               is_fix_noise=is_fix_noise,
                                               noise_batch_size=noise_batch_size,
                                               noise_seed=noise_seed,
                                               k_exc=k_exc,
                                               image_size=image_size,
                                               visual_degrees=visual_degrees,
                                               ksize=ksize,
                                               stride=stride)
    else:
        vone_block = VOneBlockEnsemble(block_dict, sf_corr=sf_corr,
                                       rand_param=rand_param,
                                       noise_scale=noise_scale,
                                       noise_level=noise_level,
                                       is_fix_noise=is_fix_noise,
                                       noise_batch_size=noise_batch_size,
                                       noise_seed=noise_seed,
                                       k_exc=k_exc, image_size=image_size,
                                       visual_degrees=visual_degrees,
                                       ksize=ksize,
                                       stride=stride)

    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1,
                               bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out',
                                nonlinearity='relu')

        if model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50Ensemble')
            model_back_end = ResNetBackEnd(block=Bottleneck,
                                           layers=[3, 4, 6, 3],
                                           num_classes=num_classes)
        elif model_arch.lower() == 'resnet18':
            print('Model: ', 'VOneResnet18Ensemble')
            model_back_end = ResNetBackEnd(block=BasicBlock,
                                           layers=[2, 2, 2, 2],
                                           num_classes=num_classes)
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNetEnsemble')
            model_back_end = AlexNetBackEnd(num_classes=num_classes)
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-SEnsemble')
            model_back_end = CORnetSBackEnd(num_classes=num_classes)

        model = nn.Sequential(OrderedDict([
            ('vone_block', vone_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
        print('Model: ', 'VOneNet')
        model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.arch_params = arch_params

    return model
