
# VOneNet: CNNs with a Primary Visual Cortex Front-End

A family of biologically-inspired Convolutional Neural Networks (CNNs). VOneNets have the following features:
- Fixed-weight neural network model of the primate primary visual cortex (V1) as the front-end.
- Robust to image perturbations
- Brain-mapped
- Flexible: can be adapted to different back-end architectures

[read more...](#longer-motivation)

## Available Models
*(Click on model names to download the weights of ImageNet-trained models. Alternatively, you can use the function get_model in the vonenet package to download the weights.)*

| Name     | Description                                                              |
| -------- | ------------------------------------------------------------------------ |
| [VOneResNet50](https://vonenet-models.s3.us-east-2.amazonaws.com/voneresnet50_e70.pth.tar) | Our best performing VOneNet with a ResNet50 back-end |
| [VOneCORnet-S](https://vonenet-models.s3.us-east-2.amazonaws.com/vonecornets_e70.pth.tar) | VOneNet with a recurrent neural network back-end based on the CORnet-S |
| [VOneAlexNet](https://vonenet-models.s3.us-east-2.amazonaws.com/vonealexnet_e70.pth.tar) | VOneNet with a back-end based on AlexNet         |


## Quick Start

VOneNets was trained with images normalized with mean=[0.5,0.5,0.5] and std=[0.5,0.5,0.5]

More information coming soon...

## Adversarial Evaluation
For evaluating adversarial robustness, we use the [adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox) v1.5.0, which can be installed with
```
pip install adversarial-robustness-toolbox==1.5.0
```

To evaluate the VOneResNet50 with projected gradient descent (PGD) using $L_{\infty}_$ $\eps=1/255$, 64 iterations and k=10 gradient samples per step on 100 imagenet validation images, use:
```
python evaluate.py --in_path=PATH_TO_IMAGENET_VAL --DS_factor=500 --model_arch='resnet50' --norm='inf' --eps=0.0039215686 --iterations=64 --k=10
```

## Longer Motivation

Current state-of-the-art object recognition models are largely based on convolutional neural network (CNN) architectures, which are loosely inspired by the primate visual system. However, these CNNs can be fooled by imperceptibly small, explicitly crafted perturbations, and struggle to recognize objects in corrupted images that are easily recognized by humans. Recently, we observed that CNN models with a neural hidden layer that better matches primate primary visual cortex (V1) are also more robust to adversarial attacks. Inspired by this observation, we developed VOneNets, a new class of hybrid CNN vision models. Each VOneNet contains a fixed weight neural network front-end that simulates primate V1, called the VOneBlock, followed by a neural network back-end adapted from current CNN vision models. The VOneBlock is based on a classical neuroscientific model of V1: the linear-nonlinear-Poisson model, consisting of a biologically-constrained Gabor filter bank, simple and complex cell nonlinearities, and a V1 neuronal stochasticity generator. After training, VOneNets retain high ImageNet performance, but each is substantially more robust, outperforming the base CNNs and state-of-the-art methods by 18% and 3%, respectively, on a conglomerate benchmark of perturbations comprised of white box adversarial attacks and common image corruptions. Additionally, all components of the VOneBlock work in synergy to improve robustness. 
Read more: [Dapello\*, Marques\*, et al. (biorxiv, 2020)](https://doi.org/10.1101/2020.06.16.154542)



## Requirements

- Python 3.6+
- PyTorch 0.4.1+
- numpy
- pandas
- tqdm
- scipy
- adversarial-robustness-toolkit 1.5.0+

## Citation

Dapello, J., Marques, T., Schrimpf, M., Geiger, F., Cox, D.D., DiCarlo, J.J. (2020) Simulating a Primary Visual Cortex at the Front of CNNs Improves Robustness to Image Perturbations. *biorxiv.* doi.org/10.1101/2020.06.16.154542


## License

GNU GPL 3+


## FAQ

Q: When I run the defense and compute the backwards pass through the network, pytorch says it can't compute the gradient.
A: We have used inplace ReLU's in the VOneBlock. Simply changing the code to `inplace=False` will resolve the issue; this can be done in the attack code, ie:
```
import torch.nn as nn
from vonenet import get_model

model = get_model(model_arch='resnet50', pretrained=True)
model.module.vone_block.simple = nn.ReLU(inplace=False)
model.module.vone_block.noise = nn.ReLU(inplace=False)
```
