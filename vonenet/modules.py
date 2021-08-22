import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .utils import gabor_kernel
from .params import generate_gabor_param

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):
    def forward(self, x):
        return x


class GFB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)

        # Param instantiations
        self.weight = torch.zeros(
            (out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = torch.randint(0, self.in_channels,
                                       (self.out_channels,))
        for i in range(self.out_channels):
            self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i],
                                                             sigma_x=sigx[i],
                                                             sigma_y=sigy[i],
                                                             theta=theta[i],
                                                             offset=phase[i],
                                                             ks=
                                                             self.kernel_size[
                                                                 0])
        self.weight = nn.Parameter(self.weight, requires_grad=False)


class VOneBlock(nn.Module):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25.0, noise_mode=None, noise_scale=1.0,
                 poisson_scale=1.0,
                 noise_level=1.0, is_fix_noise=False, noise_batch_size=None,
                 noise_seed=None, simple_channels=128,
                 complex_channels=128, ksize=25, stride=4,
                 input_size=224):
        super().__init__()

        self.in_channels = 3

        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level, poisson_scale)
        if is_fix_noise:
            self.fix_noise(batch_size=noise_batch_size, seed=noise_seed)
        else:
            self.fixed_noise = None

        self.simple_conv_q0 = GFB(self.in_channels, self.out_channels, ksize,
                                  stride)
        self.simple_conv_q1 = GFB(self.in_channels, self.out_channels, ksize,
                                  stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta,
                                       sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta,
                                       sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)

        self.simple = nn.ReLU(inplace=True)
        self.complex = Identity()
        self.gabors = Identity()
        # self.noise = nn.ReLU(inplace=True)
        self.noise = Identity()
        self.output = Identity()

    def forward(self, x):
        # Gabor activations [Batch, out_channels, H/stride, W/stride]
        x = self.gabors_f(x)
        # Noise [Batch, out_channels, H/stride, W/stride]
        x = self.noise_f(x)
        # V1 Block output: (Batch, out_channels, H/stride, W/stride)
        x = self.output(x)
        return x

    def gabors_f(self, x):
        s_q0 = self.simple_conv_q0(x)
        s_q1 = self.simple_conv_q1(x)
        c = self.complex(torch.sqrt(s_q0[:, self.simple_channels:, :, :] ** 2 +
                                    s_q1[:, self.simple_channels:, :,
                                    :] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, 0:self.simple_channels, :, :])
        return self.gabors(self.k_exc * torch.cat((s, c), 1))

    def noise_f(self, x):
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale  # slope
            x += self.noise_level  # intercept
            if self.fixed_noise is not None:
                print(self.fixed_noise.shape,
                      torch.sqrt(F.relu(x.clone())).shape)
                x += self.fixed_noise * torch.sqrt(F.relu(x.clone()))  # +
                # torch.as_tensor(eps)
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x),
                                                       scale=1).rsample() * \
                     torch.sqrt(F.relu(x.clone()) + torch.as_tensor(eps)) * \
                     self.poisson_scale
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += torch.distributions.normal.Normal(torch.zeros_like(x),
                                                       scale=1).rsample() * \
                     self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1.0, noise_level=1.0,
                       poisson_scale=1.0):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level
        self.poisson_scale = poisson_scale

    def fix_noise(self, batch_size=256, seed=None):
        noise_mean = torch.zeros(batch_size, self.out_channels,
                                 int(self.input_size / self.stride),
                                 int(self.input_size / self.stride))
        if seed:
            torch.manual_seed(seed)
        if self.noise_mode:
            self.fixed_noise = torch.distributions.normal.Normal(noise_mean,
                                                                 scale=1).rsample().to(
                device)

    def unfix_noise(self):
        self.fixed_noise = None


def Create_VOneBlock(sf_corr=0.75, sf_max=11.3, sf_min=0, rand_param=False,
                     gabor_seed=0, simple_channels=256, complex_channels=256,
                     noise_mode='neuronal', noise_scale=0.286,
                     noise_level=0.071,
                     poisson_scale=1.0, is_fix_noise=False,
                     noise_batch_size=None, noise_seed=None,
                     k_exc=23.5, image_size=64,
                     visual_degrees=2, ksize=25, stride=2):

    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed,
                                                    rand_param, sf_corr, sf_max,
                                                    sf_min)

    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta / 180 * np.pi
    phase = phase / 180 * np.pi

    return VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy,
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


class VOneBlockEnsemble(nn.Module):
    def __init__(self, block_dict, sf_corr=0.75, rand_param=False,
                 noise_scale=0.286, noise_level=0.071, is_fix_noise=False,
                 noise_batch_size=None, noise_seed=None,
                 k_exc=23.5, image_size=64,
                 visual_degrees=2, ksize=25, stride=2):

        super(VOneBlockEnsemble, self).__init__()

        self.block_dict = block_dict
        self.models = nn.ModuleList()

        for block_type in self.block_dict.keys():
            block_params = self.block_dict[block_type]
            model = Create_VOneBlock(sf_corr=sf_corr,
                                     sf_max=block_params['sf_max'],
                                     sf_min=block_params['sf_min'],
                                     rand_param=rand_param,
                                     gabor_seed=block_params[
                                         'gabor_seed'],
                                     simple_channels=block_params[
                                         'simple_channels'],
                                     complex_channels=block_params[
                                         'complex_channels'],
                                     noise_mode=block_params[
                                         'noise_mode'],
                                     noise_scale=noise_scale,
                                     noise_level=noise_level,
                                     poisson_scale=block_params[
                                         'poisson_scale'],
                                     is_fix_noise=is_fix_noise,
                                     noise_batch_size=noise_batch_size,
                                     noise_seed=noise_seed,
                                     k_exc=k_exc, image_size=image_size,
                                     visual_degrees=visual_degrees,
                                     ksize=ksize,
                                     stride=stride)

            self.models.append(model)

    def forward(self, x):

        out = self.models[0](x)

        for i in range(1, len(self.block_dict)):
            out += self.models[i](x)

        return out