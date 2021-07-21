import os, argparse, time, subprocess, io, shlex

import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

""" 
For attacking VOneNets, this code uses the adversarial-robustness-toolbox, v1.5.0+ (art). 
Install with:
    pip install adversarial-robustness-toolbox==1.5.0+
"""
from art.estimators.classification import PyTorchClassifier, EnsembleClassifier
from art.attacks.evasion import ProjectedGradientDescent

from vonenet import get_model

parser = argparse.ArgumentParser(description='Adversarial ImageNet Validation')

parser.add_argument('--in_path', required=True,
                    help='path to ImageNet folder that contains val folder')
parser.add_argument('--batch_size', default=128, type=int,
                    help='size of batch for validation')
parser.add_argument('--workers', default=20,
                    help='number of data loading workers')
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use; 0 if you want to run on CPU')
parser.add_argument('--model_arch', choices=['alexnet', 'resnet50', 'cornets'], default='resnet50',
                    help='back-end model architecture to load')
parser.add_argument('--DS_factor', default=10, type=int,
                    help='downsample ImageNet val images -- take every n (ie val[::DS_factor])')
parser.add_argument('--norm', default='inf', type=str,
                    help='L norm of adversary')
parser.add_argument('--eps', default=1/255, type=float,
                    help='bound of norm for attack')
parser.add_argument('--step_size', default=1/4, type=float,
                    help='PGD step size, relative to epsilon')
parser.add_argument('--iterations', default=64, type=int,
                    help='how many iterations of PGD to perform')
parser.add_argument('--k', default=8, type=int,
                    help='how many samples of the gradient to take per step')
parser.add_argument('--restarts', default=0, type=int,
                    help='how many times to restart PGD from a random location')

FLAGS, FIRE_FLAGS = parser.parse_known_args()

NORM_MAP = {
    'inf' : np.inf,
    '2' : 2,
    '1' : 1
}


def set_gpus(n=2):
    """
    Finds all GPUs on the system and restricts to n of them that have the most
    free memory.
    """
    if n > 0:
        gpus = subprocess.run(shlex.split(
            'nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,nounits'), check=True,
            stdout=subprocess.PIPE).stdout
        gpus = pd.read_csv(io.BytesIO(gpus), sep=', ', engine='python')
        gpus = gpus[gpus['memory.total [MiB]'] > 10000]  # only above 10 GB
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            visible = [int(i)
                       for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
            gpus = gpus[gpus['index'].isin(visible)]
        gpus = gpus.sort_values(by='memory.free [MiB]', ascending=False)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # making sure GPUs are numbered the same way as in nvidia_smi
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            [str(i) for i in gpus['index'].iloc[:n]])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


set_gpus(FLAGS.ngpus)


device = torch.device("cuda" if FLAGS.ngpus > 0 else "cpu")


def adversarial_val():
    model = get_model(model_arch=FLAGS.model_arch, pretrained=True)
    if hasattr(model.module, 'vone_block'):
        print('replacing inplace ReLUs on VOneBlock')
        model.module.vone_block.simple = nn.ReLU(inplace=False)
        model.module.vone_block.noise = nn.ReLU(inplace=False)

    if FLAGS.ngpus == 0:
        print('Running on CPU')
    if FLAGS.ngpus > 0 and torch.cuda.device_count() > 1:
        print('Running on multiple GPUs')
        model = model.to(device)
    elif FLAGS.ngpus > 0 and torch.cuda.device_count() is 1:
        print('Running on single GPU')
        model = model.to(device)
    else:
        print('No GPU detected!')
        model = model.module

    validator = AdvImageNetVal(model)
    record = validator()

    print(f'PGD l{FLAGS.norm} eps={FLAGS.eps}, ' 
         +f'{FLAGS.iterations} iterations with '
         +f'{FLAGS.k} averaged gradients')
    print(f"accuracy (top1):{record['top1']}")
    print(f"accuracy (top5):{record['top5']}")
    return


class AdvImageNetVal(object):

    def __init__(self, model):
        self.name = 'AdversarialVal'
        self.model = model
        self.data_loader = self.data()
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.loss = self.loss.to(device)

    def data(self):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(FLAGS.in_path, 'val'),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
            ]))
        dataset.imgs = dataset.imgs[::FLAGS.DS_factor]
        dataset.samples = dataset.samples[::FLAGS.DS_factor]
        dataset.targets = dataset.targets[::FLAGS.DS_factor]
        print(f'Evaluating on {dataset.__len__} images')
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=FLAGS.batch_size,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers,
                                                  pin_memory=True)

        return data_loader

    def __call__(self):
        self.model.eval()
        start = time.time()
        record = {'loss': 0, 'top1': 0, 'top5': 0}
        mean = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))
        std = np.array([0.5, 0.5, 0.5]).reshape((3, 1, 1))

        classifier = PyTorchClassifier(
            model=self.model,
            clip_values=(0, 1),
            preprocessing=(mean, std),
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.SGD(self.model.parameters(), lr=0.01),
            input_shape=(3, 224, 224),
            nb_classes=1000,
        )

        attack = ProjectedGradientDescent(
            estimator=Ensemblize(classifier, k=FLAGS.k), 
            norm=NORM_MAP[FLAGS.norm],
            max_iter=FLAGS.iterations, 
            eps=FLAGS.eps, 
            eps_step=FLAGS.eps*FLAGS.step_size,
            num_random_init=FLAGS.restarts,
            targeted=False,
        )

        for (inp, target) in tqdm.tqdm(self.data_loader, desc=self.name):
            inp = inp.detach().cpu().numpy().astype(np.float32)
            target = target.detach().cpu().numpy().astype(np.float32)
            adv_input = attack.generate(x=inp, y=target)
            output = classifier.predict(adv_input)

            p1, p5 = accuracy(torch.tensor(output), torch.tensor(target), topk=(1, 5))
            record['top1'] += p1
            record['top5'] += p5

        for key in record:
            record[key] /= len(self.data_loader.dataset.samples)
        record['dur'] = (time.time() - start) / len(self.data_loader)

        return record

# create self-ensemble of k networks for computing attack gradients
def Ensemblize(classifier, k=8):
    return EnsembleClassifier(
        classifiers=[classifier for _ in range(k)],
        channels_first=True,
        clip_values=(0, 1)
    )

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res


if __name__ == '__main__':
    adversarial_val()
