# from the tutorial https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

# -*- coding: utf-8 -*-
"""
Model Parallel Best Practices
*************************************************************
**Author**: `Shen Li <https://mrshenli.github.io/>`_
Data parallel and model parallel are widely-used in distributed training
techniques. Previous posts have explained how to use
`DataParallel <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html>`_
to train a neural network on multiple GPUs. ``DataParallel`` replicates the
same model to all GPUs, where each GPU consumes a different partition of the
input data. Although it can significantly accelerate the training process, it
does not work for some use cases where the model is too large to fit into a
single GPU. This post shows how to solve that problem by using model parallel
and also shares some insights on how to speed up model parallel training.
The high-level idea of model parallel is to place different sub-networks of a
model onto different devices, and implement the ``forward`` method accordingly
to move intermediate outputs across devices. As only part of a model operates
on any individual device, a set of devices can collectively serve a larger
model. In this post, we will not try to construct huge models and squeeze them
into a limited number of GPUs. Instead, this post focuses on showing the idea
of model parallel. It is up to the readers to apply the ideas to real-world
applications.
**Recommended Reading:**
-  https://pytorch.org/ For installation instructions
-  :doc:`/beginner/blitz/data_parallel_tutorial` Single-Machine Data Parallel
-  :doc:`/intermediate/ddp_tutorial` Combine Distributed Data Parallel and Model Parallel
"""

######################################################################
# Apply Model Parallel to Existing Modules
# =======================
#
# It is also possible to run an existing single-GPU module on multiple GPUs
# with just a few lines of changes. The code below shows how to decompose
# ``torchvision.models.reset50()`` to two GPUs. The idea is to inherit from
# the existing ``ResNet`` module, and split the layers to two GPUs during
# construction. Then, override the ``forward`` method to stitch two
# sub-networks by moving the intermediate outputs accordingly.
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000


class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))


######################################################################
# The above implementation solves the problem for cases where the model is too
# large to fit into a single GPU. However, you might have already noticed that
# it will be slower than running it on a single GPU if your model fits. It is
# because, at any point in time, only one of the two GPUs are working, while
# the other one is sitting there doing nothing. The performance further
# deteriorates as the intermediate outputs need to be copied from ``cuda:0`` to
# ``cuda:1`` between ``layer2`` and ``layer3``.
#
# Let us run an experiment to get a more quantitative view of the execution
# time. In this experiment, we train ``ModelParallelResNet50`` and the existing
# ``torchvision.models.reset50()`` by running random inputs and labels through
# them. After the training, the models will not produce any useful predictions,
# but we can get a reasonable understanding of the execution times.


import torchvision.models as models

num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
        .random_(0, num_classes) \
        .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
            .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))

        # run backward pass
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()


######################################################################
# The ``train(model)`` method above uses ``nn.MSELoss`` as the loss function,
# and ``optim.SGD`` as the optimizer. It mimics training on ``128 X 128``
# images which are organized into 3 batches where each batch contains 120
# images. Then, we use ``timeit`` to run the ``train(model)`` method 10 times
# and plot the execution times with standard deviations.


import matplotlib.pyplot as plt

plt.switch_backend('Agg')
import numpy as np
import timeit

num_repeat = 10

stmt = "train(model)"

setup = "model = ModelParallelResNet50()"
# globals arg is only available in Python 3. In Python 2, use the following
# import __builtin__
# __builtin__.__dict__.update(locals())
mp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


plot([mp_mean, rn_mean],
     [mp_std, rn_std],
     ['Model Parallel', 'Single GPU'],
     'mp_vs_rn.png')

