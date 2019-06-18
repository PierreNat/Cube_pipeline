import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 6
device0 = torch.device('cuda:0')
device1 = torch.device('cuda:0')


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
        ).to(device0)

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to(device1)

        self.fc.to(device1)

    def forward(self, x):
        x = self.seq2(self.seq1(x).to(device1))
        return self.fc(x.view(x.size(0), -1))


class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, device0='cuda:0', device1='cuda:0', *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size
        self.device0 = device0
        self.device1 = device1

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits) #take the first mini batch
        s_prev = self.seq1(s_next) #go through the sequ1 with the first mini batch and the result goes in s_prev assigned with device 1
        print("s_prev batch 0 device out of seq1 {}".format(s_prev.device))
        s_prev = s_prev.to(device1)
        print("s_prev device is {}".format(s_prev.device))
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A)
            s_prev = self.seq1(s_next)
            print("s_prev batch 1 device out of seq1 {}".format(s_prev.device))
            s_prev = s_prev.to(device1)

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)