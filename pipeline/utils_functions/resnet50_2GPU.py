import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 6


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
        ).to('cuda:0')

        self.fc.to('cuda:0')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:0'))
        return self.fc(x.view(x.size(0), -1))


class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, device0='cuda:0', device1='cuda:0', *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size
        self.device0 = device0
        self.device1 = device1

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to(self.device1)
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to(self.device1)

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)