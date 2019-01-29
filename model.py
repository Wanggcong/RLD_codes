from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import init
import torchvision
from torchvision import models
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import math

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

###################################################################################################################

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=767, block=Bottleneck, layers=[3, 4, 6, 3], baseline=False):
        # BasicBlock
        self.baseline = baseline
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((8,1))

        # print('block.expansion:',block.expansion)
        self.embedding = 512
        self.fc1 = nn.Linear(2048, self.embedding)
        self.bn2 =nn.BatchNorm1d(self.embedding)
        self.relu2 =nn.LeakyReLU()
        self.dropout2 =nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.embedding, num_classes)

        self.embedding_2 = 256
        self.fc1_2 = nn.Linear(64, self.embedding_2)
        self.bn2_2 =nn.BatchNorm1d(self.embedding_2)
        self.relu2_2 =nn.LeakyReLU()
        self.dropout2_2 =nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(self.embedding_2, num_classes)        


        init.kaiming_normal(self.fc1.weight.data, a=0, mode='fan_out')
        init.constant(self.fc1.bias.data, 0.0)

        init.normal(self.bn2.weight.data, 1.0, 0.02)
        init.constant(self.bn2.bias.data, 0.0)

        init.normal(self.fc2.weight.data, std=0.001)
        init.constant(self.fc2.bias.data, 0.0)


        init.kaiming_normal(self.fc1_2.weight.data, a=0, mode='fan_out')
        init.constant(self.fc1_2.bias.data, 0.0)

        init.normal(self.bn2_2.weight.data, 1.0, 0.02)
        init.constant(self.bn2_2.bias.data, 0.0)

        init.normal(self.fc2_2.weight.data, std=0.001)
        init.constant(self.fc2_2.bias.data, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x1 = self.avgpool1(x)
        x1 = x1.view(x1.size(0), -1)

        x1 = self.fc1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.dropout2(x1)
        x1 = self.fc2(x1)

        if not self.baseline:
            f_stripes = self.avgpool2(x)
            ff_structure = torch.FloatTensor(x.size(0),64).zero_().cuda()
            for j in range(x.size(0)):
                sample = torch.squeeze(f_stripes[j,:,:])
                sample=sample/torch.norm(sample.transpose(1,0), 2, 1)
                score = torch.mm(sample.transpose(1,0),sample)
                score=score.view(1,score.size(0)*score.size(1))
                ff_structure[j,:] = score

            x2 = self.fc1_2(ff_structure)
            x2 = self.bn2_2(x2)
            x2 = self.relu2_2(x2)
            x2 = self.dropout2_2(x2)
            x2 = self.fc2_2(x2)

            return x1, x2
        else:
            return x1

class ResNet_test(nn.Module):

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, \
        num_features=0, norm=False, dropout=0, num_classes=767, block=Bottleneck, layers=[3, 4, 6, 3]):
        # BasicBlock
        self.inplanes = 64
        super(ResNet_test, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((8,1))

        print('block.expansion:',block.expansion)
        self.embedding = 512
        self.fc1 = nn.Linear(2048, self.embedding)
        self.bn2 =nn.BatchNorm1d(self.embedding)
        self.relu2 =nn.LeakyReLU()
        self.dropout2 =nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.embedding, num_classes)

        self.embedding_2 = 256
        self.fc1_2 = nn.Linear(64, self.embedding_2)
        self.bn2_2 =nn.BatchNorm1d(self.embedding_2)
        self.relu2_2 =nn.LeakyReLU()
        self.dropout2_2 =nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(self.embedding_2, num_classes)        


        init.kaiming_normal(self.fc1.weight.data, a=0, mode='fan_out')
        init.constant(self.fc1.bias.data, 0.0)

        init.normal(self.bn2.weight.data, 1.0, 0.02)
        init.constant(self.bn2.bias.data, 0.0)

        init.normal(self.fc2.weight.data, std=0.001)
        init.constant(self.fc2.bias.data, 0.0)


        init.kaiming_normal(self.fc1_2.weight.data, a=0, mode='fan_out')
        init.constant(self.fc1_2.bias.data, 0.0)

        init.normal(self.bn2_2.weight.data, 1.0, 0.02)
        init.constant(self.bn2_2.bias.data, 0.0)

        init.normal(self.fc2_2.weight.data, std=0.001)
        init.constant(self.fc2_2.bias.data, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #########################################

        f_stripes = self.avgpool2(x)
        ff_structure = torch.FloatTensor(x.size(0),64).zero_().cuda()
        for j in range(x.size(0)):
            sample = torch.squeeze(f_stripes[j,:,:])
            sample=sample/torch.norm(sample.transpose(1,0), 2, 1)
            score = torch.mm(sample.transpose(1,0),sample)
            score=score.view(1,score.size(0)*score.size(1))
            ff_structure[j,:] = score

        x2 = ff_structure

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x,x2

def resnet50(num_classes=767, baseline=False, **kwargs):
    pretrained=True
    print(num_classes)
    model = ResNet(num_classes=num_classes, block=Bottleneck, layers=[3, 4, 6, 3], baseline=baseline)
    if pretrained:
        print('***************************wgc will succeed! load model!********************************8')
        updated_params = model_zoo.load_url(model_urls['resnet50'],'./')
        updated_params.pop('fc.weight')
        updated_params.pop('fc.bias')
        new_params = model.state_dict()
        new_params.update(updated_params)
        model.load_state_dict(new_params)
    if False:
        print('***************************wgc will succeed! no pretrained********************************8')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

        init.kaiming_normal(model.fc1.weight, mode='fan_out')
        init.constant(model.fc1.bias, 0)
        init.constant(model.bn2.weight, 1)
        init.constant(model.bn2.bias, 0)

        init.normal(model.fc2.weight, std=0.001)
        init.constant(model.fc2.bias, 0)
    return model

def resnet50_test(num_classes=767, **kwargs):
    pretrained=True
    model = ResNet_test(Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes,**kwargs)
    if pretrained:
    # if False:
        print('***************************wgc will succeed! load model!********************************8')
        updated_params = model_zoo.load_url(model_urls['resnet50'],'./')
        updated_params.pop('fc.weight')
        updated_params.pop('fc.bias')

        new_params = model.state_dict()
        new_params.update(updated_params)
        model.load_state_dict(new_params)
    # else:
    if False:
        print('***************************wgc will succeed! no pretrained********************************8')
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

        init.kaiming_normal(model.fc1.weight, mode='fan_out')
        init.constant(model.fc1.bias, 0)
        init.constant(model.bn2.weight, 1)
        init.constant(model.bn2.bias, 0)

        init.normal(model.fc2.weight, std=0.001)
        init.constant(model.fc2.bias, 0)
    return model
