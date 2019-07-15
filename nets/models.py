import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
import torch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        net = models.inception_v3(pretrained=False, aux_logits=False)
        net.load_state_dict(torch.load('fonts_inception_v3.pth', map_location=torch.device('cpu')), strict=False)
        num_fc = net.fc.in_features
        net.fc = nn.Linear(num_fc, num_classes)
        net.fc = Identity()
        self.net = net
        # # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc2 = nn.Linear(num_fc, num_classes)

        # self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        # self.prelu1_1 = nn.PReLU()
        # self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        # self.prelu1_2 = nn.PReLU()
        #
        # self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        # self.prelu2_1 = nn.PReLU()
        # self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        # self.prelu2_2 = nn.PReLU()
        #
        # self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        # self.prelu3_1 = nn.PReLU()
        # self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        # self.prelu3_2 = nn.PReLU()
        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(2048, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)


    def forward(self, x):
        x = self.net(x)
        # x = self.fc1(x)
        # x = self.avgpool(x)
        # y = self.fc2(x)
        # x = self.prelu1_1(self.conv1_1(x))
        # x = self.prelu1_2(self.conv1_2(x))
        # x = F.max_pool2d(x, 2)
        #----------------------------------------
        # x = self.prelu2_1(self.conv2_1(x))
        # x = self.prelu2_2(self.conv2_2(x))
        # x = F.max_pool2d(x, 2)
        #
        # x = self.prelu3_1(self.conv3_1(x))
        # x = self.prelu3_2(self.conv3_2(x))
        # x = F.max_pool2d(x, 2)

        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return x, y


__factory = {
    'cnn': ConvNet,
}


def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)


if __name__ == '__main__':
    pass
