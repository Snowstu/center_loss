import torchvision.models as models
from losses import *



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Incpv3Net(nn.Module):
    def __init__(self, num_classes):
        super(Incpv3Net, self).__init__()
        net = models.inception_v3(pretrained=False, aux_logits=False)
        net.load_state_dict(torch.load('./models/fonts_inception_v3.pth', map_location=torch.device('cpu')),
                            strict=False)
        num_fc = net.fc.in_features
        net.fc = nn.Linear(num_fc, num_classes)
        net.fc = Identity()
        self.net = net
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.net(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        y = self.fc1(x)

        return x, y


class CusAngleLoss(nn.Module):
    def __init__(self, num_classes):
        super(CusAngleLoss, self).__init__()
        net = models.inception_v3(pretrained=False, aux_logits=False)
        net.load_state_dict(torch.load('../models/fonts_inception_v3.pth', map_location=torch.device('cpu')),
                            strict=False)
        num_fc = net.fc.in_features
        net.fc = nn.Linear(num_fc, num_classes)
        net.fc = Identity()
        self.net = net
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = CusAngleLinear(num_fc, num_classes)

    def forward(self, x):
        x = self.net(x)
        # x = self.avgpool(x)
        y = self.fc1(x)

        return x, y


class BCNN(nn.Module):
    """
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> mean field bilinear pooling
    -> fc.

    The network accepts a 3*448*448 input, and the relu5-3 activation has shape
    512*28*28 since we down-sample 4 times.
    """
    def __init__(self, num_classes, is_all):
        nn.Module.__init__(self)
        self._is_all = is_all
        if self._is_all:
            self.features = models.vgg16(pretrained=True).features
            self.features = nn.Sequential(*list(self.features.children())[:-2])#remove pool5
        # mean filed pooling layer
        self.relu5_3 = nn.ReLU(inplace=False)
        # classification layer
        self.fc = nn.Linear(512 * 512, num_classes)
        if not self._is_all:
            self.apply(BCNN._initParameter)

    def _initParameter(module):
        if isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, val=1.0)
            nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.constant_(module.bias, val=0.0)

    def forward(self, x):
        """Forward pass of the network.
             Args:
                 X, torch.Tensor (N*3*448*448).
             Returns:
                 score, torch.Tensor (N*200).
        """
        N = x.size()[0]
        if self._is_all:
            assert x.size() == (N, 3, 448, 448)
            x = self.features(x)
        assert x.size() == (N, 512, 28, 28)

        # the main branch
        x = self.relu5_3(x)
        assert x.size() == (N, 512, 28, 28)

        # classical bilinear pooling
        x = torch.reshape(x, (N, 512, 28 ** 2))
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (28 ** 2)
        assert x.size() == (N, 512, 512)
        x = torch.reshape(x, (N, 512 * 512))

        # Normalization
        x = torch.sqrt(x * 1e-5)
        x = nn.functional.normalize(x)

        # classification
        x = self.fc(x)
        return x


__factory = {
    "inception_v3": Incpv3Net,
    "cus_angle_loss": CusAngleLoss,
    "bilinear": BCNN
}


def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)


if __name__ == '__main__':
    pass
