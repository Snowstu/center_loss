import torch
from torch.autograd.function import Function
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import math


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


class CusAngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=4, phiflag=True):
        super(CusAngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.m = m
        self.phiflag = phiflag
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, x):
        eps = 1e-12

        with torch.no_grad():
            self.fc.weight.div_(torch.norm(self.fc.weight, dim=1, keepdim=True))

        x_norm = F.normalize(x, dim=1)
        x_len = x.norm(2, 1, True).clamp_min(eps)
        # cos_theta = self.fc(x_norm)

        # cos_theta = torch.matmul(x_norm, F.normalize(self.weight))

        cos_theta = F.linear(x_norm, F.normalize(self.weight))

        cos_m_theta = self.mlambda[self.m](cos_theta)

        theta = Variable(cos_theta.data.acos())
        k = (self.m * theta / math.pi).floor()
        n_one = k * 0.0 - 1
        phi_theta = (n_one ** k) * cos_m_theta - 2 * k
        cos_theta = cos_theta * x_len
        phi_theta = phi_theta * x_len
        return cos_theta, phi_theta


class CusAngleLoss(nn.Module):
    def __init__(self):
        super(CusAngleLoss, self).__init__()
        self.iter = 0

        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.gamma = 0

    def forward(self, input, labels):
        self.iter += 1

        target = labels.view(-1, 1)  # size=(B,1)
        cos_theta, phi_theta = input
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        index = Variable(index)

        # self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.1 * self.iter))
        output = cos_theta * 1.0  # size=(B,Classnum)
        # output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        # output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        output[index] -= cos_theta[index] * (1.0 + 0)
        output[index] += phi_theta[index] * (1.0 + 0)

        loss = F.cross_entropy(output, target.squeeze())

        # softmax loss

        # logit = F.log_softmax(output)
        #
        # logit = logit.gather(1, target).view(-1)
        # pt = logit.data.exp()
        #
        # loss = -1 * (1 - pt) ** self.gamma * logit
        # loss = loss.mean()
        return loss
