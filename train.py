import datetime
import time
import sys
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import nets.networks as nnets

from losses import CenterLoss
from utils import AverageMeter,save_checkpoint

sys.path.append('center_loss')

eval_freq = 10
stepsize = 20
max_epoch = 10
num_classes = 45
input_size = 299
data_dir = "data"
best_acc = 0.0

device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(300),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Scale(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor()
    ])
}
def main():
    global best_acc
    use_gpu = torch.cuda.is_available()
    data_dir = r'data'
    dsetset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
             for x in ['train', 'val']}
    dsetset_loaders = {x: torch.utils.data.DataLoader(dsetset[x], batch_size=64, shuffle=True)
                    for x in ['train', 'val']}
    dsetset_sizes = {x: len(dsetset[x]) for x in ['train', 'val']}
    class_to_idx = dsetset['train'].class_to_idx
    with open("labels_index.txt","w") as f:
        f.write(str(class_to_idx))


    model = nnets.create(name="cnn", num_classes=num_classes)
    model.to(device)
    # if use_gpu:
    #     model = nn.DataParallel(model).cuda()


    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes, feat_dim=2048).to(device)
    optimizer_model = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)
    scheduler = lr_scheduler.StepLR(optimizer_model, step_size=stepsize, gamma=0.5)
    start_time = time.time()

    for epoch in range(max_epoch):
        print("==> Epoch {}/{}".format(epoch + 1, max_epoch))

        train(model, criterion_xent, criterion_cent,
              optimizer_model, optimizer_centloss,
              dsetset_loaders["train"], use_gpu, num_classes, epoch)

        if stepsize > 0: scheduler.step()

        if eval_freq > 0 and (epoch + 1) % eval_freq == 0 or (epoch + 1) == max_epoch:
            print("==> Test")
            acc, err = test(model, dsetset_loaders["test"], use_gpu, num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            if acc > best_acc:
                is_best = True
                best_acc = max(acc, best_acc)
                save_checkpoint(model.state_dict(), is_best, fpath='models/checkpoint.pth.tar')


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))

    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
          trainloader, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()

    for data in trainloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        features, outputs = model(inputs)
        labels = torch.max(labels, 1)[1]
        loss_cent = criterion_cent(labels,features)
        loss_xent = criterion_xent(outputs, labels)

        loss_cent *= 1
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / 1)
        optimizer_centloss.step()

        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

    print(" Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
          .format(losses.val, losses.avg, xent_losses.val, xent_losses.avg,
                  cent_losses.val, cent_losses.avg))


def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            # if use_gpu:
            #     data, labels = data.cuda(), labels.cuda()
            data = data.to(device)
            labels = labels.to(device)
            labels = torch.max(labels, 1)[1]

            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


if __name__ == '__main__':
    main()
