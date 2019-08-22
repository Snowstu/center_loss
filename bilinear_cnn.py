import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from nets.networks import BCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "data/"
input_size = 448


class BCNNManager(object):
    def __init__(self, options, path):
        print("Prepare the network and data.")
        self._options = options
        self._path = path
        # networks
        if self._path['pretrained'] is not None:
            self._net = BCNN(num_classes=26, is_all=True).to(device)
            self._net.load_state_dict(torch.load(self._path["pretained"]), strict=False)
        else:
            self._net = BCNN(26, is_all=False).to(device)

        print(self._net)
        self._criterion = nn.CrossEntropyLoss().to(device)
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self._options["lr"], momentum=0.9,
                                          weight_decay=self._options["weight_decay"])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode='max', factor=0.1, patience=3, verbose=True,
            threshold=1e-4)

        data_transforms = transforms.Compose([
            transforms.Resize(size=input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])
        data_loader = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                       for x in ['train', 'val']}
        self._train_loader = torch.utils.data.DataLoader(data_loader["train"], batch_size=64, shuffle=True)
        self._val_loader = torch.utils.data.DataLoader(data_loader["val"], batch_size=64, shuffle=True)

    def train(self):
        print("training...")
        best_acc = 0.0
        best_epoch = None
        print("Epoch\tTrain loss\tTrain acc\tTest acc")
        for t in range(self._options["epochs"]):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for x, y in self._train_loader:
                x = x.to(device)
                y = y.to(device)

                # forward pass
                score = self._net(x)
                loss = self._criterion(score, y)
                with torch.no_grad():
                    epoch_loss.append(loss.data[0])
                    # prediction
                    _, prediction = torch.max(score.data, 1)
                    num_total += y.size(0)
                    num_correct += torch.sum(prediction == y.data)
                # backword pass
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._val_loader)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print("*", end="")

                torch.save(self._net.state_dict(), 'models/best_model.pth.tar')
            print("%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%" % (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
            self._scheduler.step(test_acc)
        print("best at epoch %d, test accuaray %f" % (best_epoch, best_acc))

    def _accuracy(self, data_loader):

        with torch.no_grad():
            self._net.eval()
            num_correct = 0
            num_total = 0
            for X, y in data_loader:
                # Data.
                X = X.to(device)
                y = y.to(device)

                # Prediction.
                score = self._net(X)
                _, prediction = torch.argmax(score, dim=1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y).item()
            self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total


def main():
    import argparse
    parser = argparse.ArgumentParser(description="train bilinear CNN")
    parser.add_argument("base_lr", dest="base_lr", type=float, required=True)
    parser.add_argument("--batch_size", dest="batch_size", type=int, required=True)
    parser.add_argument("--epochs", dest="epochs", type=int, required=True)
    parser.add_argument("--weight_decay", dest="weight_decay", type=int, required=True)
    parser.add_argument('--pretrained', dest='pretrained', type=str,
                        required=False, help='Pre-trained model.')
    args = parser.parse_args()

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }

    project_root = os.popen('pwd').read().strip()
    path = {"data": os.path.join(project_root, "data/"),
            "model": os.path.join(project_root, "model", args.model)}
    for d in path:
        if d == "pretrained":
            assert path[d] is None or os.path.isfile(path[d])
        else:
            assert os.path.isdir(path[d])

    manager = BCNNManager(options, path)
    manager.train()


if __name__ == "__main__":
    main()
