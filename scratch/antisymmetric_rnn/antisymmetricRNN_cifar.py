import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from sklearn.metrics import accuracy_score
import time

from src.antisymmetricRNN import AntisymmetricRNNModel


permuted = False
num_epochs = 2
num_channels = 3
num_classes = 10
num_units = 256
batch_size = 128
path_base = '/home/bodrue/Data/neural_control/antisymmetricRNN'
path_model = os.path.join(path_base, 'models', 'cifar')
path_dataset = os.path.join(path_base, 'datasets', 'cifar')
os.makedirs(path_model, exist_ok=True)
os.makedirs(path_dataset, exist_ok=True)

torch.manual_seed(42)

train = torchvision.datasets.CIFAR10(path_dataset, train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.ToTensor()]))

test = torchvision.datasets.CIFAR10(path_dataset, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()]))

x_train = torch.Tensor(train.data)
y_train = torch.Tensor(train.targets)

x_test = torch.Tensor(test.data)
y_test = torch.Tensor(test.targets)

x_train = torch.reshape(x_train, (len(x_train), -1, num_channels))
x_test = torch.reshape(x_test, (len(x_test), -1, num_channels))

x_train = x_train.float() / 255
x_test = x_test.float() / 255

if permuted:
    permuted_idx = torch.randperm(x_train.shape[1])
    x_train = x_train[:, permuted_idx]
    x_test = x_test[:, permuted_idx]

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size,
                          shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size,
                         shuffle=True)

num_train = len(x_train)
num_test = len(x_test)

model = AntisymmetricRNNModel(num_channels, num_classes, num_units, gamma=0.01,
                              eps=0.01, batch_size=batch_size).cuda()
opt = torch.optim.SGD(model.parameters(), momentum=0.5, lr=0.1)
loss = nn.CrossEntropyLoss()

acc_test = None
for e in range(num_epochs):
    time_start = time.time()
    ce_train, ce_test, acc_train, acc_test = 0, 0, 0, 0
    for batch_x, batch_y in train_loader:
        _batch_size = len(batch_x)
        h = torch.zeros(_batch_size, num_units).cuda()
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        opt.zero_grad()
        output = model(batch_x, h)
        loss_value = loss(output, batch_y.long())
        loss_value.backward()
        opt.step()
        ce_train += loss_value.item() * _batch_size
        _, preds = torch.max(output, 1)
        acc_train += torch.sum(torch.eq(preds, batch_y)).item()
    ce_train /= num_train
    acc_train *= 100 / num_train
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            _batch_size = len(batch_x)
            h = torch.zeros(_batch_size, num_units).cuda()
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = model(batch_x, h)
            loss_value = loss(output, batch_y.long())
            ce_test += loss_value.item() * _batch_size
            _, preds = torch.max(output, 1)
            acc_test += torch.sum(torch.eq(preds, batch_y)).item()
    ce_test /= num_test
    acc_test *= 100 / num_test
    time_end = time.time()
    print(f"Epoch {e}, train loss {ce_train}, train acc {acc_train}, "
          f"test loss {ce_test}, test acc {acc_test}, "
          f"time {time_end - time_start}")

torch.save(model.state_dict(), os.path.join(path_model, f'{acc_test}.pt'))

predicts = []
true = []
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        _batch_size = len(batch_x)
        h = torch.zeros(_batch_size, num_units).cuda()
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        output = model(batch_x, h)
        _, preds = torch.max(output, 1)
        predicts.append(preds.detach().cpu().numpy())
        true.append(batch_y.detach().cpu().numpy())
predicts = np.concatenate(predicts)
true = np.concatenate(true)

print("Final test accuracy: {:.2%}".format(accuracy_score(true, predicts)))
