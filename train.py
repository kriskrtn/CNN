import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(model, loader):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in loader:

        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():

          predictions = model(data)
          criterion = nn.CrossEntropyLoss()
          loss = criterion(predictions, target)

        loss_log.append(loss.item())

        probs, classes = torch.max(predictions, dim=1)
        acc = (torch.eq(classes, target).sum()) / target.size(0)

        acc_log.append(acc.item())

    return np.mean(loss_log), np.mean(acc_log)

def train_epoch(model, optimizer, train_loader):
    loss_log = []
    acc_log = []
    model.train()

    for data, target in train_loader:

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        predictions = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(predictions, target)

        loss.backward()
        optimizer.step()

        loss_log.append(loss.item())

        probs, classes = torch.max(predictions, dim=1)
        acc = (torch.eq(classes, target).sum()) / target.size(0)

        acc_log.append(acc.item())

    return loss_log, acc_log

def train(model, optimizer, n_epochs, train_loader, val_loader, scheduler=None):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader)
        val_loss, val_acc = test(model, val_loader)

        train_loss_log.extend(train_loss)
        train_acc_log.extend(train_acc)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        print(f"Epoch {epoch}")
        print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")

        if scheduler is not None:
            scheduler.step()

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log