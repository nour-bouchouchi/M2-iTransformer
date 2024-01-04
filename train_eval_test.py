import torch
import torch.nn as nn
import numpy as np

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def eval(model, val_loader, device, criterion, writer, epoch):
    model.eval()

    epoch_loss = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.float().to(device), target.float().to(device)
            yhat = model(data)
            loss = criterion(yhat, target)
            
            epoch_loss.append(loss.item())

        """
        writer.add_scalar('Loss validation', np.mean(epoch_loss), epoch)
    
        for name, weight in model.named_parameters():
            writer.add_histogram(name, weight, epoch)
            writer.add_histogram(f'{name}.grad', weight.grad, epoch)

        entropie = criterion(logits,target)
        writer.add_histogram('Entropy validation', entropie, epoch)
        
        writer.add_scalar('Accuracy validation', np.mean(epoch_accuracy), epoch)
        """

        return np.mean(epoch_loss)
        
def train(model, optimizer, train_loader, val_loader, nb_epoch, device, writer=None):
    
    criterion = nn.MSELoss()
    
    loss_values = []
    loss_eval = []


    for epoch in range(nb_epoch):
        print("---- epoch : ", epoch)

        model.train()
        epoch_loss = []

        for i, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.float().to(device)
            optimizer.zero_grad()

            yhat = model(data)
            loss = criterion(yhat, target)
            epoch_loss.append(loss.item())

            loss.backward()
            optimizer.step()


        loss_values.append(np.mean(epoch_loss))

        """
        writer.add_scalar('Loss train', np.mean(epoch_loss), epoch)
    
        for name, weight in model.named_parameters():
            writer.add_histogram(f'{name}', weight, epoch)
            writer.add_histogram(f'{name}.grad', weight.grad, epoch)

        entropie = criterion(logits,target)
        writer.add_histogram('Entropy train', entropie, epoch)
        
        writer.add_scalar('Accuracy train', np.mean(epoch_accuracy), epoch)

        """

        epoch_loss_eval = eval(model, val_loader, device, criterion, writer, epoch)
        loss_eval.append(epoch_loss_eval)
        print("loss train :", loss_values[-1])
        print("loss eval :", loss_eval[-1])

        
    return loss_values, loss_eval


def test(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    loss_batch = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.float().to(device)
            yhat = model(data)
            loss = criterion(yhat, target)
            loss_batch.append(loss.item())
        
        return np.mean(loss_batch), target, yhat