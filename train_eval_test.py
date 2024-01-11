import torch
import torch.nn as nn
import numpy as np

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs(input - target))

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
        
def train(model, optimizer, train_loader, val_loader, nb_epoch, device, writer=None, patience=3):
    criterion = nn.MSELoss()

    loss_values = []
    loss_eval = []
    best_val_loss = float('inf')
    counter = 0

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

        epoch_loss_eval = eval(model, val_loader, device, criterion, writer, epoch)
        loss_eval.append(epoch_loss_eval)
        print("loss train :", loss_values[-1])
        print("loss eval :", loss_eval[-1])

        # Early stopping check
        if loss_eval[-1] < best_val_loss:
            best_val_loss = loss_eval[-1]
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break

    return loss_values, loss_eval

def test(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    maeLoss = MAELoss()
    loss_batch = []
    loss_batch_mae = []
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.float().to(device)
            yhat = model(data)

            loss = criterion(yhat, target)
            loss_mae = maeLoss(yhat, target)

            loss_batch.append(loss.item())
            loss_batch_mae.append(loss_mae.item())
        return np.mean(loss_batch), np.mean(loss_batch_mae), target, yhat