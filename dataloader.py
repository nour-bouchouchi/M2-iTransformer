import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class MonDataLoaderAllData(Dataset):
    """
    Classe permettant d'obtenir un dataloader pour des données
    """
    def __init__(self, data, lookback_size, lookforward_size, scaler=None):     
        self.lookback_size = lookback_size
        self.lookforward_size = lookforward_size
        self.data = data
        self.scaler = scaler
        if self.scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(data)
        else:
            self.scaler = None

        d = []
        for i in range(0, 1 + len(data) - (self.lookback_size + self.lookforward_size), 1):
            seq_x = self.data[i:i+self.lookback_size, :]
            seq_y = self.data[i+self.lookback_size:i+self.lookback_size+self.lookforward_size, :]
            d.append((seq_x, seq_y))
        self.data = d

    def __len__(self):
        return len(self.data) - self.lookback_size -1 #- self.lookback_size - self.lookforward_size + 1

    def __getitem__(self, idx):
      seq_x, seq_y = self.data[idx]
      return torch.tensor(seq_x), torch.tensor(seq_y)
      


def get_loaders(data, batch_size, n_train, n_eval, n_test, T=96, S=96):
    """
    Fonction retournant le data loader de train, eval et test (dont on spécifie en entrée la taille). 
    """
    train_data = data[ : n_train+T+S]
    val_data = data[n_train : n_train+n_eval+T+S]
    test_data = data[n_train+n_eval : n_train+n_eval+n_test+T+S]

    scaler = StandardScaler()
    scaler.fit(train_data)

    train_dataset = MonDataLoaderAllData(train_data, T, S, scaler=scaler)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    eval_dataset = MonDataLoaderAllData(val_data, T, S, scaler=scaler)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MonDataLoaderAllData(test_data, T, S, scaler = scaler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader