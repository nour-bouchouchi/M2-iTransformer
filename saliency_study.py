from dataloader import *
from model import *
from train_eval_test import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

LOOK_FORWARD = 96
BATCH_SIZE = 32
PATH = "resultats/saliency"

def perturbation_data(x, num_window, mod, mode="zero", size_window=1):
    x_noisy = x.clone()
    if mode =="zero":
        x_noisy[num_window * size_window: (num_window+1) * size_window, mod] = 0
    elif mode == "mean_mod":
        x_noisy[num_window * size_window: (num_window+1) * size_window, mod] = x_noisy[:, mod].mean(dim=0)
    elif mode =="mean_time":
        x_noisy[num_window * size_window: (num_window+1) * size_window, mod] =x_noisy[num_window * size_window: (num_window+1) * size_window, :].mean(dim=1)
    return x_noisy

def score(target, yhat, yhat_noisy):
    criterion = torch.nn.MSELoss()
    loss = criterion(target, yhat)
    loss_noisy = criterion(target, yhat_noisy)

    diff_loss = loss - loss_noisy
    return diff_loss


def saliency_map(dataset, itransformer, x, target, N, mode="zero", size_window=1, idx_ex=0):
    saliency_map = np.zeros((N, LOOK_FORWARD//size_window))
    for mod in range(N):
        for num_window in range(LOOK_FORWARD//size_window): 
            x_noisy = perturbation_data(x, num_window, mod, mode="zero", size_window=1)
            yhat_noisy = itransformer(x_noisy.reshape((1, 96, N)))[0]
            yhat = itransformer(x.reshape((1, 96, N)))[0]
            saliency_map[mod, num_window] = score(target, yhat, yhat_noisy)

    plt.imshow(saliency_map, cmap='viridis', aspect='auto')
    plt.title('Saliency Map')
    plt.xlabel('Time')
    plt.ylabel('Modality')
    plt.colorbar()
    plt.tight_layout()

    plt.savefig(f"{PATH}/{dataset}/saliency_{mode}_{idx_ex}.png")

    print(f"saved {PATH}/{dataset}/saliency_{mode}_{idx_ex}.png")


def predict(dataset, train_loader, eval_loader, test_loader, N, lr, D, hidden_dim, nb_blocks, mode="zero", idx_ex=0, size_window=1):
    
    itransformer = iTransformer(N, 96, D, 96, hidden_dim, nb_blocks).to(device)
    optimizer = torch.optim.Adam(itransformer.parameters(), lr=lr, weight_decay=1e-5)  

    mse, mae = train(itransformer, optimizer, train_loader, eval_loader, 10, device)

    for x, target in test_loader:
        x = x[idx_ex].float().to(device)
        target = target[idx_ex].float().to(device)
        break


    saliency_map(dataset, itransformer, x, target, N, mode=mode, size_window=size_window, idx_ex=idx_ex)
    return 



def main(args):


    liste_datasets = ["weather", "electricity", "traffic", "solar", "ETTh1"]

    liste_n_train = [36792, 18317, 12185, 36601, 8545]
    liste_n_eval = [5271, 2633, 1757, 5161, 2881]
    liste_n_test = [10540, 5261, 3509, 10417, 2881]

    liste_N = [21, 321, 862, 137, 7]

    liste_D = [512, 512, 512, 512, 512]
    liste_lr = [5*1e-4 , 1e-3, 1e-3, 1e-4, 1e-4] 
    liste_hidden_dim = [64, 512 ,512, 512, 64] 
    liste_nb_blocks = [2, 2, 2, 2, 1] 

    if args.dataset not in liste_datasets:
        raise ValueError("Invalid dataset name. Choose from: weather, electricity, traffic, solar, ETTh1")

    if args.mode not in ["zero", "mean_mod", "mean_time"]:
        raise ValueError("Invalid mode name. Choose from: zero, mean_mod, mean_time")
    
    if args.size_window < 1:
        raise ValueError("Invalid size_window. Must be > 0")

    if args.idx_ex < 0 or args.idx_ex>=BATCH_SIZE:
        raise ValueError("Invalid idx_ex. Must have 0 <= idx_ex < 32")


    idx_ex = args.idx_ex
    mode = args.mode
    size_window= args.size_window

    i = liste_datasets.index(args.dataset)

    dataset = liste_datasets[i]
    n_train = liste_n_train[i]
    n_eval = liste_n_eval[i]
    n_test = liste_n_test[i]
    N = liste_N[i]
    lr = liste_lr[i]
    D = liste_D[i]
    hidden_dim = liste_hidden_dim[i]
    nb_blocks = liste_nb_blocks[i]


    if not os.path.exists(f"{PATH}/{dataset}"):
        os.makedirs(f"{PATH}/{dataset}")
        

    torch.manual_seed(7)
    data = pd.read_csv(f'data/{dataset}.csv', header=None).to_numpy()  
    train_loader, eval_loader, test_loader = get_loaders(data, BATCH_SIZE, n_train, n_eval, n_test, T=96, S=96)

    predict(dataset, train_loader, eval_loader, test_loader, N, lr, D, hidden_dim, nb_blocks, mode=mode, idx_ex=idx_ex, size_window=size_window)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ablation study script")
    parser.add_argument("--dataset", type=str, required=True, choices=["weather", "electricity", "traffic", "solar", "ETTh1"], help="Name of the dataset for the experiment")
    parser.add_argument("--mode", type=str, default="zero", choices=["zero", "mean_mod", "mean_time"], help="Kind of perturbation in the data")
    parser.add_argument("--size_window", type=int, default=1, help="Size of the perturbation window in the data")
    parser.add_argument("--idx_ex", type=int, default=0, help="Index of the example")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)


