from dataloader import *
from model import *
from train_eval_test import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from torch.utils.data import Subset, DataLoader
import argparse



PATH = "resultats/correlations"
BATCH_SIZE = 32
NB_BATCH_SAVE = 30
NB_TEST_PER_BATCH = 3


def save(Att_first, Att_last, corr_lookback, corr_lookforward, nb_test, i):
    fig = plt.figure(figsize=(5, 5))

    ax1 = plt.subplot2grid((2, 2), (0, 0))
    im1 = ax1.imshow(corr_lookback, cmap='viridis', interpolation='nearest')
    ax1.set_title('Lookback correlations')

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    im2 = ax2.imshow(corr_lookforward, cmap='viridis', interpolation='nearest')
    ax2.set_title('Future correlations')
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    
    im3 = ax3.imshow(Att_first, cmap='viridis', interpolation='nearest')
    ax3.set_title('Score map of layer 1')

    ax4 = plt.subplot2grid((2, 2), (1, 1))
    im4 = ax4.imshow(Att_last, cmap='viridis', interpolation='nearest')
    ax4.set_title('Score map of layer L')
    
    cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    fig.colorbar(im1, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig(f"{PATH}/coor_att_{nb_test}_{i}.png")

    print(f"save {PATH}/coor_att_{nb_test}_{i}.png")


def calcule_att_corr(itransformer, x, y, nb_test): 
    A_first = itransformer.liste_attention[0] #attention maps first layer
    A_last = itransformer.liste_attention[-1] #attention maps last layer

    liste_idx_examples = torch.randint(32, (5,))


    for i in liste_idx_examples : 
        #calcule des matrices de correlation
        c_lookback = x[i,:,:].T.cpu()
        corr_lookback = np.corrcoef(c_lookback)
        c_lookforward = y[i,:,:].T.cpu()
        corr_lookforward = np.corrcoef(c_lookforward)

        Att_first = A_first[i, 0, :, :].detach().to('cpu')
        Att_last = A_last[i, 0, :, :].detach().to('cpu')

        save(Att_first, Att_last, corr_lookback, corr_lookforward, nb_test, i)

def predict(dataset, n_train, n_eval, n_test, N, lr, D, hidden_dim, nb_blocks):
    torch.manual_seed(7)

    data = pd.read_csv(f'data/{dataset}.csv', header=None).to_numpy()
    
    train_loader, eval_loader, test_loader = get_loaders(data, BATCH_SIZE, n_train, n_eval, n_test, T=96, S=96)
    
    itransformer = iTransformer(N, 96, D, 96, hidden_dim, nb_blocks).to(device)
    optimizer = torch.optim.Adam(itransformer.parameters(), lr=lr, weight_decay=1e-5)  

    mse, mae = train(itransformer, optimizer, train_loader, eval_loader, 10, device)

    nb_test = 0
    for x,y in test_loader : 

        itransformer.liste_attention = []
        x = x.float().to(device)
        itransformer(x, True)

        calcule_att_corr(itransformer, x, y, nb_test)

        if nb_test == NB_BATCH_SAVE: 
            break
        
        nb_test+=1
    
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

    i = liste_datasets.index(args.dataset)

    if not os.path.exists(f"{PATH}"):
        os.makedirs(f"{PATH}")

    dataset = liste_datasets[i]
    n_train = liste_n_train[i]
    n_eval = liste_n_eval[i]
    n_test = liste_n_test[i]
    N = liste_N[i]
    lr = liste_lr[i]
    D = liste_D[i]
    hidden_dim = liste_hidden_dim[i]
    nb_blocks = liste_nb_blocks[i]

    predict(dataset, n_train, n_eval, n_test, N, lr, D, hidden_dim, nb_blocks)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ablation study script")
    parser.add_argument("--dataset", type=str, required=True, choices=["weather", "electricity", "traffic", "solar", "ETTh1"], help="Name of the dataset for the experiment")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)


