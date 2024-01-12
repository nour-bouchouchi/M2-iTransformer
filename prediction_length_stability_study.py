from dataloader import *
from model import *
from train_eval_test import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import csv



BATCH_SIZE = 32

def save_scores(path, liste_loss_mse, liste_loss_mae):
    with open(path, 'w', newline='') as fichier_sortie:
        writer = csv.writer(fichier_sortie)
        
        writer.writerow(['Longueur de prédiction', 'Run 1 (MSE)', 'Run 1 (MAE)', 'Run 2 (MSE)', 'Run 2 (MAE)', 'Run 3 (MSE)', 'Run 3 (MAE)', 'Run 4 (MSE)', 'Run 4 (MAE)', 'Run 5 (MSE)', 'Run 5 (MAE)', 'MSE Moyenne', 'MSE Variance', 'MAE Moyenne', 'MAE Variance'])
        
        for i, longueur in enumerate([96, 192, 336, 720]):
            mse_resultats = liste_loss_mse[i]
            mae_resultats = liste_loss_mae[i]
            
            mse_moyenne = np.mean(mse_resultats)
            mse_variance = np.var(mse_resultats)
            
            mae_moyenne = np.mean(mae_resultats)
            mae_variance = np.var(mae_resultats)
            
            alternate_columns = [col for pair in zip(mse_resultats, mae_resultats) for col in pair]
            writer.writerow([longueur] + alternate_columns + [mse_moyenne, mse_variance, mae_moyenne, mae_variance])

    
def predict_lookforward(train_loader, eval_loader, test_loader, N, s, lr, D, hidden_dim, nb_blocks):
    """
    Permet de faire les prédictions pour un lookforward length donné (5 prédictions sur 5 seeds différentes)
    """    
    liste_loss_mse = []
    liste_loss_mae = []
    for i in range(5):
        print(f"  --- i : {i} --- ")
        torch.manual_seed(i)
        np.random.seed(i)

        # initialise le modèle et l'optimizer

        itransformer = iTransformer(N, 96, D, s, hidden_dim, nb_blocks).to(device)
        optimizer = torch.optim.Adam(itransformer.parameters(), lr=lr, weight_decay=1e-5)  

        # train
        _, _ = train(itransformer, optimizer, train_loader, eval_loader, 10, device)

        loss_mse, loss_mae, _, _ = test(itransformer, test_loader, device)

        liste_loss_mse.append(loss_mse)
        liste_loss_mae.append(loss_mae)

    # save le res dans le csv
    return liste_loss_mse, liste_loss_mae
    


def predict(dataset, n_train, n_eval, n_test, N, lr, D, hidden_dim, nb_blocks): 
    """
    Permet de faire les prédictions pour un dataset donné 
    dataset : path du dataset 
    """
    data = pd.read_csv(f'data/{dataset}.csv', header=None).to_numpy()
    
    liste_mse = []
    liste_mae = []
    
    for s in [96, 192, 336, 720] : 
        print(f" ----- S {s} ----- ")
        train_loader, eval_loader, test_loader = get_loaders(data, BATCH_SIZE, n_train, n_eval, n_test, T=96, S=s)

        liste_loss_mse, liste_loss_mae = predict_lookforward(train_loader, eval_loader, test_loader, N, s, lr, D, hidden_dim, nb_blocks)
        
        liste_mse.append(liste_loss_mse)
        liste_mae.append(liste_loss_mae)
        
    save_scores(f'resultats/{dataset}.csv', liste_mse, liste_mae)



def main(): 
    liste_datasets = ["ETTh1", "weather", "electricity", "traffic", "solar"]

    n_train = [8545, 36792, 18317, 12185, 36601]
    n_eval = [2881, 5271, 2633, 1757, 5161]
    n_test = [2881, 10540, 5261, 3509, 10417]

    liste_N = [7, 21, 321, 862, 137]

    liste_D = [512, 512, 512, 512, 512]
    liste_lr = [1e-4, 5*1e-4 , 1e-3, 1e-3, 1e-3] #solar à trouver
    liste_hidden_dim = [64, 64, 512 ,512, 512] #solar à trouver
    liste_nb_blocks = [1, 2, 2,2,2] #solar à trouver

    for i in range(len(liste_datasets)): 
        print(f"-------------------- {liste_datasets[i]} --------------------")

        dataset = liste_datasets[i]
        n_train = n_train[i]
        n_eval = n_eval[i]
        n_test = n_test[i]
        N = liste_N[i]
        lr = liste_lr[i]
        D = liste_D[i]
        hidden_dim = liste_hidden_dim[i]
        nb_blocks = liste_nb_blocks[i]
        
        predict(dataset, n_train, n_eval, n_test, N, lr, D, hidden_dim, nb_blocks)
        
        



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()