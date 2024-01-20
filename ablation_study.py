from dataloader import *
from model import *
from train_eval_test import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os.path
from torch.utils.data import Subset, DataLoader, Dataset
from datetime import datetime 


BATCH_SIZE = 32
PATH = "resultats/ablation"



def save_scores(path, loss_mse, loss_mae, trm, i):
    file_exists = os.path.isfile(path)

    with open(path, 'a' if file_exists else 'w', newline='') as fichier_sortie:
        writer = csv.writer(fichier_sortie)

        if not file_exists:
            writer = csv.writer(fichier_sortie)
            writer.writerow(['Model', 'Run 1 (MSE)', 'Run 1 (MAE)', 'Run 2 (MSE)', 'Run 2 (MAE)', 'Run 3 (MSE)', 'Run 3 (MAE)', 'MSE Moyenne', 'MSE Variance', 'MAE Moyenne', 'MAE Variance'])
            writer.writerow([trm, loss_mse, loss_mae] )
        
        else :   
            if i==0 : 
                writer.writerow([trm, loss_mse, loss_mae] )
            else : 
                res = pd.read_csv(path)
                res.loc[res["Model"] == trm, f"Run {i+1} (MSE)"] = loss_mse
                res.loc[res["Model"] == trm, f"Run {i+1} (MAE)"] = loss_mae
                res.to_csv(path, index=False)

        if i==2 : 
            mse = list(res.iloc[-1, 1::2][:-2])
            mae = list(res.iloc[-1, 2::2][:-2])
            mse_moyenne = np.mean(mse)
            mse_variance = np.var(mae)
            
            mae_moyenne = np.mean(mse)
            mae_variance = np.var(mae)
            
            res.loc[res["Model"] == trm, "MSE Moyenne"] = mse_moyenne
            res.loc[res["Model"] == trm, "MAE Moyenne"] = mae_moyenne
            res.loc[res["Model"] == trm, "MSE Variance"] = mse_variance
            res.loc[res["Model"] == trm, "MAE Variance"] = mae_variance
            res.to_csv(path, index=False)

    print("saved at ", datetime.now().time().strftime("%H:%M"))


def predict_lookforward(dataset, train_loader, eval_loader, test_loader, N, trm, lr, D, hidden_dim, nb_blocks):
    """
    Permet de faire les prédictions pour un lookforward length donné (5 prédictions sur 5 seeds différentes)
    """    
    for i in range(3):
        print(f"  --- i : {i} --- ")
        torch.manual_seed(i)
        np.random.seed(i)

        #####tester si a déjà fait ce i
        file_exists = os.path.isfile(f'{PATH}/{dataset}_ablation.csv')
        if not file_exists  : 
            i_done = False

        else : 
            res = pd.read_csv(f"{PATH}/{dataset}_ablation.csv")
            if res.empty : 
                i_done = False
            else : 
                last_lign = res.iloc[-1]
                last_valid = last_lign.last_valid_index()
                if last_valid=="MAE Variance":
                    i_done = False
                else : 
                    last_i = int(last_valid.split(" ")[1])
                    i_done = last_i > i

        if not i_done :   
            itransformer = iTransformer(N, 96, D, 96, hidden_dim, nb_blocks, typeTrmBlock=trm).to(device)
            optimizer = torch.optim.Adam(itransformer.parameters(), lr=lr, weight_decay=1e-5)  

            _, _ = train(itransformer, optimizer, train_loader, eval_loader, 10, device)

            loss_mse, loss_mae, _, _ = test(itransformer, test_loader, device)
            
            save_scores(f'{PATH}/{dataset}_ablation.csv', loss_mse, loss_mae, trm, i)

    return 


def predict(dataset, n_train, n_eval, n_test, N, lr, D, hidden_dim, nb_blocks): 
    """
    Permet de faire les prédictions pour un dataset donné 
    dataset : path du dataset 
    """
    data = pd.read_csv(f'data/{dataset}.csv', header=None).to_numpy()

    liste_model=["inverted", "Att_Att", "FFN_Att", "FFN_FFN", "Att_variate", "FFN_temporal"]
    for i,trm in enumerate(liste_model) : 
        print(f" ----- Trm {trm} ----- ")
        file_exists = os.path.isfile(f'{PATH}/{dataset}_ablation.csv')
        if file_exists : 
            res = pd.read_csv(f'{PATH}/{dataset}_ablation.csv')
            ligne_done = pd.notna(res.iloc[-1]["MAE Variance"])
            last_done = res.iloc[-1]["Model"] 
            trm_done = (i < liste_model.index(last_done) ) or (last_done==trm and ligne_done)
        else : 
            trm_done = False

        if not trm_done : 
            train_loader, eval_loader, test_loader = get_loaders(data, BATCH_SIZE, n_train, n_eval, n_test, T=96, S=96)

            predict_lookforward(dataset, train_loader, eval_loader, test_loader, N, trm, lr, D, hidden_dim, nb_blocks)




def main():
    liste_datasets = ["weather", "electricity", "traffic", "solar", "ETTh1"]

    liste_n_train = [36792, 18317, 12185, 36601, 8545]
    liste_n_eval = [5271, 2633, 1757, 5161, 2881]
    liste_n_test = [10540, 5261, 3509, 10417, 2881]

    liste_N = [21, 321, 862, 137, 7]

    liste_D = [512, 512, 512, 512, 512]
    liste_lr = [5*1e-4 , 1e-3, 1e-3, 1e-4, 1e-4] 
    liste_hidden_dim = [64, 512 ,512, 512, 64] 
    liste_nb_blocks = [2, 2, 2, 2, 1] 

    if not os.path.exists(f"{PATH}"):
        os.makedirs(f"{PATH}")

    for i in range(len(liste_datasets)):
        print(f"-------------------- {liste_datasets[i]} --------------------")

        dataset = liste_datasets[i]

        file_exists = os.path.isfile(f'{PATH}/{dataset}_ablation.csv')

        if file_exists :
            res = pd.read_csv(f'{PATH}/{dataset}_ablation.csv')
            done = (res.iloc[-1]["Model"]=="FFN_temporal") and pd.notna(res.iloc[-1]["MAE Variance"])


        if not file_exists or not done : 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
