from dataloader import *
from model import *
from train_eval_test import *
import pandas as pd
import csv
import os.path

PATH = "resultats/statistics"
BATCH_SIZE = 32


def save_statistics(path, dataset,  mean_loss, var_loss , median_loss, top5_best_indices, top5_worst_indices, best_values, worst_values):
    """
    Fonction permettant d'enregistrer les statistiques (meilleures score, score moyen, meilleures et pire modalités)
    """
    file_exists = os.path.isfile(path)
    with open(path, 'a' if file_exists else 'w', newline='') as fichier_sortie:
            writer = csv.writer(fichier_sortie)

            if not file_exists:
                writer = csv.writer(fichier_sortie)
                writer.writerow(['Dataset', "Best MSE", "Worst MSE", "Top 5 modalities", "Worst 5 modalities", "Mean", "Median", "Variance"])
            
            writer.writerow([dataset, best_values, worst_values, top5_best_indices, top5_worst_indices, mean_loss, median_loss, var_loss] )


def predict(dataset, n_train, n_eval, n_test, N, lr, D, hidden_dim, nb_blocks):
    """
    Fonction permettant d'entraîner un modèle et de calculer des statistiques sur les score et modalités. 
    """
    torch.manual_seed(7)

    data = pd.read_csv(f'data/{dataset}.csv', header=None).to_numpy()
    
    train_loader, eval_loader, test_loader = get_loaders(data, BATCH_SIZE, n_train, n_eval, n_test, T=96, S=96)
    
    itransformer = iTransformer(N, 96, D, 96, hidden_dim, nb_blocks).to(device)
    optimizer = torch.optim.Adam(itransformer.parameters(), lr=lr, weight_decay=1e-5)  

    mse, mae = train(itransformer, optimizer, train_loader, eval_loader, 10, device)

    criterion = torch.nn.MSELoss()
    loss_mse_per_mod = np.zeros((N, test_loader.__len__()))
    for i, (x, target) in enumerate(test_loader):
        x = x.float().to(device)
        yhat = itransformer(x)
        for mod in range(N):
            target_mod = target[:,:,mod].float().to(device)
            yhat_mod = yhat[:,:,mod].float().to(device)
            loss = criterion(target_mod, yhat_mod)
            loss_mse_per_mod[mod, i] = loss
    mean_loss_mod = loss_mse_per_mod.mean(axis=1)
    mean_loss = loss_mse_per_mod.mean()
    var_loss = loss_mse_per_mod.var()
    median_loss = np.median(loss_mse_per_mod)
    top5_best_indices = np.argsort(mean_loss_mod)[:5]   
    top5_worst_indices = np.argsort(mean_loss_mod)[-5:]
    best_values = mean_loss_mod[top5_best_indices[0]]
    worst_values = mean_loss_mod[top5_worst_indices[-1]]

    save_statistics(f"{PATH}/stat.csv", dataset,  mean_loss, var_loss , median_loss, top5_best_indices, top5_worst_indices, best_values, worst_values)
    

    

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

        file_exists = os.path.isfile(f'{PATH}/stat.csv')
    
        if file_exists :
            res = pd.read_csv(f'{PATH}/stat.csv')
            done = (liste_datasets.index(res.iloc[-1]["Dataset"])>=i)

        if not file_exists or not done : 

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()