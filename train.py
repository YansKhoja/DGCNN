# Bibliothèques utilisés pour l'explotation du projet et la visualisation des résultats
import pandas as pd
import argparse
import numpy as np
from plotly.subplots import make_subplots # graphique interactive
import plotly.graph_objects as go # graphique interactive
# Import de la fonction de tranformation des données
from utils import TransFormGetDegree

# Bibliothèque permettant l'implémentation du modèle
import torch
from torch import nn
from model import Dgcnn # Pour le modèle
from torch.optim import Adam # Pour l'optimisateur

from torch_geometric.datasets import TUDataset # pour les datasets
from torch_geometric.data import DataLoader # Découpage des datasets

# Fonction permettant l'entrainement du modèle sur un ensemble de données dédiées à l'entrainement
def train(loader):
    model.train()
    running_loss = 0
    for data in loader:
        classes = model(data)
        loss = loss_criterion(classes, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# Fonction permettant d'évaluer le modèle sur un ensemble de données dédiées aux tests
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    # retourne le pourcentage de graph bien classé
    return correct / len(loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='DGCNN',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''\
    ========================
    Dgcnn : graph classifier 
    ========================       
    Launching Method 
    ------------------------       
    TRAIN : Launch this prog training mode need a valoriszation of different arguments.
    NORMAL : Launch normal mode need having runnning previous the training on the three datasets (REDDIT-BINARY, PROTEINS, IMDB-BINARY)''',
                                     epilog='ex : python train.py IMDB-BINARY 50 20 NORMAL')
    parser.add_argument("DATASET_NAME", help="Dataset name (REDDIT-BINARY, PROTEINS, IMDB-BINARY)")
    parser.add_argument("BATCH_SIZE", help="Batch size", type=int, default=50)
    parser.add_argument("NUM_EPOCHS", help="Numbers epoch", type=int, default=10)
    parser.add_argument("MODE", help="Executing mode TRAIN/NORMAL", type=str, default="TRAIN")
    args = parser.parse_args()

    DATASET_NAME = args.DATASET_NAME
    BATCH_SIZE = args.BATCH_SIZE
    NUM_EPOCHS = args.NUM_EPOCHS
    MODE = args.MODE
    name_file = "result_" + DATASET_NAME + ".txt"

    if MODE == 'TRAIN':
        dataset = TUDataset(root='C:/Users/Utilisateur/Documents/GitHub/DGCNN/data', name=DATASET_NAME, transform = TransFormGetDegree())
        torch.manual_seed(12345)
        dataset = dataset.shuffle()

        train_dataset = dataset[:700]
        print(f'Nombre de graphes pour l\'apprentissage: {len(train_dataset)}')
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = dataset[700:]
        print(f'Nombre de graphes pour le test : {len(test_dataset)}')
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        dataset_features, dataset_classes, dataset_len = dataset.num_features, dataset.num_classes, len(dataset)
        print(f'Dataset: {dataset}:')
        print('======================')
        print(f'Nombre de graphes: {dataset_len}')
        print(f'Nombre d\'attributs: {dataset_features}')
        print(f'Nombre de classes: {dataset_classes}')

        model = Dgcnn(dataset_features, dataset_classes)

        optimizer = Adam(model.parameters())
        loss_criterion = nn.NLLLoss()

        with open("C:/Users/Utilisateur/Documents/GitHub/DGCNN/result/"+name_file,"w", encoding='utf-8') as file :
            print(f"Train start on {NUM_EPOCHS}")
            i = 0
            tab = np.empty((NUM_EPOCHS,4))
            print(tab)
            for epoch in range(0, NUM_EPOCHS):
                train_acc = test(train_loader)
                test_acc = test(test_loader)
                print(f'Epoch: {epoch:},Train Acc: {train_acc},Test Acc: {test_acc}')
                file.write(f'{epoch},{train_acc},{test_acc} \n')
                tab[i,:] = np.array([epoch, train_acc, test_acc])
                i += 1
            print(f"Train finished")
            print(f"execution finish")
        file.close()
    else:
        #Read result for each dataset in order to compare visually scores
        df_IMDB = pd.read_table("C:/Users/Utilisateur/Documents/GitHub/DGCNN/result/result_IMDB-BINARY.txt", delimiter=",",
                                names=['epoch', 'train_acc', 'test_acc'],
                                dtype={'epoch': int, 'train_acc': float, 'test_acc': float})
        df_PROTEINS = pd.read_table("C:/Users/Utilisateur/Documents/GitHub/DGCNN/result/result_PROTEINS.txt", delimiter=",",
                                names=['epoch', 'train_acc', 'test_acc', 'train_loss'],
                                dtype={'epoch': int, 'train_acc': float, 'test_acc': float})
        df_REDDIT = pd.read_table("C:/Users/Utilisateur/Documents/GitHub/DGCNN/result/result_REDDIT-BINARY.txt",
                                    delimiter=",",
                                    names=['epoch', 'train_acc', 'test_acc', 'train_loss'],
                                    dtype={'epoch': int, 'train_acc': float, 'test_acc': float})

        # Create figure
        fig = make_subplots(rows=3, cols=1, subplot_titles=("train_acc", "test_acc"))

        fig.add_trace(
            go.Scatter(x=df_IMDB['epoch'],
                       y=df_IMDB['train_acc'],
                       name="IMDB",
                       legendgroup="IMDB",
                       line=dict(color='blue')),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(x=df_PROTEINS['epoch'],
                       y=df_PROTEINS['train_acc'],
                       name="PROTEINS",
                       legendgroup="PROTEINS",
                       line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_REDDIT['epoch'],
                       y=df_REDDIT['train_acc'],
                       name="REDDIT",
                       legendgroup="REDDIT",
                       line=dict(color='green')),
            row=1,
            col=1
        )

        fig.append_trace(
            go.Scatter(x=df_IMDB['epoch'],
                       y=df_IMDB['test_acc'],
                       name="IMDB",
                       legendgroup="IMDB",
                       line=dict(color='blue'),
                       showlegend=False),
            row=2,
            col=1
        )

        fig.append_trace(
            go.Scatter(x=df_PROTEINS['epoch'],
                       y=df_PROTEINS['test_acc'],
                       name="PROTEINS",
                       egendgroup="PROTEINS",
                       line=dict(color='red'),
                       showlegend=False),
            row=2,
            col=1
        )

        fig.append_trace(
            go.Scatter(x=df_REDDIT['epoch'],
                       y=df_REDDIT['test_acc'],
                       name="REDDIT",
                       legendgroup="REDDIT",
                       line=dict(color='green'),
                       showlegend=False),
            row=2,
            col=1
        )

        fig.update_xaxes(title_text="epoch",
                         range=[min(df_PROTEINS['epoch']) - 1, max(df_PROTEINS['epoch']) + 1],
                         showgrid=False,
                         row=1,
                         col=1)
        fig.update_yaxes(title_text="Train_acc",
                         showgrid=False,
                         row=1,
                         col=1)
        fig.update_xaxes(title_text="epoch",
                         range=[min(df_PROTEINS['epoch']) - 1, max(df_PROTEINS['epoch']) + 1],
                         showgrid=False,
                         row=2,
                         col=1)
        fig.update_yaxes(title_text="Test_acc",
                         showgrid=False,
                         row=2,
                         col=1)
        fig.update_xaxes(title_text="epoch",
                         range=[min(df_PROTEINS['epoch']) - 1, max(df_PROTEINS['epoch']) + 1],
                         showgrid=False,
                         row=3,
                         col=1)

        fig.update_layout(title_text="DGCNN results", height=700)

        fig.write_html('./dataviz/result.html', auto_open=True)

