import torch
from torch import nn

# Chargement des datasets
from torch_geometric.data import DataLoader

# Récupération des datasets
from torch_geometric.datasets import TUDataset


# Import de la class model
from model import Dgcnn

# Import de la fonction de tranformation des données
from utils import TransFormGetDegree

# Optimisateur
from torch.optim import Adam

# Fonction permettant l'entrainement du modèle sur un ensemble de données dédiées à l'entrainement
def train():
    model.train()
    running_loss = 0
    for data in train_loader:
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


dataset = TUDataset(root='C:/Users/Utilisateur/Documents/GitHub/DGCNN/data', name='PROTEINS', transform = TransFormGetDegree())

print(f'Dataset: {dataset}:')
print('======================')
print(f'Nombre de graphes: {len(dataset)}')
print(f'Nombre d\'attributs: {dataset.num_features}')
print(f'Numbre de classes: {dataset.num_classes}')

data = dataset[0]  # on prend le premier
print()
print(data.x)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Nombre de nœuds : {data.num_nodes}')
print(f'Nombre d\'arêtes : {data.num_edges}')
print(f'Degré moyen : {data.num_edges / data.num_nodes:.2f}')
print(f'Nœuds isolés : {data.contains_isolated_nodes()}')
print(f'Boucles : {data.contains_self_loops()}')
print(f'Est non-orienté : {data.is_undirected()}')

BATCH_SIZE = 50
torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:700]
test_dataset = dataset[700:]

print(f'Nombre de graphes pour l\'apprentissage: {len(train_dataset)}')
print(f'Nombre de graphes pour le test : {len(test_dataset)}')

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


NUM_FEATURES, NUM_CLASSES = dataset.num_features, dataset.num_classes
print(f'Nombre d\'attributs: {dataset.num_features}, Nombre de classes: {dataset.num_classes}')

model = Dgcnn(NUM_FEATURES, NUM_CLASSES)
print(model)

optimizer = Adam(model.parameters())
loss_criterion = nn.NLLLoss()



NUM_EPOCHS = 5
for epoch in range(1, NUM_EPOCHS):
    train_loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:}, Train Acc: {train_acc}, Test Acc: {test_acc}, Train loss: {train_loss}')


