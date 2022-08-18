import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV
import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, ProgressBar
import tqdm

from config import (
    LR ,
    BATCH_SIZE ,
    EPOCHS ,
    HIDDEN_LAYER ,
    LOSS_FUNCTION ,
    SEED_NUM,
    )

from main import (
    IrisClassifier,
    plot_loss_accuracy_over_epoch,
)

torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

def tqdm_iterator(dataset, **kwargs):
    return tqdm.tqdm(torch.utils.data.DataLoader(dataset, **kwargs))

  
def main_skorch():
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    iris = load_iris()
    
    

    net = NeuralNetClassifier(
        module = IrisClassifier,
        module__hidden1_dim = HIDDEN_LAYER,
        module__input_dim = 4,
        module__output_dim = 3,
        max_epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        callbacks=[
        EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
        #ProgressBar()
        ],
        verbose = 1,
        iterator_train__shuffle=True,
        #iterator_train=tqdm_iterator,
    )
    
    X = torch.tensor(iris.data)
    X = X.float()
    
    print(type(iris.data[0,0]))
    print(type(iris.target[0]))
    
    net.fit(X,iris.target)
    
    print(f'net.get_all_learnable_params() => {net.get_all_learnable_params()}')
    
    X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.8)
    
    """X_test_tensor = torch.from_numpy(X_test)
    X_test_tensor = X_test_tensor.float()
    
    
    labels_index = net.predict(X_test_tensor)
    
    y_test = torch.tensor(y_test)
    
    print(labels_index.shape)
    print(type(labels_index))
    print(type(labels_index[0]))
    
    print(y_test.shape)
    print(type(y_test))
    print(type(y_test[0]))"""
    
    X_test_tensor = torch.from_numpy(X_test)
    X_test_tensor = X_test_tensor.float()
    labels_index = net.predict(X_test_tensor)
    
    
    
    
    print(float(sum(labels_index == y_test)/y_test.shape[0])*100)
    
    history = net.history
    
    #history.to_file("saving_history.json")
    
    train_loss = net.history[:, 'train_loss']
    valid_loss = net.history[:, 'valid_loss']
    train_acc = net.history[:, 'train_acc']
    valid_acc = net.history[:, 'valid_acc']
    
    plot_loss_accuracy_over_epoch(train_loss, train_acc, valid_loss, valid_acc)
    

    # plt.plot(train_loss, '-c', label='training')
    # plt.plot(valid_loss, '-m', label='validation')
    # plt.legend()
    # plt.show()
    
    # Test a GridSearch
    net2 = net = NeuralNetClassifier(
        module = IrisClassifier,
        max_epochs=EPOCHS,
        #batch_size=BATCH_SIZE,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        callbacks=[
        EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
        #ProgressBar()
        ],
        verbose = 0,
        iterator_train__shuffle=True,
        #iterator_train=tqdm_iterator,
    )
    
    params = {
    'lr': [0.001, 0.005, 0.01],
    #'max_epochs': [100],
    'module__hidden1_dim': [30, 45, 50, 55],
    'batch_size' : [10, 64, 32]
    }
    
    # grid_search = GridSearchCV(net2, params, refit=False, cv=5, scoring='accuracy', verbose=3)
    # grid_search.fit(X, iris.target)
    # print(grid_search.best_score_, grid_search.best_params_)
    
    n_iter_search = 15
    random_search = RandomizedSearchCV(net2, params, refit=False, cv=3, scoring='accuracy', n_iter=n_iter_search)#, verbose=3)
    random_search.fit(X, iris.target)
    print(random_search.best_score_, random_search.best_params_)

    
if __name__ == '__main__':
    main_skorch()
    