import random
import itertools
from timeit import timeit

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
    seq_model,
    train_model,
    get_dataloaders
)

from cross_validation import (
    train_model_with_CV,
)

torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

class ParameterGrid():
    def __init__(self, dictionnary : dict) -> None:
        super(ParameterGrid, self).__init__()
        self.dictionnary = dictionnary
    
    def get_combinations(self) -> list:
        return [*itertools.product(*self.dictionnary.values())]      

class GridSearch():
    def __init__(self, ) -> None:
        pass

def main_GS():
    dictionnary = {'lr' : [0.001, 0.002, 0.003],
                       'bs' : [8, 16, 32],
                       'hidden' : [45, 50, 55]
        }
    grid = ParameterGrid(dictionnary=dictionnary)
    
    iterable_list = grid.get_combinations()
    
    results = {'best_combination' : None, 
               'cominations' : iterable_list,
               'test_acc' : [],
               'test_loss' : [],
               'train_acc' : [],
               'train_loss' :  []
               }
    
    previous_test_acc = 0.0
    previous_test_loss = 1e27
    previous_train_acc = 0.0
    previous_train_loss = 1e27

    
    
    print(iterable_list)
    
    print(f'--------------Total number of trys = {len(iterable_list)}--------------')
    
    idx = 0
    
    for lr, bs, hidden in iterable_list:
        print(f'no{idx+1} : Trying the hyperparameters : lr={lr} , bs={bs}, hidden={hidden}')
        train_dataloader, test_dataloader = get_dataloaders(batch_size=bs)
        model = IrisClassifier(hidden1_dim=hidden)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        history = train_model(model, EPOCHS, LOSS_FUNCTION, optimizer, train_dataloader, test_dataloader)
        results['test_acc'].append(history['test_acc'][-1])
        results['test_loss'].append(history['test_loss'][-1])
        results['train_acc'].append(history['train_acc'][-1])
        results['train_loss'].append(history['train_loss'][-1])
        if previous_test_acc < history['test_acc'][-1]:
            results['best_combination'] = {'lr' : lr, 'bs' : bs, 'hidden' : hidden}
            previous_test_acc = history['test_acc'][-1]
            previous_test_loss = history['test_loss'][-1]
            previous_train_acc = history['train_acc'][-1]
            previous_train_loss = history['train_loss'][-1]
        elif previous_test_acc == history['test_acc'][-1]:
            if previous_test_loss > history['test_loss'][-1]:
                results['best_combination'] = {'lr' : lr, 'bs' : bs, 'hidden' : hidden}
                previous_test_acc = history['test_acc'][-1]
                previous_test_loss = history['test_loss'][-1]
                previous_train_acc = history['train_acc'][-1]
                previous_train_loss = history['train_loss'][-1]
        idx+=1
    
    print(results)
    print('Best hyperparams are (lr, bs, hideen) = {}'.format(results['best_combination']))

if __name__ == '__main__':
    main_GS()