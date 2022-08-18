import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, SubsetRandomSampler
import torch.nn as nn       
import torch.nn.functional as F 
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tqdm
import pandas as pd
from sklearn.model_selection import KFold

from config import (
    LR ,
    BATCH_SIZE ,
    EPOCHS ,
    HIDDEN_LAYER ,
    LOSS_FUNCTION ,
    SEED_NUM,
    NUM_FOLDS,
    VALID_SIZE,
    TEST_SIZE,
    )

from main import (
    IrisClassifier,
    seq_model,
    load_data,
    test,
    train_model,
    load_data,
    convert_data_to_tensordataset,
    convert_data_to_dataloader,  
)

torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


def train_model_with_CV(model:nn.Module, dataset:TensorDataset, optimizer:optim.Optimizer, batch_size:int=BATCH_SIZE, num_folds:int= NUM_FOLDS):
    
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    histories_folds = []
    
    for fold, (train_idx,test_idx) in enumerate(kfold.split(dataset)):
        
        print(f'-------------------------------fold no:{fold}-------------------------------')
        train_subsampler = SubsetRandomSampler(train_idx)
        test_subsampler = SubsetRandomSampler(test_idx)

        trainloader = DataLoader(
                            dataset, 
                            batch_size=batch_size, sampler=train_subsampler)
        testloader = DataLoader(
                            dataset,
                            batch_size=batch_size, sampler=test_subsampler)

        model.apply(reset_weights)
        
        histories_folds.append(train_model(model, EPOCHS, LOSS_FUNCTION, optimizer, trainloader, testloader))
        
    return histories_folds
        
        
def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def main_CV():
    X, y = load_data()
    X, X_valid, y, y_valid = train_test_split(X, y, test_size=VALID_SIZE)
    
    train_and_test_dataset = convert_data_to_tensordataset(X, y)
    xtra_valid_loader = convert_data_to_dataloader(X_valid, y_valid, batch_size=BATCH_SIZE, shuffle=False)
    
    train_dl = convert_data_to_dataloader(X, y, BATCH_SIZE, shuffle=True)
    
    model = IrisClassifier(HIDDEN_LAYER)
    #model = seq_model
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    histories_folds = train_model_with_CV(model, train_and_test_dataset, optimizer, BATCH_SIZE, NUM_FOLDS)
    
    for fold, history in enumerate(histories_folds):
        print('For fold no{}, the final test accuracy was : {:.4f}%'.format(fold+1, history['test_acc'][-1]*100))
    
    train_model(model, EPOCHS, LOSS_FUNCTION, optimizer, train_dl , test_loader=None, train_only=True)
    
    print(f'On the Xtra validation set the model had an accuracy of : {round(test(model,xtra_valid_loader),4)*100}%')
    
if __name__ == '__main__':
    main_CV()
