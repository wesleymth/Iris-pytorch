import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring



torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from main import (
    LOSS_FUNCTION,
    IrisClassifier,
    fit,
    test,
    load_data,
    get_data_as_dataloaders,
    predict_iris_class,
    LR,
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_LAYER,
)

  
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
        ],
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

    plt.plot(train_loss, '-c', label='training')
    plt.plot(valid_loss, '-m', label='validation')
    plt.legend()
    plt.show()
    
    # Test a GridSearch
    
    params = {
    'lr': [0.001, 0.05, 0.01],
    'max_epochs': [100, 200],
    'module__hidden1_dim': [30, 50, 55],
    }
    gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

    gs.fit(X, iris.target)
    print(gs.best_score_, gs.best_params_)
    

    
if __name__ == '__main__':
    main_skorch()
    