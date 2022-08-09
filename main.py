import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn       
import torch.nn.functional as F 
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tqdm
from tqdm.notebook import tqdm_notebook
import pandas as pd

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def load_data(test_size=0.2):
    iris=load_iris()
    X=iris.data
    y=iris.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    X_train, y_train, X_test, y_test = map(
    torch.tensor, (X_train, y_train, X_test, y_test)
)
    return X_train.float(),X_test.float(),y_train,y_test

def get_data_as_dataloaders(train_ds : TensorDataset, test_ds : TensorDataset, batch_size : int):
    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size * 2),
    )
    

class IrisClassifier(nn.Module):
    
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.layer1 = nn.Linear(4, 120)
        self.layer2 = nn.Linear(120, 3)
        #self.layer3 = nn.Linear(64, 3)
        #self.layer4 = nn.Linear(12, 3)
        self.ReLU = nn.ReLU()
        
        
    def forward(self, x):
        z = self.ReLU(self.layer1(x))
        z = self.ReLU(self.layer2(z))
        #z = self.ReLU(self.layer3(z))
        #z = self.ReLU(self.layer4(z))
        z = F.softmax(z, dim=1)
        return z


def get_model_with_optimizer(lr, momentum):
    model = IrisClassifier()
    return model, optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def fit(model, epochs, train_dataloader, test_dataloader, optimizer=None, criterion=nn.CrossEntropyLoss()):
    
    train_loss_history=np.zeros((epochs,))
    train_accuracy_history=np.zeros((epochs,))
    test_loss_history=np.zeros((epochs,))
    test_accuracy_history=np.zeros((epochs,))
    
    if optimizer == None: 
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    #for epoch in range(epochs):  
    for epoch in tqdm.trange(epochs) : #tqdm.trange(epochs) #tqdm_notebook(range(epochs))
        model.train() # Training the model
        for x_training_batches, y_training_labels in train_dataloader:
            
            size = len(train_dataloader.dataset)
            num_batches = len(train_dataloader)
            
            output = model(x_training_batches.float())
            loss = criterion(output, y_training_labels)
            
            
            
            
            # Zero gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            with torch.no_grad():
                # Save the loss of the training set at each epoch
                train_loss_history[epoch] += loss.item()
                train_accuracy_history[epoch] += get_batch_classification_accuracy(output,y_training_labels)
        
        model.eval() # Evaluating the model
        
            
        for x_test_batches, y_test_labels in test_dataloader:
            with torch.no_grad():
                test_output = model(x_test_batches)
                loss = criterion(test_output, y_test_labels)
                test_loss_history[epoch] += loss.item()
                
                test_accuracy_history[epoch] += get_batch_classification_accuracy(test_output,y_test_labels)
        
        
                
                
    train_loss_history /= len(train_dataloader)
    train_accuracy_history /= len(train_dataloader)
    test_loss_history /= len(test_dataloader)
    test_accuracy_history /= len(test_dataloader)
    
    return (train_loss_history,
            train_accuracy_history,
            test_loss_history,
            test_accuracy_history)

def get_batch_classification_accuracy(output, truth):
    return float(sum(torch.max(output, 1)[1] == truth)/truth.shape[0])
    prediction = torch.max(output, 1)[1] # get the index of the max log-probability
    size = truth.shape[0]
    percentage_correct = float(sum(prediction == truth)/size)
    return percentage_correct

def save_csv_file(train_loss_history,train_accuracy_history, test_loss_history, test_accuracy_history,
                  #lr, batch_size, epochs, momentum, network_architecture
                  ):
    now = datetime.now() # datetime object containing current date and time
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    dict = {"train_loss_history": train_loss_history, "train_accuracy_history": train_accuracy_history,
        "test_loss_history": test_loss_history, "test_accuracy_history": test_accuracy_history} 
        #"lr" : lr, "batch_size" : batch_size, "epochs" : epochs, "momentum" : momentum, "network_architecture" : network_architecture}
    
    df = pd.DataFrame(dict)
    df.to_csv(f'{dt_string}.csv')
    

    

#def get_epoch_classification_accuracy()
    
def main():
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = 0.001
    batch_size = 10
    epochs = 500
    momentum = 0.9
    network_architecture = (4,120,3)

    X_train,X_test,y_train,y_test = load_data()
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader, test_dataloader = get_data_as_dataloaders(train_dataset, test_dataset, batch_size)

    model, opt = get_model_with_optimizer(lr, momentum)
    loss_function = nn.CrossEntropyLoss()

    (train_loss_history,
    train_accuracy_history,
    test_loss_history,
    test_accuracy_history) = fit(model, epochs, train_dataloader, test_dataloader, optimizer=opt, criterion=loss_function)
    
    
    save_csv_file(train_loss_history,train_accuracy_history, test_loss_history, test_accuracy_history)
                  #lr, batch_size, epochs, momentum, network_architecture)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 20), sharex=True)
    ax1.plot(test_accuracy_history)
    ax1.set_ylabel("testing accuracy")
    ax2.plot(test_loss_history)
    ax2.set_ylabel("testing loss")
    ax3.plot(train_accuracy_history)
    ax3.set_ylabel("training accuracy")
    ax4.plot(train_loss_history)
    ax4.set_ylabel("training loss")
    ax4.set_xlabel("epochs")
    plt.show()

if __name__ == '__main__':
    main()