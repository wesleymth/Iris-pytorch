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
import pandas as pd

from config import (
    LR ,
    BATCH_SIZE ,
    EPOCHS ,
    HIDDEN_LAYER ,
    LOSS_FUNCTION ,
    SEED_NUM,
    TEST_SIZE,
    VALID_SIZE,
    )

torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


def get_iris_names():
    return load_iris().target_names
    
def load_data():
    iris=load_iris()
    return iris.data, iris.target

def convert_data_to_tensordataset(X : np.ndarray, y : np.ndarray):
    X, y = map(torch.tensor,(X, y))
    X = X.float()
    dataset = TensorDataset(X, y)
    return dataset

def convert_data_to_dataloader(X : np.ndarray, y : np.ndarray, batch_size=BATCH_SIZE, shuffle=False):
    tensor_dataset = convert_data_to_tensordataset(X,y)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

def convert_data_to_dataloaders(X: np.ndarray, y: np.ndarray, test_size: float = TEST_SIZE, batch_size=BATCH_SIZE):
    X1, X2, y1, y2 = train_test_split(X, y, test_size=test_size)
    return (convert_data_to_dataloader(X1, y1, batch_size, shuffle=True), 
            convert_data_to_dataloader(X2, y2, batch_size, shuffle=False))

def get_dataloaders(test_size: float = TEST_SIZE, batch_size : int = BATCH_SIZE):
    X, y = load_data()
    return convert_data_to_dataloaders(X, y , test_size=test_size, batch_size=batch_size)
    
def split_get_dataloaders(split_size: float = VALID_SIZE,test_size: float = TEST_SIZE, batch_size : int = BATCH_SIZE):
    X, y = load_data()
    X, X_split, y, y_split = train_test_split(X, y, test_size=split_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return (convert_data_to_dataloader(X_train, y_train, batch_size, shuffle=True), 
            convert_data_to_dataloader(X_test, y_test, batch_size, shuffle=False),
            convert_data_to_dataloader(X_split, y_split, batch_size, shuffle=False)
            )

class IrisClassifier(nn.Module):
    
    def __init__(self, hidden1_dim=50, input_dim=4, output_dim=3):
        super(IrisClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden1_dim)
        self.layer2 = nn.Linear(hidden1_dim, hidden1_dim)
        self.layer3 = nn.Linear(hidden1_dim, output_dim)
        #self.layer4 = nn.Linear(hidden1_dim/2, output_dim)
        self.ReLU = nn.ReLU()
        
        
    def forward(self, x):
        z = self.ReLU(self.layer1(x))
        z = self.ReLU(self.layer2(z))
        #z = self.ReLU(self.layer3(z))
        z = self.layer3(z)
        #z = self.ReLU(self.layer4(z))
        #z = F.softmax(z, dim=1)
        return z

seq_model = nn.Sequential(
    #nn.Dropout(0.25),
    nn.Linear(4, 50),
    nn.ReLU(),
    #nn.Dropout(0.25),
    nn.Linear(50, 50),
    nn.ReLU(),
    #nn.Dropout(0.25),
    nn.Linear(50, 3),
    #nn.ReLU()
)


# def get_batch_classification_accuracy(output, truth):
#     return float(sum(torch.max(output, 1)[1] == truth)/truth.shape[0])
#     prediction = torch.max(output, 1)[1] # get the index of the max log-probability
#     size = truth.shape[0]
#     percentage_correct = float(sum(prediction == truth)/size)
#     return percentage_correct

def save_csv_file(train_loss_history,train_accuracy_history, test_loss_history, test_accuracy_history,
                  #lr, batch_size, epochs, momentum, network_architecture
                  ):
    now = datetime.now() # datetime object containing current date and time
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    dict = {"train_loss_history": train_loss_history, "train_accuracy_history": train_accuracy_history,
        "test_loss_history": test_loss_history, "test_accuracy_history": test_accuracy_history} 
    
    df = pd.DataFrame(dict)
    df.to_csv(f'{dt_string}.csv')

def predict_iris_class(model, features, target_names):
    with torch.no_grad():
        output = model(features)
        prediction = torch.max(output, 0)[1]
        prediction_name = target_names[prediction]
    return prediction_name

def test(model, data_loader, return_loss = False, criterion = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        if return_loss and criterion != None: #######@
            running_loss = 0.0 #######@
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if return_loss and criterion != None: #######@
                loss = criterion(outputs, target) ########
                #batch_size = data.size(0) ########
                running_loss += loss.item() #* batch_size #######@
    if return_loss and criterion != None: #######@
        return correct / total, running_loss / len(data_loader) #######@
    
    return correct / total
    

def train(model, optimizer, criterion, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    running_loss = 0.0 #######@
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        #batch_size = data.size(0) ########
        running_loss += loss.item() #* batch_size #######@
    return running_loss / (len(train_loader))#*batch_size) ########
    
        
def train_model(model, num_epochs, criterion, optimizer, train_loader, test_loader, train_only=False):
    history = {'train_acc' : [], 
               'train_loss' : [],
               'test_acc' : [],
               'test_loss' : []
               }
    for epoch in tqdm.trange(num_epochs):
        #train(model, optimizer, criterion, train_loader)
        history['train_loss'].append(train(model, optimizer, criterion, train_loader))
        history['train_acc'].append(test(model, train_loader))
        if not(train_only):
            test_acc, test_loss = test(model, test_loader, return_loss=True, criterion=criterion)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
        else:
            history['test_loss'].append('train_only=True')
            history['test_acc'].append('train_only=True')
    
    return history
        

# def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
 
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)
 
#         model.train()  # Set model to training mode
         
#         running_loss = 0.0
 
#             # Iterate over data.
#         for inputs, labels in dataloaders:
             
#             inputs = inputs.to(device)
#             labels = labels.to(device)
 
#             # zero the parameter gradients
#             optimizer.zero_grad()
 
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
             
#             loss = criterion(outputs, labels)
 
#             # backward + optimize only if in training phase
#             loss.backward()
#             optimizer.step()
                  
#             # statistics
#             running_loss += loss.item() * inputs.size(0)
                  
#             scheduler.step()
 
#         epoch_loss = running_loss / dataset_sizes
         
#         print(f' Loss: {epoch_loss:.4f} ')
        

        
    
def save_hyper_parameters(
    file_name,
    batch_size, 
    epochs, 
    final_train_loss, 
    final_train_accuracy,
    final_test_loss,
    final_test_accuracy,
    model,
    optimizer
    ):
    f=open(file_name,"a")
    now = datetime.now() # datetime object containing current date and time
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    f.write(f'Run : {dt_string}\n')
    f.write(f'Batch size : {batch_size}\n')
    f.write(f'epochs : {epochs}\n')
    f.write(f'final_train_loss : {final_train_loss}\n')
    f.write(f'final_train_accuracy : {final_train_accuracy}\n')
    f.write(f'final_test_loss : {final_test_loss}\n')
    f.write(f'final_test_accuracy : {final_test_accuracy}\n')
    f.write(f'model : {model}\n')
    f.write(f'optimizer : {optimizer}\n')
    f.write("#####################################################################################\n")
    f.close()

def plot_loss_accuracy_over_epoch(train_loss_history,train_accuracy_history, test_loss_history, test_accuracy_history):
    fig, axs = plt.subplots(2,2,sharex=True, sharey=True)
    
    # Plotting the accuracies
    
    axs[0,0].plot(test_accuracy_history, 'c--')
    axs[0,0].set_ylabel("accuracy")
    axs[0,0].set_title(f'TEST final : {round(test_accuracy_history[-1],4)*100}%')
    
    axs[0,1].set_title(f'TRAINING final : {round(train_accuracy_history[-1],4)*100}%')
    axs[0,1].plot(train_accuracy_history, 'm--')
    
    
    # Plotting the losses
    
    axs[1,0].plot(test_loss_history, 'b-')
    axs[1,0].set_ylabel("loss")
    axs[1,0].set_xlabel("epochs")
    axs[1,0].set_title(f'TEST final : {round(test_loss_history[-1],3)}')
    
    
    axs[1,1].plot(train_loss_history, 'r-')
    axs[1,1].set_xlabel("epochs")
    axs[1,1].set_title(f'TRAIN final : {round(train_loss_history[-1],3)}')
    
    plt.tight_layout()
    plt.savefig(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    
def main():
    
    train_dataloader, test_dataloader = get_dataloaders()
    
    #model = IrisClassifier(HIDDEN_LAYER)
    model = seq_model
    #opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # (train_loss_history,
    # train_accuracy_history,
    # test_loss_history,
    # test_accuracy_history) = fit(model, epochs, train_dataloader, optimizer=opt, criterion=loss_function, save_histories=True, test_dataloader=test_dataloader)
    
    history = train_model(model, EPOCHS, LOSS_FUNCTION, optimizer, train_dataloader, test_dataloader)
    
    
    # save_csv_file(train_loss_history,train_accuracy_history, test_loss_history, test_accuracy_history)
    
    save_hyper_parameters(
    'hyper_params.txt',
    BATCH_SIZE, 
    EPOCHS, 
    history['train_loss'][-1], 
    history['train_acc'][-1],
    history['test_loss'][-1],
    history['test_acc'][-1],
    model,
    optimizer
    )
    
    print(f'Testing the model on the whole test set gives {round(test(model,test_dataloader),4)*100}% accuracy')

    plot_loss_accuracy_over_epoch(history['train_loss'],history['train_acc'], history['test_loss'], history['test_acc'])
    

if __name__ == '__main__':
    main()