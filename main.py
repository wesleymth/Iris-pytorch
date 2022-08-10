from calendar import EPOCH
import random
from datetime import datetime
from webbrowser import get

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

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LR = 0.007163636363636364
BATCH_SIZE = 10
EPOCHS = 100
HIDDEN_LAYER = 50
LOSS_FUNCTION = nn.CrossEntropyLoss()

def load_data(test_size=0.2):
    iris=load_iris()
    X=iris.data
    y=iris.target
    y_names = iris.target_names
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
    X_train, y_train, X_test, y_test = map(
    torch.tensor, (X_train, y_train, X_test, y_test)
    )
    return X_train.float(),X_test.float(),y_train,y_test, y_names

def get_data_as_dataloaders(train_ds : TensorDataset, test_ds : TensorDataset, batch_size : int):
    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size * 2),
    )
    

class IrisClassifier(nn.Module):
    
    def __init__(self, hidden1_dim, input_dim=4, output_dim=3):
        super(IrisClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden1_dim)
        self.layer2 = nn.Linear(hidden1_dim, hidden1_dim)
        self.layer3 = nn.Linear(hidden1_dim, output_dim)
        #self.layer4 = nn.Linear(hidden1_dim/2, output_dim)
        self.ReLU = nn.ReLU()
        
        
    def forward(self, x):
        z = self.ReLU(self.layer1(x))
        z = self.ReLU(self.layer2(z))
        z = self.ReLU(self.layer3(z))
        #z = self.ReLU(self.layer4(z))
        #z = F.softmax(z, dim=1)
        return z


def fit(model, epochs, train_dataloader, test_dataloader, optimizer=None, criterion=nn.CrossEntropyLoss(), save_histories=True):
    
    if save_histories:
        train_loss_history=np.zeros((epochs,))
        train_accuracy_history=np.zeros((epochs,))
        test_loss_history=np.zeros((epochs,))
        test_accuracy_history=np.zeros((epochs,))
    
    if optimizer == None: 
        optimizer=optim.Adam(model.parameters(), lr=0.01)
    
    #for epoch in range(epochs):  
    for epoch in tqdm.trange(epochs) : #tqdm.trange(epochs) #tqdm_notebook(range(epochs))
        model.train() # Training the model
        for x_training_batches, y_training_labels in train_dataloader:
            
            output = model(x_training_batches.float())
            train_loss = criterion(output, y_training_labels)

            # Zero gradients
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if save_histories:
                with torch.no_grad():
                    # Save the loss of the training set at each epoch
                    train_loss_history[epoch] += train_loss.item()
                    train_accuracy_history[epoch] += get_batch_classification_accuracy(output,y_training_labels)
        
        model.eval() # Evaluating the model
        
        if save_histories:
            for x_test_batches, y_test_labels in test_dataloader:
                with torch.no_grad():
                    test_output = model(x_test_batches)
                    test_loss = criterion(test_output, y_test_labels)
                    test_loss_history[epoch] += test_loss.item()
                    
                    test_accuracy_history[epoch] += get_batch_classification_accuracy(test_output,y_test_labels)
        
    if save_histories: 
        number_train_batches = len(train_dataloader)
        number_test_batches = len(test_dataloader)
                    
        train_loss_history /= number_train_batches
        train_accuracy_history /= number_train_batches
        test_loss_history /= number_test_batches
        test_accuracy_history /= number_test_batches
        
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

def predict_iris_class(model, features, target_names):
    with torch.no_grad():
        output = model(features)
        prediction = torch.max(output, 0)[1]
        prediction_name = target_names[prediction]
    return prediction_name

def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return round(correct / total,4)*100
        
    
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
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lr = LR
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    momentum = 0.9
    hidden_nums = HIDDEN_LAYER

    X_train, X_test,y_train,y_test, y_names = load_data()
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_dataloader, test_dataloader = get_data_as_dataloaders(train_dataset, test_dataset, batch_size)

    model = IrisClassifier(hidden_nums)
    #opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_function = LOSS_FUNCTION
    
    (train_loss_history,
    train_accuracy_history,
    test_loss_history,
    test_accuracy_history) = fit(model, epochs, train_dataloader, test_dataloader, optimizer=opt, criterion=loss_function)
    
    # save_csv_file(train_loss_history,train_accuracy_history, test_loss_history, test_accuracy_history)
    
    save_hyper_parameters(
    'hyper_params.txt',
    batch_size, 
    epochs, 
    train_loss_history[-1], 
    train_accuracy_history[-1],
    test_loss_history[-1],
    test_accuracy_history[-1],
    model,
    opt
    )
    
    prediction_test_index = random.randint(0,len(X_test))
    predicted_class = predict_iris_class(model, X_test[prediction_test_index], y_names)
    actual_class = y_names[y_test[prediction_test_index]]
    print(f'Predicted class : {predicted_class}\nCorrect class : {actual_class}')
    
    print(f'Testing the model on the whole test set gives {test(model,test_dataloader)}% accuracy')

    plot_loss_accuracy_over_epoch(train_loss_history,train_accuracy_history, test_loss_history, test_accuracy_history)
    

if __name__ == '__main__':
    main()