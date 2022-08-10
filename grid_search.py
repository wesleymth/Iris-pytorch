import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset
import torch


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


def plot_change_in_lr(lr_max, lr_min, number_points, testing_data, training_data, output_path = "lr_hyperparam_behaviour.png"):
    lr_to_try = np.linspace(lr_min, lr_max, number_points)
    accuracy_tests = np.zeros_like(lr_to_try)
    
    for idx, lr in enumerate(lr_to_try):
        model = IrisClassifier(HIDDEN_LAYER)
        opt = optim.Adam(model.parameters(), lr=lr)
        fit(model, EPOCHS, training_data, testing_data, opt, LOSS_FUNCTION, False)
        accuracy_tests[idx] = test(model,testing_data)
    
    plt.plot(lr_to_try, accuracy_tests,'c-')
    plt.ylabel("accuracy (percentage)")
    plt.xlabel("learning rate")
    plt.title("Change in classification accuracy depending on lr (other params fixed)")
    plt.tight_layout()
    plt.savefig(output_path)
    
    accuracy_tests = torch.tensor(accuracy_tests)
    _, index = torch.max(accuracy_tests.data, 0)
    
    print(f'The best lrs are : {lr_to_try[index]}')
    
X_train, X_test,y_train,y_test, y_names = load_data()
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_dataloader, test_dataloader = get_data_as_dataloaders(train_dataset, test_dataset, BATCH_SIZE)
    
print(plot_change_in_lr(0.1, 0.0001, 100, test_dataloader, train_dataloader, "lr_zoom_in.png"))