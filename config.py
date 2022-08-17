import torch

LR = 0.001#0.05
BATCH_SIZE = 10#32#64
EPOCHS = 200#100
HIDDEN_LAYER = 50#45#50 
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED_NUM = 0
NUM_FOLDS = 5
TEST_SIZE = 0.2
VALID_SIZE = 0.1

#0.86 {'batch_size': 32, 'lr': 0.001, 'module__hidden1_dim': 50}