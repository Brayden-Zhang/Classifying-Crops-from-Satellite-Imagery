


# Set training config
root = "./files"
experiment_name = "seq2one-poc"
gpu = 0
min_epochs, max_epochs = 3, 30

# Set the hyperparameters
batch_size = 64
learning_rate = 0.001
hidden_size = 128
num_layers = 3
early_stopping_patience = 15