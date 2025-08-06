import numpy as np
import torch
import torch.nn as nn

from moore_penrose_regression_torch import moore_penrose_regression_torch

def build_e2nn_one_layer(x_train, y_train, emulator_train, 
                         activation="fourier", freq=1.0, device="cuda",
                         dtype=torch.double):
    """
    Builds an e2nn model with one hidden layer
    Returns e2nn model and maximum weight from training
    """

    # convert y_train if needed (x_train and emulator_train are
    # handled automatically)
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train)
    y_train = y_train.to(device).to(dtype)

    # setup parameters
    N_EMULATORS = emulator_train.shape[1]
    n_train, N_DIM = x_train.shape
    n_nodes_L1 = 2*n_train  #8*n_train  #max(2*n_train, 5_000)  #

    def get_initializer(bias_range):
        """ Returns function that initializes weights and biases"""
        def init_weights_bias(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.uniform_(
                    m.bias, a=bias_range[0], b=bias_range[1])
        return init_weights_bias


    # Define the neural network model
    class ModelOneLayerSwish(nn.Module):
        def __init__(self, x_size, em_size, hidden_size, device="cuda", 
                     dtype=torch.double):
            super().__init__()  # Call the parent class's __init__
            self.fc1 = nn.Linear(x_size, hidden_size)   # fully connected layer
            self.fc2 = nn.Linear(hidden_size + x_size + em_size, 1)
            self.swish = nn.SiLU()
            self.device = device
            self.dtype = dtype
   
        def convert(self, var):
            if isinstance(var, np.ndarray):
                var = torch.from_numpy(var)
            var = var.to(self.device).to(self.dtype)
            return var
    
        def forward(self, x, em):
            x = self.convert(x)
            em = self.convert(em)
            L = self.fc1(x)
            L = self.swish(L)
            L = torch.cat((L, x, em), dim=1)
            L = self.fc2(L)
            return L

        def get_hidden_activations(self, x, em):
            x = self.convert(x)
            em = self.convert(em)
            L = self.fc1(x)
            L = self.swish(L)
            L = torch.cat((L, x, em), dim=1)
            return L


    # Define the neural network model
    class ModelOneLayerFourier(nn.Module):
        def __init__(self, x_size, em_size, hidden_size, freq=1.0,
                     device="cuda", dtype=torch.double):
            super().__init__()  # Call the parent class's __init__
            self.fc1 = nn.Linear(x_size, hidden_size)   # fully connected layer
            self.fc2 = nn.Linear(hidden_size + x_size + em_size, 1)
            self.freq = freq
            self.device = device
            self.dtype = dtype

        def convert(self, var):
            if isinstance(var, np.ndarray):
                var = torch.from_numpy(var)
            var = var.to(self.device).to(self.dtype)
            return var

        def forward(self, x, em):
            x = self.convert(x)
            em = self.convert(em)
            L = self.fc1(x)
            L = torch.sin(self.freq * L)
            L = torch.cat((L, x, em), dim=1)
            L = self.fc2(L)
            return L

        def get_hidden_activations(self, x, em):
            x = self.convert(x)
            em = self.convert(em)
            L = self.fc1(x)
            L = torch.sin(self.freq * L)
            L = torch.cat((L, x, em), dim=1)
            return L


    # bias
    if activation=="fourier":
        nn_model = ModelOneLayerFourier(
            N_DIM, N_EMULATORS, n_nodes_L1, freq, device, dtype
        )
        bias_range = [0, 2*np.pi]
    elif activation=="swish":
        nn_model = ModelOneLayerSwish(
            N_DIM, N_EMULATORS, n_nodes_L1, device, dtype
        )
        bias_range = [-4, 4]
    else:
        Error("Activation not 'fourier' varient or 'swish'")

    nn_model.to(device).to(dtype)

    # initialize weights and biases
    nn_model.apply(get_initializer(bias_range))

    # Rapid training
    x_hidden = nn_model.get_hidden_activations(x_train, emulator_train)


    Beta = moore_penrose_regression_torch(
        x_hidden, y_train, device=device, dtype=dtype)

    new_bias = Beta[0]    # shape [1,1]
    new_weights = Beta[1:].reshape(1, -1)

    with torch.no_grad():   # do not track gradients during operation
        nn_model.fc2.bias.copy_(new_bias)
        nn_model.fc2.weight.copy_(new_weights)
    
    max_weight = torch.max(new_weights)

    return(nn_model, max_weight)

