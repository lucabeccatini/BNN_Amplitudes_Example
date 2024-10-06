# Bayesian neural network with variational inference


import torch
from torch import nn
import numpy as np
import os



# Variational Bayesian Linear Layer
# ---------------------------------

class VBLinear(nn.Module):
    def __init__(self, in_features, out_features, net_activation_inner, prior_prec=1.0):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features
        self.prior_prec = prior_prec
        self.random = None
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.net_activation_inner = net_activation_inner
        self.reset_parameters()

    def reset_parameters(self):
        if self.net_activation_inner=="tanh":
            nn.init.xavier_normal_(self.mu_w.data)                                             # Xavier/Glorot initialization
        elif self.net_activation_inner in ["relu", "elu"]:
            nn.init.kaiming_normal_(self.mu_w.data, mode='fan_in', nonlinearity='relu')       # Kaiming/He initialization
        self.logsig2_w.data.zero_().normal_(-9, 0.001)
        self.bias.data.zero_()

    def reset_random(self):
        self.random = None

    def KL(self):
        logsig2_w = self.logsig2_w.clamp(-20, 11)
        kl = 0.5 * (self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                    - logsig2_w - 1 - np.log(self.prior_prec)).sum()
        return kl

    def forward(self, input):
        if self.training:
            mu_out = nn.functional.linear(input, self.mu_w, self.bias)
            logsig2_w = self.logsig2_w.clamp(-20, 11)
            s2_w = logsig2_w.exp()
            var_out = nn.functional.linear(input.pow(2), s2_w) #+ 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

        else:
            logsig2_w = self.logsig2_w.clamp(-20, 11)
            logsig2_w = self.logsig2_w
            if self.random is None:
                self.random = torch.randn_like(self.logsig2_w)
            s2_w = logsig2_w.exp()
            weight = self.mu_w + s2_w.sqrt() * self.random
            return nn.functional.linear(input, weight, self.bias)



# Bayesian neural network
# -----------------------

class BNN_model(nn.Module): 
    def __init__(self, net_training_size, net_inner_layers, net_input_dim, net_activation_inner, net_batchnorm=False, net_activation_output=None):
        super(BNN_model, self).__init__() 

        self.training_size = net_training_size
        self.input_dim = net_input_dim
        self.all_layers = []
        self.vb_layers = []
        self.best_epoc_tot = 0
        self.best_epoc_mse = 0

        # data prepossessing parameters 
        self.x_mean = 0
        self.x_std = 0
        self.y_mean = 0 
        self.y_std = 0 


        net_inner_layers = [self.input_dim] + net_inner_layers + [2] 
        for i in range(len(net_inner_layers)-1):
            vb_layer = VBLinear(net_inner_layers[i], net_inner_layers[i+1], net_activation_inner)
            self.vb_layers.append(vb_layer)
            self.all_layers.append(vb_layer)
            if i < (len(net_inner_layers)-2): # skipping last layer
                if net_batchnorm:
                    self.all_layers.append(nn.BatchNorm1d(net_inner_layers[i+1]))
                    
                if net_activation_inner.lower() == "tanh":
                    self.all_layers.append(nn.Tanh())
                elif net_activation_inner.lower() == "relu":
                    self.all_layers.append(nn.ReLU())
                elif net_activation_inner.lower() == "elu":
                    self.all_layers.append(nn.ELU())
                else:
                    raise NotImplementedError("Option for activation function of inner layers is not implemented! Given: {}".format(net_activation_inner))
            if i == (len(net_inner_layers)-2): # last layer
                if net_activation_output == "tanh":
                    self.all_layers.append(nn.Tanh())
                elif net_activation_output == "relu":
                    self.all_layers.append(nn.ReLU())
                elif net_activation_output != None:
                    raise NotImplementedError("Option for activation function of output layers is not implemented! Given: {}".format(net_activation_output))
        self.model = nn.Sequential(*self.all_layers)

    def forward(self, x_fw):
        x_fw = x_fw[:, 0:self.input_dim]
        y_net = self.model(x_fw)
        return y_net 

    def KL(self):
        kl = 0
        for vb_layer in self.vb_layers:
            kl += vb_layer.KL()
        return kl / self.training_size

    def neg_log_gauss(self, outputs, targets):
        mu = outputs[:, 0]
        logsigma2 = outputs[:, 1]
        out = torch.square(mu - targets) / ( 2 * logsigma2.exp() + 1e-10  ) + 1./2. * logsigma2
        return torch.mean(out)
    
    def reset_random(self):
        for module in self.modules():
            if isinstance(module, VBLinear):
                module.reset_random()
        return 




class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def create_directory(dir_path):
    max_trial = 10                     # Maximum number of trials to create a unique directory name
    # Create a unique directory name
    print("")
    for counter in range(max_trial):
        dir_name = f"{dir_path}_{counter}"
        if not os.path.exists(dir_name):
            break
    if counter == max_trial-1:
        raise ValueError(f"Could not create a unique directory name after {max_trial} trials. Please check the directory path.")
    # Create the directory
    os.makedirs(dir_name)
    print(f"Created directory: {dir_name}") 
    print("")
    
    return dir_name

