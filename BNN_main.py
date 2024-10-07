import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import BNN_model



# _______________________________________________
# PARAMETERS
# ---------------

seed_all = 3                 # seed for reproducibility 

data_name = "ddx_Z2g"       # name of the dataset
#data_name = "ddx_Z3g"       # name of the dataset

test_name = f"{data_name}_3x128_KinInv"

# number of total particles in the event
if data_name=="ddx_Z2g":
    n_part = 5 
elif data_name=="ddx_Z3g":
    n_part = 6 

path_info = "/home/lb_linux/BNN_Amplitudes_Example"               # path to dataset.txt
path_output = path_info                                          # path to output directory



# _______________________________________________
# FUNCTIONS
# ---------

# scalar product of two 4-momenta
def prod_4mom(p1, p2):
    res = (p1[:,0]*p2[:,0] - p1[:,1]*p2[:,1] - p1[:,2]*p2[:,2] - p1[:,3]*p2[:,3])
    return res 



# _______________________________________________
# LOAD AND PREPROCESS DATASET
# --------------------------- 

# Load dataset 
data = torch.empty((8*10**5, n_part*4+1))
with open(f'{path_info}/dataset_{data_name}.txt', 'r') as infof:
    event = 0
    # in each line of info: [E, px, py, pz] for each particle
    # particles are ordered: d, dbar, Z, g, g, g
    for line in infof.readlines():
        data[event, :] = torch.tensor([float(i) for i in line.split()])
        event +=1
n_events = data.shape[0]

# add kinematic invariants to data 
n_kin_inv = int(n_part * (n_part-1) / 2)
kin_inv = torch.empty((n_events, n_kin_inv))
count = 0
for i in range(n_part):
    for j in range(i+1, n_part):
        kin_inv[:, count] = torch.log(prod_4mom(data[:, i*4:(i+1)*4], data[:, j*4:(j+1)*4]))
        count+=1
data = torch.cat((data[:, :-1], kin_inv, data[:, -1].unsqueeze(1)), axis=1)

# remove x, y axis for initial particles (initial particles along z axis)
data = np.delete(data, [1, 2, 5, 6], axis=1)

# Split dataset 
train, test = train_test_split(data, test_size=10**5, random_state=seed_all)
train, val = train_test_split(train, test_size=2*10**5, random_state=seed_all)

# input preprocessing (standardization)
x_mean, x_std = torch.mean(train[:, :-1], axis=0), torch.std(train[:, :-1], axis=0) 
train[:, :-1] = (train[:, :-1] - x_mean) / x_std
val[:, :-1] = (val[:, :-1] - x_mean) / x_std
# output preprocessing (logarithm and standardization)
train[:, -1] = torch.log(train[:, -1])
val[:, -1] = torch.log(val[:, -1])
y_mean, y_std = torch.mean(train[:, -1]), torch.std(train[:, -1])
train[:, -1] = (train[:, -1] - y_mean) / y_std
val[:, -1] = (val[:, -1] - y_mean) / y_std 

batch_size = 256
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)



# _______________________________________________
# MODEL INITIALIZATION 
# --------------------

# model parameters
net_training_size = train.shape[0]
net_input_dim = train.shape[1] - 1     # number of input features, -1 because last column is the output 
net_inner_layers = [128, 128, 128] 
net_activation_inner = "tanh"

model = BNN_model.BNN_model(net_training_size=net_training_size, net_inner_layers=net_inner_layers, net_input_dim=net_input_dim, net_activation_inner=net_activation_inner)

print(model)

nn_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of model parameters: {}".format(nn_params))



# _______________________________________________
# TRAINING 
# --------

def loss_mse(outputs, targets):
    recon_loss = torch.mean(torch.square(targets-outputs))
    return recon_loss


# create directory for output
dir_name = BNN_model.create_directory(f"{path_output}/Results_{test_name}") 


n_epochs = 1000 

lr = 10**(-4) 
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

patience_earlystop = 100
early_stopper = BNN_model.EarlyStopper(patience=patience_earlystop, min_delta=10**(-6))


print(f"Start training with {n_epochs} epochs")
model.train()

loss_train_dict = defaultdict(list)
loss_val_dict = defaultdict(list)
for t in range(n_epochs):
    print(f"--------------------------------\nEpoch {t+1}")
    # gradient updates
    loss_tot_b, kl_b, neg_log_b, mse_b, pen_b = 0, 0, 0, 0, 0
    n_batches = 0 # there has to be a more pythonic way
    for batch in train_dataloader:
        # Compute prediction and loss
        x_b = batch[:, 0:-1] 
        y_b = batch[:, -1]
        pred_b = model(x_b)
        neg_log = model.neg_log_gauss(pred_b, y_b) 
        kl = model.KL()
        loss = neg_log + kl 
        mse = loss_mse(pred_b[:, 0], y_b)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save losses for later plotting
        loss_tot_b += loss.item()
        kl_b += kl.item()
        neg_log_b += neg_log.item()
        mse_b += mse.item()
        n_batches += 1

    loss_tot_b /= n_batches
    kl_b /= n_batches
    neg_log_b /= n_batches
    mse_b /= n_batches
    print(f"Train loss: {loss_tot_b:.5f} KL: {kl_b:.5f} Neg-log {neg_log_b:.5f} MSE: {mse_b:.5f}")
    loss_train_dict['kl'].append(kl_b)
    loss_train_dict['loss_tot'].append(loss_tot_b)
    loss_train_dict['neg_log'].append(neg_log_b)
    loss_train_dict['mse'].append(mse_b)


    # validation step 
    with torch.no_grad():
        pred_b = model(x_b)
        neg_log = model.neg_log_gauss(pred_b, y_b) 
        kl = model.KL()
        loss = neg_log + kl 
        mse = loss_mse(pred_b[:, 0], y_b)

    print(f"Validation loss: {loss:.5f} KL: {kl:.5f} Neg-log {neg_log:.5f} MSE: {mse:.5f}")
    loss_val_dict['kl'].append(kl) 
    loss_val_dict['loss_tot'].append(loss) 
    loss_val_dict['neg_log'].append(neg_log) 
    loss_val_dict['mse'].append(mse) 


    # early stopping
    if early_stopper.early_stop(loss_val_dict['mse'][t]):
        break

    # save best model weights
    if loss_val_dict['loss_tot'][t]<=loss_val_dict['loss_tot'][model.best_epoc_tot]:
        best_weights_tot = model.state_dict()
        model.best_epoc_tot = t    
    if loss_val_dict['mse'][t]<=loss_val_dict['mse'][model.best_epoc_mse]:
        best_weights_mse = model.state_dict()
        model.best_epoc_mse = t    

# load best weights 
if model.best_epoc_mse!=t:
    model.load_state_dict(best_weights_mse)
checkpoint_path = f"{dir_name}/state_best_mse"
torch.save({'epoch': t+1, 'model_state_dict': best_weights_mse}, checkpoint_path)
    


# _______________________________________________
# PLOT LOSSES
# -----------

with PdfPages(f"{dir_name}/Plot_Losses_BNN.pdf") as pdf:
    for key in loss_train_dict.keys():

        loss_train_list = np.array(loss_train_dict[key])
        loss_val_list = np.array(loss_val_dict[key])

        fig = plt.figure(figsize=[6, 5.5])
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(1, len(loss_train_list)+1, 1), loss_train_list, label=key, alpha=0.8)
        ax.plot(np.arange(1, len(loss_val_list)+1, 1), loss_val_list, label=key + " val", alpha=0.8)
        ax.axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax.axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss: " + key)
        ax.legend(frameon=False)

        pdf.savefig(fig) 
        plt.close(fig)



# _______________________________________________
# RESULTS ON TEST SET
# -------------------

# preprocess
x_test_norm = (test[:, :-1] - x_mean) / x_std
y_test_norm = torch.log(test[:, -1])
y_test_norm = (y_test_norm - y_mean) / y_std

n_draws = 50                 # number of draws for the evaluation of the network
amp_pred_norm_list = torch.zeros((n_draws, len(test)))
sig_pred_norm_list = torch.zeros((n_draws, len(test)))

model.eval()
with torch.no_grad():
    for i_draw in range(n_draws):
        pred_test = model(x_test_norm)
        amp_pred_norm_list[i_draw] = pred_test[:, 0]
        sig_pred_norm_list[i_draw] = torch.exp(pred_test[:, 1]/2)
        # reset the random weights of Bayesian layers 
        model.reset_random() 

# predictions 
pred_amp_norm = torch.mean(amp_pred_norm_list, axis=0)                  # prediction at normalized scale 
amp_pred_ori_list = torch.exp(amp_pred_norm_list * y_std + y_mean)      # predictions at original scale (amplitude level)
pred_ori_test = torch.mean(amp_pred_ori_list, axis=0)                            # final prediction at original scale (amplitude level)

# sigmas 
sig_pred_ori_list = sig_pred_norm_list * y_std * amp_pred_ori_list             # sigmas at original scale (amplitude level)
sigma_model_test = torch.sqrt(torch.mean(torch.square(sig_pred_ori_list), axis=0))
sigma_stoch_test = torch.std(amp_pred_ori_list, axis=0) 
sigma_tot_test = torch.sqrt(sigma_model_test**2 + sigma_stoch_test**2)

delta_norm_test = (y_test_norm - pred_amp_norm) / y_test_norm
delta_ori_test = (test[:, -1] - pred_ori_test) / test[:, -1]

# pass to numpy
y_test_ori = test[:, -1].numpy()
y_test_norm = y_test_norm.numpy()
pred_amp_norm = pred_amp_norm.numpy()
pred_ori_test = pred_ori_test.numpy()
delta_norm_test = delta_norm_test.numpy()
delta_ori_test = delta_ori_test.numpy()
sigma_model_test = sigma_model_test.numpy()
sigma_stoch_test = sigma_stoch_test.numpy()
sigma_tot_test = sigma_tot_test.numpy()



# _______________________________________________
# PLOT RESULTS 
# ------------

# bins for histograms
bins_amp_norm = np.linspace(-3, 3, 50)
if data_name=="ddx_Z2g":
    bins_amp_ori = np.logspace(np.log10(1e-5), np.log10(1e-0), 50)
elif data_name=="ddx_Z3g":
    bins_amp_ori = np.logspace(np.log10(1e-6), np.log10(1e-1), 50)
bins_delta_abs = np.logspace(np.log10(1e-5), np.log10(1e1), 50)
bins_sigma = np.logspace(np.log10(1e-7), np.log10(1e-3), 50)

n_test = len(y_test_ori)

# function to normalize weights for plotting
def weight_plt(x):
    return np.ones(x) / float(x)


with PdfPages(f"{dir_name}/Plot_Test_BNN.pdf") as pdf:

    # Plot total amplitude and accuracy for Normalized Amplitudes 
    fig, ax = plt.subplots(2, figsize=(8.27, 11.69)) 

    ax[0].hist([np.clip(y_test_norm, bins_amp_norm[0], bins_amp_norm[-1]), np.clip(pred_amp_norm, bins_amp_norm[0], bins_amp_norm[-1])],
            bins = bins_amp_norm, histtype='step', weights=[weight_plt(n_test), weight_plt(n_test)], label=['Truth', 'Prediction'])
    ax[0].set_xlabel("Normalized Amplitudes")
    ax[0].set_ylabel("Events")
    ax[0].set_yscale('log')
    ax[0].legend()

    perc_delta_1 = np.sum(np.abs(delta_norm_test)<0.01)*100/len(delta_norm_test)
    perc_delta_2 = np.sum(np.abs(delta_norm_test)<0.001)*100/len(delta_norm_test)
    ax[1].hist(np.clip(np.abs(delta_norm_test), bins_delta_abs[0], bins_delta_abs[-1]), bins=bins_delta_abs, weights=weight_plt(n_test))
    ax[1].set_xlabel("Accuracy of Normalized Amplitudes")
    ax[1].set_ylabel("Percentage of events")
    ax[1].set_xscale('log')
    ax[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax[1].legend(title="$|\Delta|<10^{-2}: $"+f"{perc_delta_1:.1f}%"+"\n"+"$|\Delta|<10^{-3}: $"+f"{perc_delta_2:.1f}%", loc='upper right', frameon=False)

    pdf.savefig(fig)
    plt.close(fig)


    # Plot total amplitude and accuracy for Original Amplitudes 
    fig, ax = plt.subplots(2, figsize=(8.27, 11.69)) 

    ax[0].hist([np.clip(y_test_ori, bins_amp_ori[0], bins_amp_ori[-1]), np.clip(pred_ori_test, bins_amp_ori[0], bins_amp_ori[-1])],
            bins = bins_amp_ori, histtype='step', weights=[weight_plt(n_test), weight_plt(n_test)], label=['Truth', 'Prediction'])
    ax[0].set_xlabel("Amplitudes")
    ax[0].set_ylabel("Events")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()

    perc_delta_1 = np.sum(np.abs(delta_ori_test)<0.01)*100/len(delta_ori_test)
    perc_delta_2 = np.sum(np.abs(delta_ori_test)<0.001)*100/len(delta_ori_test)
    ax[1].hist(np.clip(np.abs(delta_ori_test), bins_delta_abs[0], bins_delta_abs[-1]), bins=bins_delta_abs, weights=weight_plt(n_test))
    ax[1].set_xlabel("Accuracy of Amplitudes")
    ax[1].set_ylabel("Percentage of events")
    ax[1].set_xscale('log')
    ax[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax[1].legend(title="$|\Delta|<10^{-2}: $"+f"{perc_delta_1:.1f}%"+"\n"+"$|\Delta|<10^{-3}: $"+f"{perc_delta_2:.1f}%", loc='upper right', frameon=False)

    pdf.savefig(fig)
    plt.close(fig)


    # Sigma stat vs Sigma model
    fig, ax = plt.subplots(2, 2, figsize=(8.27, 11.69)) 
    fig.suptitle(r"$\sigma_{model} \; vs \; \sigma_{stoch}$")

    h2_0 = ax[0, 0].hist2d(np.clip(sigma_model_test, bins_sigma[0], bins_sigma[-1]), 
                        np.clip(sigma_stoch_test, bins_sigma[0], bins_sigma[-1]), 
                        bins=[bins_sigma, bins_sigma], norm=mpl.colors.LogNorm())
    ax[0, 0].set_xlabel(r"$\sigma_{model}$")
    ax[0, 0].set_ylabel(r"$\sigma_{stoch}$")
    ax[0, 0].set_xscale('log')
    ax[0, 0].set_yscale('log')
    plt.colorbar(h2_0[3], ax=ax[0, 0]) 

    h2_1 = ax[0, 1].hist2d(np.clip(y_test_ori, bins_amp_ori[0], bins_amp_ori[-1]), 
                        np.clip(sigma_tot_test, bins_sigma[0], bins_sigma[-1]), 
                        bins=[bins_amp_ori, bins_sigma], norm=mpl.colors.LogNorm())
    ax[0, 1].set_xlabel("Amplitude")
    ax[0, 1].set_ylabel(r"$\sigma_{tot}$")
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_yscale('log')
    plt.colorbar(h2_1[3], ax=ax[0, 1]) 

    h2_2 = ax[1, 0].hist2d(np.clip(y_test_ori, bins_amp_ori[0], bins_amp_ori[-1]), 
                        np.clip(sigma_stoch_test, bins_sigma[0], bins_sigma[-1]), 
                        bins=[bins_amp_ori, bins_sigma], norm=mpl.colors.LogNorm())
    ax[1, 0].set_xlabel("Amplitude")
    ax[1, 0].set_ylabel(r"$\sigma_{stoch}$")
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('log')
    plt.colorbar(h2_2[3], ax=ax[1, 0]) 

    h2_3 = ax[1, 1].hist2d(np.clip(y_test_ori, bins_amp_ori[0], bins_amp_ori[-1]), 
                        np.clip(sigma_model_test, bins_sigma[0], bins_sigma[-1]), 
                        bins=[bins_amp_ori, bins_sigma], norm=mpl.colors.LogNorm())
    ax[1, 1].set_xlabel("Amplitude")
    ax[1, 1].set_ylabel(r"$\sigma_{model}$")
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_yscale('log')
    plt.colorbar(h2_3[3], ax=ax[1, 1]) 

    pdf.savefig(fig)
    plt.close(fig)


print("Finished")
