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

path_info = "/home/lb_linux/BayesianNN_Amplitudes"               # path to dataset.txt
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

# remove x, y axis for initial particles
data = np.delete(data, [1, 2, 5, 6], axis=1)

# Split dataset 
train_400k, test = train_test_split(data, test_size=5*10**4, random_state=seed_all)
train_400k, val = train_test_split(train_400k, test_size=10**5, random_state=seed_all)

# input preprocessing (standardization)
x_mean, x_std = torch.mean(train_400k[:, :-1], axis=0), torch.std(train_400k[:, :-1], axis=0) 
train_400k[:, :-1] = (train_400k[:, :-1] - x_mean) / x_std
val[:, :-1] = (val[:, :-1] - x_mean) / x_std
# output preprocessing (logarithm and standardization)
train_400k[:, -1] = torch.log(train_400k[:, -1])
val[:, -1] = torch.log(val[:, -1])
y_mean, y_std = torch.mean(train_400k[:, -1]), torch.std(train_400k[:, -1])
train_400k[:, -1] = (train_400k[:, -1] - y_mean) / y_std
val[:, -1] = (val[:, -1] - y_mean) / y_std 

train_400k, train_50k = train_test_split(train_400k, test_size=5*10**4, random_state=seed_all)
train_400k, train_100k = train_test_split(train_400k, test_size=10**5, random_state=seed_all)
train_400k, train_200k = train_test_split(train_400k, test_size=2*10**5, random_state=seed_all)

batch_size = 256
train_50k_dataloader = DataLoader(train_50k, batch_size=batch_size, shuffle=True)
train_100k_dataloader = DataLoader(train_100k, batch_size=batch_size, shuffle=True)
train_200k_dataloader = DataLoader(train_200k, batch_size=batch_size, shuffle=True)
train_400k_dataloader = DataLoader(train_400k, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)



# _______________________________________________
# MODEL INITIALIZATION 
# --------------------

# model parameters
net_50k_training_size = train_50k.shape[0]
net_100k_training_size = train_100k.shape[0]
net_200k_training_size = train_200k.shape[0]
net_400k_training_size = train_400k.shape[0]

net_input_dim = train_50k.shape[1] - 1     # number of input features, -1 because last column is the output 
net_inner_layers = [128, 128, 128] 
net_activation_inner = "tanh"

model_50k = BNN_model.BNN_model(net_training_size=net_50k_training_size, net_inner_layers=net_inner_layers, net_input_dim=net_input_dim, net_activation_inner=net_activation_inner)
model_100k = BNN_model.BNN_model(net_training_size=net_100k_training_size, net_inner_layers=net_inner_layers, net_input_dim=net_input_dim, net_activation_inner=net_activation_inner)
model_200k = BNN_model.BNN_model(net_training_size=net_200k_training_size, net_inner_layers=net_inner_layers, net_input_dim=net_input_dim, net_activation_inner=net_activation_inner)
model_400k = BNN_model.BNN_model(net_training_size=net_400k_training_size, net_inner_layers=net_inner_layers, net_input_dim=net_input_dim, net_activation_inner=net_activation_inner)



# _______________________________________________
# TRAINING 
# --------

def loss_mse(outputs, targets):
    recon_loss = torch.mean(torch.square(targets-outputs))
    return recon_loss

# create directory for output
dir_name = BNN_model.create_directory(f"{path_output}/Results_{test_name}_Dataset") 

n_epochs = 1000 

lr = 10**(-4) 

patience_earlystop = 50

loss_train_50k_dict = defaultdict(list)
loss_val_50k_dict = defaultdict(list)
loss_train_100k_dict = defaultdict(list)
loss_val_100k_dict = defaultdict(list)
loss_train_200k_dict = defaultdict(list)
loss_val_200k_dict = defaultdict(list)
loss_train_400k_dict = defaultdict(list)
loss_val_400k_dict = defaultdict(list)


#for model, train_dataloader in zip([model_50k, model_100k, model_200k, model_400k], [train_50k_dataloader, train_100k_dataloader, train_200k_dataloader, train_400k_dataloader]):
for i_dataset in range(4):

    model = [model_50k, model_100k, model_200k, model_400k][i_dataset]
    train_dataloader = [train_50k_dataloader, train_100k_dataloader, train_200k_dataloader, train_400k_dataloader][i_dataset]
    loss_train_dict = [loss_train_50k_dict, loss_train_100k_dict, loss_train_200k_dict, loss_train_400k_dict][i_dataset]
    loss_val_dict = [loss_val_50k_dict, loss_val_100k_dict, loss_val_200k_dict, loss_val_400k_dict][i_dataset]

    print(f"Start training model {train_dataloader.dataset.shape[0]} with {n_epochs} epochs")

    early_stopper = BNN_model.EarlyStopper(patience=patience_earlystop, min_delta=10**(-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
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
    checkpoint_path = f"{dir_name}/state_best_mse_{[50, 100, 200, 400][i_dataset]}k"
    torch.save({'epoch': t+1, 'model_state_dict': best_weights_mse}, checkpoint_path)



# _______________________________________________
# PLOT LOSSES
# -----------

with PdfPages(f"{dir_name}/Plot_Losses_Test_Dataset.pdf") as pdf:
    for key in loss_train_50k_dict.keys():

        loss_train_50k_list = np.array(loss_train_50k_dict[key])
        loss_val_50k_list = np.array(loss_val_50k_dict[key])
        loss_train_100k_list = np.array(loss_train_100k_dict[key])
        loss_val_100k_list = np.array(loss_val_100k_dict[key])
        loss_train_200k_list = np.array(loss_train_200k_dict[key])
        loss_val_200k_list = np.array(loss_val_200k_dict[key])
        loss_train_400k_list = np.array(loss_train_400k_dict[key])
        loss_val_400k_list = np.array(loss_val_400k_dict[key])

        fig, ax = plt.subplots(2, 2, figsize=(8.27, 11.69)) 
        ax[0, 0].plot(np.arange(1, len(loss_train_50k_list)+1, 1), loss_train_50k_list, label=key, alpha=0.8)
        ax[0, 0].plot(np.arange(1, len(loss_val_50k_list)+1, 1), loss_val_50k_list, label=key + " val", alpha=0.8)
        ax[0, 0].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[0, 0].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[0, 0].set_xlabel("Epoch")
        ax[0, 0].set_ylabel("Loss: " + key + " 50k model")

        ax[0, 1].plot(np.arange(1, len(loss_train_100k_list)+1, 1), loss_train_100k_list, label=key, alpha=0.8)
        ax[0, 1].plot(np.arange(1, len(loss_val_100k_list)+1, 1), loss_val_100k_list, label=key + " val", alpha=0.8)
        ax[0, 1].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[0, 1].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[0, 1].set_xlabel("Epoch")
        ax[0, 1].set_ylabel("Loss: " + key + " 100k model")

        ax[1, 0].plot(np.arange(1, len(loss_train_200k_list)+1, 1), loss_train_200k_list, label=key, alpha=0.8)
        ax[1, 0].plot(np.arange(1, len(loss_val_200k_list)+1, 1), loss_val_200k_list, label=key + " val", alpha=0.8)
        ax[1, 0].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[1, 0].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[1, 0].set_xlabel("Epoch")
        ax[1, 0].set_ylabel("Loss: " + key + " 200k model")

        ax[1, 1].plot(np.arange(1, len(loss_train_400k_list)+1, 1), loss_train_400k_list, label=key, alpha=0.8)
        ax[1, 1].plot(np.arange(1, len(loss_val_400k_list)+1, 1), loss_val_400k_list, label=key + " val", alpha=0.8)
        ax[1, 1].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[1, 1].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
        ax[1, 1].set_xlabel("Epoch")
        ax[1, 1].set_ylabel("Loss: " + key + " 400k model")

        pdf.savefig(fig) 
        plt.close(fig)


        fig, ax = plt.subplots(2, 2, figsize=(8.27, 11.69)) 
        if (len(loss_train_50k_list)>100):
            ax[0, 0].plot(np.arange(100, len(loss_train_50k_list)+1, 1), loss_train_50k_list[100:], label=key, alpha=0.8)
            ax[0, 0].plot(np.arange(100, len(loss_val_50k_list)+1, 1), loss_val_50k_list[100:], label=key + " val", alpha=0.8)
            ax[0, 0].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
            ax[0, 0].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
            n_xticks = len(loss_train_50k_list)//100 
            xticks = np.linspace(100, (len(loss_train_50k_list)//100)*100, n_xticks) 
            ax[0, 0].set_xticks(ticks=xticks)
            ax[0, 0].set_xlabel("Epoch")
            ax[0, 0].set_ylabel("Loss: " + key + " 50k model")
        else:
            ax[0, 0].remove()

        if (len(loss_train_100k_list)>100):
            ax[0, 1].plot(np.arange(100, len(loss_train_100k_list)+1, 1), loss_train_100k_list[100:], label=key, alpha=0.8)
            ax[0, 1].plot(np.arange(100, len(loss_val_100k_list)+1, 1), loss_val_100k_list[100:], label=key + " val", alpha=0.8)
            ax[0, 1].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
            ax[0, 1].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
            n_xticks = len(loss_train_100k_list)//100 
            xticks = np.linspace(100, (len(loss_train_100k_list)//100)*100, n_xticks) 
            ax[0, 1].set_xticks(ticks=xticks)
            ax[0, 1].set_xlabel("Epoch")
            ax[0, 1].set_ylabel("Loss: " + key + " 100k model")
        else:
            ax[0, 1].remove()

        if (len(loss_train_200k_list)>100):
            ax[1, 0].plot(np.arange(100, len(loss_train_200k_list)+1, 1), loss_train_200k_list[100:], label=key, alpha=0.8)
            ax[1, 0].plot(np.arange(100, len(loss_val_200k_list)+1, 1), loss_val_200k_list[100:], label=key + " val", alpha=0.8)
            ax[1, 0].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
            ax[1, 0].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
            n_xticks = len(loss_train_200k_list)//100 
            xticks = np.linspace(100, (len(loss_train_200k_list)//100)*100, n_xticks) 
            ax[1, 0].set_xticks(ticks=xticks)
            ax[1, 0].set_xlabel("Epoch")
            ax[1, 0].set_ylabel("Loss: " + key + " 200k model")
        else:
            ax[1, 0].remove()

        if (len(loss_train_400k_list)>100):
            ax[1, 1].plot(np.arange(100, len(loss_train_400k_list)+1, 1), loss_train_400k_list[100:], label=key, alpha=0.8)
            ax[1, 1].plot(np.arange(100, len(loss_val_400k_list)+1, 1), loss_val_400k_list[100:], label=key + " val", alpha=0.8)
            ax[1, 1].axvline(x=model.best_epoc_tot+1, label="best epoch loss tot", color='olive', linewidth=1.5, linestyle='dotted', alpha=0.6)
            ax[1, 1].axvline(x=model.best_epoc_mse+1, label="best epoch mse", color='lime', linewidth=1.5, linestyle='dotted', alpha=0.6)
            n_xticks = len(loss_train_400k_list)//100 
            xticks = np.linspace(100, (len(loss_train_400k_list)//100)*100, n_xticks) 
            ax[1, 1].set_xticks(ticks=xticks)
            ax[1, 1].set_xlabel("Epoch")
            ax[1, 1].set_ylabel("Loss: " + key + " 400k model")
        else:
            ax[1, 1].remove()

        pdf.savefig(fig) 
        plt.close(fig)



# _______________________________________________
# RESULTS ON TEST SET
# -------------------

# preprocess
x_test_norm = (test[:, :-1] - x_mean) / x_std
y_test_norm = torch.log(test[:, -1])
y_test_norm = (y_test_norm - y_mean) / y_std

pred_ori_tests = np.zeros((4, test.shape[0]))
delta_ori_tests = np.zeros((4, test.shape[0]))
sigma_model_tests = np.zeros((4, test.shape[0]))
sigma_stoch_tests = np.zeros((4, test.shape[0]))
sigma_tot_tests = np.zeros((4, test.shape[0]))

n_draws = 50                 # number of draws for the evaluation of the network
for i_dataset in range(4):

    model = [model_50k, model_100k, model_200k, model_400k][i_dataset]


    amp_pred_norm_list = torch.zeros((n_draws, len(test)))
    sig_pred_norm_list = torch.zeros((n_draws, len(test)))

    model.eval()
    with torch.no_grad():
        for i_draw in range(n_draws):
            pred_i = model(x_test_norm)
            amp_pred_norm_list[i_draw] = pred_i[:, 0]
            sig_pred_norm_list[i_draw] = torch.exp(pred_i[:, 1]/2)
            # reset the random weights of Bayesian layers 
            model.reset_random() 

    # predictions 
    pred_amp_norm = torch.mean(amp_pred_norm_list, axis=0)                  # prediction at normalized scale 
    amp_pred_ori_list = torch.exp(amp_pred_norm_list * y_std + y_mean)      # predictions at original scale (amplitude level)
    pred_ori_i = torch.mean(amp_pred_ori_list, axis=0)                            # final prediction at original scale (amplitude level)

    # sigmas 
    sig_pred_ori_list = sig_pred_norm_list * y_std * amp_pred_ori_list             # sigmas at original scale (amplitude level)
    sigma_model_i = torch.sqrt(torch.mean(torch.square(sig_pred_ori_list), axis=0))
    sigma_stoch_i = torch.std(amp_pred_ori_list, axis=0) 
    sigma_tot_i = torch.sqrt(sigma_model_i**2 + sigma_stoch_i**2)

    delta_ori_i = (test[:, -1] - pred_ori_i) / test[:, -1]

    pred_ori_tests[i_dataset] = pred_ori_i.numpy()
    delta_ori_tests[i_dataset] = delta_ori_i.numpy()
    sigma_model_tests[i_dataset] = sigma_model_i.numpy()
    sigma_stoch_tests[i_dataset] = sigma_stoch_i.numpy()
    sigma_tot_tests[i_dataset] = sigma_tot_i.numpy()

# pass to numpy
y_test_ori = test[:, -1].numpy()
y_test_norm = y_test_norm.numpy()



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

colors = col_comb = ['maroon', 'orange', 'olive', 'teal'] 


n_test = len(y_test_ori)

# function to normalize weights for plotting
def weight_plt(x):
    return np.ones(x) / float(x)


with PdfPages(f"{dir_name}/Plot_Test_{test_name}_Dataset.pdf") as pdf:

    # Plot total amplitude and accuracy for Original Amplitudes 
    fig, ax = plt.subplots(2, figsize=(8.27, 11.69)) 

    ax[0].hist(np.clip(y_test_ori, bins_amp_ori[0], bins_amp_ori[-1]), bins = bins_amp_ori, histtype='step', weights=weight_plt(n_test), label='Truth', color='black', lw=1.5, alpha=0.7)
    for i_dataset in range(4):
        ax[0].hist(np.clip(pred_ori_tests[i_dataset], bins_amp_ori[0], bins_amp_ori[-1]), bins = bins_amp_ori, histtype='step', weights=weight_plt(n_test), 
                   label=f"{[50, 100, 200, 400][i_dataset]}k model", color=colors[i_dataset], lw=1.5, alpha=0.7)
    ax[0].set_xlabel("Amplitudes")
    ax[0].set_ylabel("Events")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()

    for i_dataset in range(4):
        ax[1].hist(np.clip(np.abs(delta_ori_tests[i_dataset]), bins_delta_abs[0], bins_delta_abs[-1]), bins=bins_delta_abs, histtype='step', weights=weight_plt(n_test), 
                   label=f"{[50, 100, 200, 400][i_dataset]}k model", color=colors[i_dataset], lw=1.5, alpha=0.7)
    ax[1].set_xlabel("Accuracy of Amplitudes")
    ax[1].set_ylabel("Percentage of events")
    ax[1].set_xscale('log')
    ax[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    ax[1].legend()

    pdf.savefig(fig)
    plt.close(fig)


    # table accuracy
    fig = plt.figure(figsize=[8.27, 5.5])
    ax = fig.add_subplot(1, 1, 1)
    table_data = [["Dataset", "$|\Delta|<10^{-2}$", "$|\Delta|<10^{-2}$", "$<\sigma_{model}>$", "$<\sigma_{stoch}>$"], 
                  ["50k", f"{np.sum(np.abs(delta_ori_tests[0])<0.01)*100/len(delta_ori_tests[0]):.2f}%", f"{np.sum(np.abs(delta_ori_tests[0])<0.001)*100/len(delta_ori_tests[0]):.2f}%", 
                   f"{np.mean(sigma_model_tests[0]):.2e}", f"{np.mean(sigma_stoch_tests[0]):.2e}"],
                  ["100k", f"{np.sum(np.abs(delta_ori_tests[1])<0.01)*100/len(delta_ori_tests[1]):.2f}%", f"{np.sum(np.abs(delta_ori_tests[1])<0.001)*100/len(delta_ori_tests[1]):.2f}%", 
                   f"{np.mean(sigma_model_tests[1]):.2e}", f"{np.mean(sigma_stoch_tests[1]):.2e}"],
                  ["200k", f"{np.sum(np.abs(delta_ori_tests[2])<0.01)*100/len(delta_ori_tests[2]):.2f}%", f"{np.sum(np.abs(delta_ori_tests[2])<0.001)*100/len(delta_ori_tests[2]):.2f}%", 
                   f"{np.mean(sigma_model_tests[2]):.2e}", f"{np.mean(sigma_stoch_tests[2]):.2e}"],
                  ["400k", f"{np.sum(np.abs(delta_ori_tests[3])<0.01)*100/len(delta_ori_tests[3]):.2f}%", f"{np.sum(np.abs(delta_ori_tests[3])<0.001)*100/len(delta_ori_tests[3]):.2f}%", 
                   f"{np.mean(sigma_model_tests[3]):.2e}", f"{np.mean(sigma_stoch_tests[3]):.2e}"]]

    # Create the table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1, 0.1, 0.1, 0.1, 0.1])
    table.scale(2, 2)
    # Hide the axes for the table
    ax.axis('off')

    pdf.savefig(fig)
    plt.close(fig)

