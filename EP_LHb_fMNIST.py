
# Fashion MNIST MLP
import torchvision
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import random
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import pandas as pd
import pickle
from datetime import date


class Dales_Law_Combine_LHb_Model(nn.Module):
    """
    Model follows Dale's Law by initializing separate excitatory and inhibitory populations
    
    Combined LHb. Separate EP E and I
    
    EP E and I are initialized purely positive and negative respectively
    
    LHb initialized random
    
    DAN initialized negative, only takes negative input
    """

    def __init__(self, in_features=784, h1=512, h2=512, out_features=10, dropout_rate=0.5):
        super().__init__()

        # 50% E and I
        num_excitatory_h1 = int(0.5 * h1)
        num_inhibitory_h1 = h1 - num_excitatory_h1
        
        num_excitatory_h2 = int(0.5 * h2)
        num_inhibitory_h2 = h2 - num_excitatory_h2

        # Create layers
        self.EP_E = nn.Linear(in_features, h1)
        self.EP_I = nn.Linear(in_features, h1)
        self.bn1_E = nn.BatchNorm1d(h1)
        self.bn1_I = nn.BatchNorm1d(h1)
        self.LHb = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)

        self.dropout = nn.Dropout(dropout_rate)

        self.DAN = nn.Linear(h2, out_features)
        
        # initialize EP_E, EP_I, DAN as strictly E or I
        self.apply(self.absolute_val)
        
        # keep track of weights
        self.init_weights = self.record_params(calc_sign=False)
            

    def absolute_val(self, m):
        """
        initializes layers as pure positive or pure negative
        Init Positive: EP excitatory
        Init Negative: EP inhibitory and DAN
        
        """
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if m is self.EP_E:
                        # excitatory = all positive
                        m.weight.data = torch.abs(m.weight.data)
                        
                elif m is self.EP_I or m is self.DAN:
                        # inhibitory = all negative
                        m.weight.data = -torch.abs(m.weight.data)
                        
            
    def forward(self, x):
        x = x.view(x.size(0), -1)

        # EP
        x_e = F.relu(self.bn1_E(self.EP_E(x)))
        x_i = F.relu(self.bn1_I(self.EP_I(x)))
        
        # Converge into LHb
        x = x_e + x_i
        x = F.relu(self.bn2(self.LHb(x)))
        
        # Pure Negative LHB to DAN
        x = -torch.abs(x)
        x = self.dropout(x)
        x = self.DAN(x)
        
        return x

    
    def record_params(self, calc_sign: bool=True):
    # Save the network weights
        recorded_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    cur_data = param.data.detach().cpu().clone()
                    recorded_params[name] = (cur_data)

            if calc_sign:
                print(name)
                frac_pos = 100*(torch.sum(cur_data > 0)/cur_data.numel()).numpy()
                frac_zero = 100*(torch.sum(cur_data == 0)/cur_data.numel()).numpy()
                frac_neg = 100*(torch.sum(cur_data < 0)/cur_data.numel()).numpy()
                print(' Positive: ' + str(frac_pos) + '%; Negative: ' + str(frac_neg) + '%; Zero: ' + str(frac_zero) + '%')

        return recorded_params


class Dales_Law_Model(nn.Module):
    """
    Model follows Dale's Law by initializing separate excitatory and inhibitory populations
    
    EP E and I are initialized purely positive and negative respectively
    
    LHb E and I are initialized purely positive and negative respectively
    
    DAN initialized negative, only takes negative input
    """

    def __init__(self, in_features=784, h1=512, h2=512, out_features=10, dropout_rate=0.5):
        super().__init__()

        # 50% E and I
        num_excitatory_h1 = int(0.5 * h1)
        num_inhibitory_h1 = h1 - num_excitatory_h1

        # Create layers
        self.EP_E = nn.Linear(in_features, num_excitatory_h1)
        self.EP_I = nn.Linear(in_features, num_inhibitory_h1)
        self.bn1_E = nn.BatchNorm1d(num_excitatory_h1)
        self.bn1_I = nn.BatchNorm1d(num_inhibitory_h1)
        self.LHb_E = nn.Linear(num_excitatory_h1, h2)
        self.LHb_I = nn.Linear(num_inhibitory_h1, h2)  
        self.bn2_E = nn.BatchNorm1d(h2)
        self.bn2_I = nn.BatchNorm1d(h2)
        self.dropout = nn.Dropout(dropout_rate)
        self.DAN = nn.Linear(h2, out_features)
        
        # initialize EP_E, EP_I, LHb_E, LHb_I, DAN as E or I
        self.apply(self.absolute_val)
        
        # keep track of weights
        self.init_weights = self.record_params(calc_sign=False)
            

    def absolute_val(self, m):
        """
        initializes layers as pure positive or pure negative
        
        """
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                if m is self.EP_E or m is self.LHb_E:
                        # excitatory = all positive
                        m.weight.data = torch.abs(m.weight.data)
                        
                elif m is self.EP_I or m is self.LHb_I or m is self.DAN:
                        # inhibitory = all negative
                        m.weight.data = -torch.abs(m.weight.data)
                        
            
    def forward(self, x):
        x = x.view(x.size(0), -1)

        # EP
        x_e = F.relu(self.bn1_E(self.EP_E(x)))
        x_i = F.relu(self.bn1_I(self.EP_I(x)))
        
        # LHb
        x_e = F.relu(self.bn2_E(self.LHb_E(x_e)))
        x_i = F.relu(self.bn2_I(self.LHb_I(x_i)))
        
        # converge to DAN
        x = x_e + x_i
        
        # Pure Negative LHB to DAN
        x = -torch.abs(x)
        x = self.dropout(x)

        x = self.DAN(x)

        return x

    
    def record_params(self, calc_sign: bool=True):
    # Save the network weights
        recorded_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    cur_data = param.data.detach().cpu().clone()
                    recorded_params[name] = (cur_data)

            if calc_sign:
                print(name)
                frac_pos = 100*(torch.sum(cur_data > 0)/cur_data.numel()).numpy()
                frac_zero = 100*(torch.sum(cur_data == 0)/cur_data.numel()).numpy()
                frac_neg = 100*(torch.sum(cur_data < 0)/cur_data.numel()).numpy()
                print(' Positive: ' + str(frac_pos) + '%; Negative: ' + str(frac_neg) + '%; Zero: ' + str(frac_zero) + '%')

        return recorded_params
    
class Corelease_Model(nn.Module):
    """
    Model follows co-release: regular MLP
    
    if real: make DAN weights pure E or I - shouldn't be any I

    if combined Dale's Law E + I: Make all weights pure E or I
    
    """
    def __init__(self, in_features=784, h1=512, h2=512, out_features=10, dropout_rate=0.5, real = False, combine_EI = False, dales_law = False):
        super().__init__()
        self.real = real
        self.dales_law = dales_law
        # create layers
        self.EP = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.LHb = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.dropout = nn.Dropout(dropout_rate)
        self.DAN = nn.Linear(h2, out_features)
        
        # initialize DAN as purely inhibitory
        if self.real == True:
            self.apply(self.absolute_val)
            
        # combined EI/ dale's law
        EP_LHb_DAN_pos_neurons, EP_LHb_DAN_neg_neurons = {}, {}
        DAN_pos_neurons, DAN_neg_neurons = {}, {}
        
        

        # neurons will only project pure excitatory/inhibitory 
        with torch.no_grad():
            for name, param in self.named_parameters():
                if combine_EI == True:
                    print(combine_EI)
                    if "weight" in name:
                        # categorize neurons as excitatory/inhibitory
                        EP_LHb_DAN_pos_neurons[name] = torch.sum(param.data, axis = 0) >= 0
                        EP_LHb_DAN_neg_neurons[name] = torch.sum(param.data, axis = 0) < 0

                        # make neuron all excitatory/inhibitory
                        param.data[:, EP_LHb_DAN_pos_neurons[name]] = torch.sign(param[:, EP_LHb_DAN_pos_neurons[name]]) * param[:, EP_LHb_DAN_pos_neurons[name]]
                        param.data[:, EP_LHb_DAN_neg_neurons[name]] = -torch.sign(param[:, EP_LHb_DAN_neg_neurons[name]]) * param[:, EP_LHb_DAN_neg_neurons[name]]
                elif self.real == True:
                    if "DAN.weight" in name:
                        DAN_pos_neurons[name] = torch.sum(param.data, axis = 0) >= 0
                        DAN_neg_neurons[name] = -torch.sum(param.data, axis = 0) < 0
                        
                        # make neuron all excitatory/inhibitory
                        param.data[:, DAN_pos_neurons[name]] = torch.sign(param[:, DAN_pos_neurons[name]]) * param[:, DAN_pos_neurons[name]]
                        param.data[:, DAN_neg_neurons[name]] = -torch.sign(param[:, DAN_neg_neurons[name]]) * param[:, DAN_neg_neurons[name]]


        # keep track of weights
        self.EP_LHb_DAN_pos_neurons = EP_LHb_DAN_pos_neurons
        self.EP_LHb_DAN_neg_neurons = EP_LHb_DAN_neg_neurons
        
        self.DAN_pos_neurons = DAN_pos_neurons
        self.DAN_neg_neurons = DAN_neg_neurons
            
        
        
        
        
        # keep track of weights
        self.init_weights = self.record_params(calc_sign=False)
        
    def absolute_val(self, m):
        """
        initializes DAN as pure positive negative
        
        """
        if isinstance(m, nn.Linear):
            with torch.no_grad():          
                if m is self.DAN:
                        # Initialize inhibitory weights and DAN weights with negative values
                        m.weight.data = -torch.abs(m.weight.data)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        print("ctx-EP", x)
        x = F.relu(self.bn1(self.EP(x)))
        print("EP-LHb", x)
        x = F.relu(self.bn2(self.LHb(x)))
        
        # pure negative -> DAN
        if self.real == True:
            x = -torch.abs(x)
        x = self.dropout(x)
        x = self.DAN(x)

        return x
    
    def record_params(self, calc_sign: bool=True):
    # Save the network weights
        recorded_params = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                with torch.no_grad():
                    cur_data = param.data.detach().cpu().clone()
                    recorded_params[name] = (cur_data)

            if calc_sign:
                print(name)
                frac_pos = 100*(torch.sum(cur_data > 0)/cur_data.numel()).numpy()
                frac_zero = 100*(torch.sum(cur_data == 0)/cur_data.numel()).numpy()
                frac_neg = 100*(torch.sum(cur_data < 0)/cur_data.numel()).numpy()
                print(' Positive: ' + str(frac_pos) + '%; Negative: ' + str(frac_neg) + '%; Zero: ' + str(frac_zero) + '%')

        return recorded_params
    

class adam(torch.optim.Optimizer):
    """
    optimizer:
    if fixed sign: prevent weights from switching sign
    
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, fixed_sign: bool = False, real: bool = False): 
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fixed_sign=fixed_sign, real=real)
        super(adam, self).__init__(params, defaults) 

    def step(self, init_weights=None):
                
        for group in self.param_groups:
            for i, p in enumerate(group['params']): 
                if p.grad is None: continue
                grad = p.grad.data 
                if grad.is_sparse: raise RuntimeError("Adam does not support sparse gradients") 

                state = self.state[p]

                # State initialization 
                if len(state) == 0:
                    state["step"] = 0
                    # Momentum: Exponential moving average of gradient values 
                    state["exp_avg"] = torch.zeros_like(p.data) 
                    # RMS prop componenet: Exponential moving average of squared gradient values 
                    state["exp_avg_sq"] = torch.zeros_like(p.data) 

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"] 
                beta1, beta2 = group["betas"] 
                state["step"] += 1

                if group['weight_decay'] != 0: 
                    grad = grad.add(p.data, alpha=group['weight_decay']) 

                # Decay the first and second moment running average coefficient
                exp_avg.lerp_(grad, 1 - beta1) # momentum
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1-beta2) # rms

                bias_correction1 = 1 - beta1 ** state["step"] 
                bias_correction2 = 1 - beta2 ** state["step"] 

                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)  
                
                # clamp fixed_sign
                if group["fixed_sign"]:
                    flip_mask = init_weights[i].sign()*p.data.sign()<0
                    p.data[flip_mask] = 0
                
                if group["real"] and group["fixed_sign"] == False:
                    for name, param in model.named_parameters():
                        if name == "DAN.weight":
                            flip_mask = param.data.sign()>0
                            param.data[flip_mask] = 0


# Train model
def train_model(train_loader, test_loader, val_loader, epochs, params_ls):
    """
    train then test model after each epoch
    
    
    """

    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step(init_weights=list(model.init_weights.values()))
            params = model.record_params(calc_sign = False)
            if i % 20 == 0:
                params_ls.append(params.copy())
            

            # Calculate Accuracy
            if i % 10 == 0:
                if val_loader is not None:
                    model.eval()
                    correct, total = 0, 0
                    # Iterate through test dataset
                    for val_data, val_labels in val_loader:
                        val_data, val_labels = val_data.to(device), val_labels.to(device)
                        
                        val_outputs = model(val_data)
                        _, predicted = torch.max(val_outputs.data, 1)
                        total += val_labels.size(0)
                        correct += (predicted == val_labels).sum()


                    accuracy = 100 * correct / total
                    val_accuracy.append(accuracy.cpu())
                    print('Epoch [%d/%d], Iteration: %d, Loss: %.4f, Val Accuracy: %.4f' %(epoch+1, epochs, i, loss.data, accuracy))
                else:
                    print('Epoch [%d/%d], Iteration: %d, Loss: %.4f'  %(epoch+1, epochs, i, loss.data))
        scheduler.step()

    # Test model
    model.eval()
    
    # Iterate through test dataset
    correct, total = 0, 0
    for i, (test_data, test_labels) in enumerate(test_loader):
        
        test_data, test_labels = test_data.to(device), test_labels.to(device)
        
        test_outputs = model(test_data)
        _, predicted = torch.max(test_outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum()


        accuracy = 100 * correct / total
        test_accuracy.append(accuracy.cpu())
    #print('Epoch [%d/%d], Iteration: %d, Loss: %.4f, Val Accuracy: %.4f' %(epoch+1, epochs, i, loss.data, accuracy))

    """
    
    
    with torch.no_grad():
        for data, target in test_loader:
            correct = 0
            total = 0
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            # record correct/accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            accuracy = correct/total
            
            test_accuracy.append(accuracy)

            
    print(f'Test Accuracy: {correct / len(test_loader.dataset) * 100:.2f}%')
    """       

# FashionMNIST dataset
# 60,000 training samples
# 10,000 test samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

train_data = datasets.FashionMNIST(root = './data', train = True,
                        transform = transforms.ToTensor(), download = True)

test_data = datasets.FashionMNIST(root = './data', train = False,
                       transform = transforms.ToTensor())

transform = transforms.Compose([transforms.ToTensor()])


# create new shuffle dataset with FashionMNIST data
shuffle_train_data = datasets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), 
                                           download=True)
shuffle_test_data = datasets.FashionMNIST(root='./data', train=False,  
                                          transform = transforms.ToTensor())


# shuffle targets
new_targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
random.shuffle(new_targets)



# reassign old targets to new targets
for i in range(len(shuffle_train_data.targets)):
    old_target = shuffle_train_data.targets[i].item()
    shuffle_train_data.targets[i] = new_targets[old_target]

for i in range(len(shuffle_test_data.targets)):
    old_target = shuffle_test_data.targets[i].item()
    shuffle_test_data.targets[i] = new_targets[old_target]

# Split both original and shuffled training data into 50,000 training and 10,000 validation samples
train_size = 50000
val_size = 10000

# Split for the unshuffled dataset
train_data, val_data = random_split(train_data, [train_size, val_size], 
                                                          generator=torch.Generator().manual_seed(42))  # For reproducibility

# Split for the shuffled dataset
shuffle_train_data, shuffle_val_data = random_split(shuffle_train_data, [train_size, val_size], 
                                                      generator=torch.Generator().manual_seed(42))

# Set batch size
batch_size = 256

# DataLoader for the unshuffled dataset
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))

# DataLoader for the shuffled dataset
shuffle_train_loader = DataLoader(dataset=shuffle_train_data, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device))
shuffle_val_loader = DataLoader(dataset=shuffle_val_data, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))
shuffle_test_loader = DataLoader(dataset=shuffle_test_data, batch_size=batch_size, shuffle=False, generator=torch.Generator(device=device))


# network dictionary
network_dict = {"network_type": ["MLP", "MLP", "MLP", "MLP", "MLP"], "EP_LHb": ["random", "random", "dales-law", "dales-law", "dales-law"],
                      "LHb_DAN": ["real", "mixed", "real", "real", "real"], "update_methods": ["corelease", "corelease", "fixed-sign", "fixed-sign", "fixed-sign"]
               , "split_EI": ["no_split", "no_split", "split_EP_only", "split_EP_LHb", "combine_EI"]}

training_loss_summary, test_accuracy_summary, val_accuracy_summary, initial_params_summary, trained_params_summary, params_summary = {}, {}, {}, {}, {}, {}

num_networks = 1

# initialize loss/accuracy variables
epochs = 1
T_max = 1
lr = 0.01

# layer neuron num
in_features = 784
#h1_ls = [512, 100, 30, 15, 5]
#h2_ls = [512, 100, 30, 15, 5]
h1_ls = [512]
h2_ls = [512]


out_features = 10

for i in range(1, 2):
    for h1_count in range(len(h1_ls)):
        for h2_count in range(len(h2_ls)):
            for num in range(1,num_networks+1):
                # initialize results storage
                training_loss = []
                val_accuracy = []
                test_accuracy = []
                params_ls = []

                # network conditions
                network_type = network_dict["network_type"][i]

                # dales-law
                EP_LHb = network_dict["EP_LHb"][i]
                if EP_LHb == "dales-law":
                    dales_law = True
                else:
                    dales_law = False

                # Real LHb to DAN
                LHb_DAN = network_dict["LHb_DAN"][i]
                if LHb_DAN == "real":
                    real = True
                else:
                    real = False

                # Update method
                update_methods = network_dict["update_methods"][i]
                if update_methods == "fixed-sign":
                    fixed_sign = True
                else:
                    fixed_sign = False

                # split EI
                split_EI = network_dict["split_EI"][i]
                if split_EI == "combine_EI":
                    combine_EI = True
                elif split_EI == "split_EP_only":
                    split_EP_only = True
                else:
                    combine_EI = False


                # Instantiate model
                torch.manual_seed(1000)
                if split_EI == "split_EP_LHb":
                    model = Dales_Law_Model(in_features=in_features, h1=h1_ls[h1_count], h2=h2_ls[h2_count], out_features=out_features).to(device)
                elif split_EI == "split_EP_only":
                    model = Dales_Law_Combine_LHb_Model(in_features=in_features, h1=h1_ls[h1_count], h2=h2_ls[h2_count], out_features=out_features).to(device)
                else:
                    model = Corelease_Model(in_features=in_features, h1=h1_ls[h1_count], h2=h2_ls[h2_count], out_features=out_features, real = real, combine_EI = combine_EI).to(device)

                # network name
                network_name = network_type+'_'+EP_LHb+'_'+LHb_DAN+'_'+update_methods+'_' + split_EI + '_' + str(num) + ' ('+ str(h1_ls[h1_count])+' EP neurons, ' + str(h2_ls[h2_count])+' LHb neurons)'

                # Criterion to measure error
                criterion = nn.CrossEntropyLoss()

                # adam optimizer
                optimizer = adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, fixed_sign = fixed_sign, real = real)

                # Learning rate scheduler
                scheduler = CosineAnnealingLR(optimizer, T_max)


                # record initial params
                initial_params = model.record_params()
                params_ls.append(initial_params.copy())
                initial_params_summary[network_name] = initial_params.copy()
                
                # train model
                print(f'training {network_name}, pre-shuffle')
                train_model(train_loader, test_loader, val_loader, epochs, params_ls)

                # record and store trained params
                trained_params = model.record_params()
                trained_params_summary[network_name] = trained_params.copy()

                # shuffle data

                # train model
                #print(f'training {network_name}, with shuffle')
                #train_model(shuffle_train_loader, shuffle_test_loader, shuffle_val_loader, epochs, params_ls)
                
                
                # add to results dictionary - training_loss_summary, test_accuracy_summary
                params_summary[network_name] = params_ls
                training_loss_summary[network_name] = training_loss
                test_accuracy_summary[network_name] = test_accuracy
                val_accuracy_summary[network_name] = val_accuracy
                


# Save as pickle file
today = date.today()
filename = f'/Users/elliotjerng/Downloads/#Northeastern/coop/EP_LHb Grid Search/fMNIST_Model_Comparison_{today.strftime("%Y%m%d_%H%M%S")}_{random.random()}.pkl'
print('Saving to',filename)



with open(filename, 'wb') as f:
    data = {"training_loss_summary": training_loss_summary, "val_accuracy_summary": val_accuracy_summary, "test_accuracy_summary": test_accuracy_summary, 
            "initial_params_summary": initial_params_summary, "trained_params_summary": trained_params_summary, "params_summary": params_summary}
    pickle.dump(data, f)

print('Done')