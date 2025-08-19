import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import cv2
import time
import datetime
import logging
import os
import copy
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))#
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..\..")))#

class HyperParameters_pretrain:
    def __init__(self, save_path, date, hour, min):#!!You need to fill in the address of the file where the neural network is located
        self.save_path = save_path# Output data saving path
        self.output_save_path = '{}/output_Data/QPCNN_output'.format(self.save_path)  # Save path for model parameters
        self.model_save_path = '{}/output_Data/QPCNN_saved/{}-{}h{}min.pth'.format(self.save_path, date, hour, min)  

        self.log_path = '{}/output_Data/QPCNN_log/{}-{}h{}min.log'.format(self.save_path,date, hour, min)  # Save path for logging
        self.logging_parameter = True 
        self.model_load_path = '{}/output_Data/QPCNN_saved/2024-07-09-6h55min.pth'.format(self.save_path)#No use in pretraining
        self.training_resume = False 
        self.data_train = '{}/Dataset/pm_uncorrelated_train.npy'.format(self.save_path)
        self.data_test = '{}/Dataset/pm_uncorrelated_validation.npy'.format(self.save_path)
        self.data_infer = '{}/Dataset/Data_infer_uncorrelated.npy'.format(self.save_path)

        self.iteration = 1 # The number of iterations of the training mode, we set 1 in our work
        self.test_log = 2000 #We set each epoch as 2000 instances.
        
        self.train_log = 1  # In training mode, each iteration (train_log) is recorded in the log
        self.val_log = 1  # During each iteration (val_log) in training mode, the relevant data is recorded in log
        self.train2val = 2  # No use
        # encoder/decoder parameters
        self.dropout_rate = 0.5
        # optimizer parameters  
        self.lr = 1e-4  # learning rate 5e-5
        self.min_lr = 1e-12  # Minimum learning rate 1e-12
        self.lr_decay = 0.3  # Learning rate decay rate0.3
        self.lr_patience = 2  #   2
        self.threshold = 1e-6   #1e-6
        # Early Stop
        self.early_patience = 3
        self.min_delta = 0
        '''
        Early stopping is a method that allows you to specify an arbitrary large number of training epochs and 
        stop training once the model performance stops improving on a hold out validation dataset.'''
        pass

class HyperParameters_output:
    def __init__(self, save_path, date, hour, min, load_path):
        self.save_path = save_path #!!You need to fill in the address of the file where the neural network is located
        self.output_save_path = '{}/output_Data/QPCNN_output'.format(self.save_path)  # Output data saving path
        self.model_save_path = '{}/output_Data/QPCNN_saved/{}-{}h{}min.pth'.format(self.save_path, date, hour, min)  # Save path for model parameters

        self.log_path = '{}/output_Data/QPCNN_log/{}-{}h{}min.log'.format(self.save_path,date, hour, min)  # Save path for logging
        self.logging_parameter = True # Whether to save data after training
        self.model_load_path = load_path#
        self.training_resume = True   # Whether to use the previously saved model to continue training
        self.data_train = '{}/Dataset/pm_correlated_train.npy'.format(self.save_path)
        self.data_test = '{}/Dataset/pm_correlated_validation.npy'.format(self.save_path)
        self.data_infer = '{}/Dataset/pm_correlated_output.npy'.format(self.save_path)

        self.iteration = 1 # The number of iterations of the training mode, we set 1 in our work
        self.test_log = 2000 #We set each epoch as 2000 instances.
        
        self.train_log = 1  # In training mode, each iteration (train_log) is recorded in the log
        self.val_log = 1  # During each iteration (val_log) in training mode, the relevant data is recorded in log
        self.train2val = 2  # No use
        # encoder/decoder parameters
        self.dropout_rate = 0.5
        # optimizer parameters  
        self.lr = 5e-5  # learning rate 5e-5
        self.min_lr = 1e-12  # Minimum learning rate 1e-12
        self.lr_decay = 0.3  # Learning rate decay rate0.3
        self.lr_patience = 2  #   2
        self.threshold = 1e-6   #1e-6
        # Early Stop
        self.early_patience = 3
        self.min_delta = 0
        '''
        Early stopping is a method that allows you to specify an arbitrary large number of training epochs and 
        stop training once the model performance stops improving on a hold out validation dataset.'''
        pass

def get_time():   
    cur_time = datetime.datetime.now()   
    date = cur_time.date()   
    hour = cur_time.hour    
    min = cur_time.minute   
    return date, hour, min

def to_npy(file_path):
    # 读取Excel文件
    excel_file_path = "{}.xlsx".format(file_path)
    df = pd.read_excel(excel_file_path)

    # 将DataFrame转换为NumPy数组
    numpy_array = df.to_numpy()

    # 保存为.npy文件
    npy_file_path = "{}.npy".format(file_path)
    np.save(npy_file_path, numpy_array)

def set_file(save_path):
    train_file_path = '{}/Dataset/pm_correlated_train'.format(save_path)
    to_npy(train_file_path)

    test_file_path = '{}/Dataset/pm_correlated_validation'.format(save_path)
    to_npy(test_file_path)

    infer_file_path = '{}/Dataset/pm_correlated_output'.format(save_path)
    to_npy(infer_file_path)

    train_file_path = '{}/Dataset/pm_uncorrelated_train'.format(save_path)
    to_npy(train_file_path)

    test_file_path = '{}/Dataset/pm_uncorrelated_validation'.format(save_path)
    to_npy(test_file_path)

    
def data_prepartion_trainandtest(data_input):#Data preparation for training set and validation set
    data_input = torch.Tensor(data_input).to(torch.float32).cuda()  # Turn to torch.float

    input_vector = data_input[:,:3]#xa,ya,za
    input_vector = input_vector.view(data_input.shape[0],1,3)
    #print('input_vector_train.shape:',input_vector.shape) #torch.Size([batch, 1, 3])

    qubit_vector = data_input[:,3:6] #xb,yb,zb
    qubit_vector = qubit_vector.view(qubit_vector.shape[0], 1, 3) #(batch, 1, 3)
    #print('qubit_vector_train.shape:',qubit_vector_train.shape)#torch.Size([batch, 1, 3])

    qubit_spin = data_input[:,6:7]    #The spin state of a
    qubit_spin = qubit_spin.view(qubit_spin.shape[0],qubit_spin.shape[1],1) #(batch, 1, 1)
    #print('qubit_spin_train.shape:',qubit_spin_train.shape)#torch.Size([batch, 1, 1])

    input_spin = data_input[:,7:9]  #pm(0|axy),pm(1|axy)
    input_spin = input_spin.view(input_spin.shape[0],1,input_spin.shape[1])
    #print('input_vector_train.shape:',input_vector_train.shape) #torch.Size([batch, 1, 2])

    return input_vector, qubit_vector, qubit_spin, input_spin

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, verbose=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf  # np.inf表示"正无穷"，没有确切的数值，类型为浮点型
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, train_loss, model):

        score = -train_loss

        if self.best_score is None:  # 预设置：self.best_score = None
            self.best_score = score  # 预设置：self.best_score = score == -train_loss
            self.save_checkpoint(train_loss, model)  # 保存模型并令"新的train_loss_min = 旧的train_loss"
        elif score < self.best_score + self.delta:  # 每次iteration中，score < best_score + delta
            self.counter += 1
            if self.counter >= self.patience - 5:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # self.trace_func即为print
            if self.counter >= self.patience:  # counter >= patience
                self.early_stop = True
        else:  # 每次iteration中，score >= best_score + delta
            self.best_score = score
            self.counter = 0  # counter置零

    def save_checkpoint(self, train_loss, model):
        '''Saves model when train loss decreases.'''
        if self.verbose:  # verbose == True
            self.trace_func(f'Train loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), self.path)  # 保存模型至path路径
        self.train_loss_min = train_loss

def data_prepartion_infer(data_input):#Data preparation for output set
    data_input = torch.Tensor(data_input).to(torch.float32).cuda()  # torch.float

    input_vector = data_input[:,:3]
    input_vector = input_vector.view(data_input.shape[0],1,3)
    #print('input_vector_train.shape:',input_vector.shape) #torch.Size([batch, 1, 3])

    qubit_vector = data_input[:,3:6]
    qubit_vector = qubit_vector.view(qubit_vector.shape[0], 1, 3) #(batch, 1, 3)
    #print('qubit_vector_train.shape:',qubit_vector_train.shape)#torch.Size([batch, 1, 3])

    qubit_spin = data_input[:,6:7]   #The spin state of a
    qubit_spin = qubit_spin.view(qubit_spin.shape[0],qubit_spin.shape[1],1) #(batch, 1, 1)
    #print('qubit_spin_train.shape:',qubit_spin_train.shape)#torch.Size([batch, 1, 1])

    return input_vector, qubit_vector, qubit_spin

class Encoder_input_vector(nn.Module):#3->64
    def __init__(self):
        super(Encoder_input_vector, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # Fully Connected
        self.fc2 = nn.Linear(32, 64)  
        self.leaky_relu = nn.LeakyReLU()  # Activiation function
        self.elu = nn.ELU()  # 
        # self.dropout = nn.Dropout(hp.dropout_rate)

    def forward(self, x):  # 
        # FC1 -> ELU -> FC2 -> ELU
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        return x
      
class Encoder_qubit_vector(nn.Module):#3->64
    def __init__(self):
        super(Encoder_qubit_vector, self).__init__()
        self.fc1 = nn.Linear(3, 32)  
        self.fc2 = nn.Linear(32, 64)  
        self.fc3 = nn.Linear(64, 128)
        self.leaky_relu = nn.LeakyReLU() 
        self.elu = nn.ELU()  
        # self.dropout = nn.Dropout(hp.dropout_rate)

    def forward(self, x):  
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        return x

class Encoder_qubit_spin(nn.Module):#1->64
    def __init__(self):
        super(Encoder_qubit_spin, self).__init__()
        self.fc1 = nn.Linear(1, 32)   
        self.fc2 = nn.Linear(32, 64)  
        self.fc3 = nn.Linear(64, 128)
        self.leaky_relu = nn.LeakyReLU()  
        self.elu = nn.ELU() 
        # self.dropout = nn.Dropout(hp.dropout_rate)

    def forward(self, x): 
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        return x

class Decoder_input_spin(nn.Module):  # output:768->2
    def __init__(self):
        super(Decoder_input_spin, self).__init__()
        self.fc00 = nn.Linear(2048, 1024)
        self.fc0 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        self.leaky_relu = nn.LeakyReLU()   
        self.elu = nn.ELU()   
        self.relu = nn.ReLU()   
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):#768->2
        x = x.view(1,x.shape[0],x.shape[1])
        #x = self.sigmoid(self.fc00(x))
        #x = self.sigmoid(self.fc0(x))
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        x = x / torch.sum(x)#Normalize
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(Bottleneck, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.conv1 = nn.Conv2d(in_size, mid_size, kernel_size=(1, 1), stride=(1, 1))  # Conv2D
        self.bn1 = nn.BatchNorm2d(mid_size)  
        self.conv2 = nn.Conv2d(mid_size, mid_size, kernel_size=(3, 3), stride=(1, 1))  # Conv2D
        self.bn2 = nn.BatchNorm2d(mid_size)  
        self.conv3 = nn.Conv2d(mid_size, out_size, kernel_size=(1, 1), stride=(1, 1))  # Conv2D
        self.bn3 = nn.BatchNorm2d(out_size)  
        self.leaky_relu = nn.LeakyReLU(inplace=True)  
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=(1, 1), stride=(1, 1)),  
            nn.BatchNorm2d(out_size),  
        )   

        #Attention mechanism
        #self.se_attention = SEAttention(out_size)
    def forward(self, x):  
        if self.in_size == self.out_size:
            identity = x
        else:
            identity = self.down_sample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')  # replicate mode performs better than zero mode
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = self.leaky_relu(x)

        #x = SEAttention(self.out_size)(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolution & Pooling
        self.bottleneck64_128 = Bottleneck(64, 32, 128)
        self.bottleneck128_256 = Bottleneck(128, 64, 256)
        self.bottleneck256_512 = Bottleneck(256, 128, 512)
        self.bottleneck512_1024 = Bottleneck(512, 256, 1024)
        # self.se_attention64_128 = SEAttention(128)  # Add SEAttention to this layer
        # self.se_attention128_256 = SEAttention(256)  # Add SEAttention to this layer
        # self.se_attention256_512 = SEAttention(512)  # Add SEAttention to this layer
        # self.se_attention512_1024 = SEAttention(1024)  # Add SEAttention to this layer
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1), return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1), return_indices=True)
        
    def convolution(self, x):
        #print(x.shape)  #torch.Size([1, 64, 3, 1])
        x = self.bottleneck64_128(x)
        #print(x.shape)  #torch.Size([1, 128, 3, 1])
        x = self.bottleneck128_256(x)
        #print(x.shape)  #torch.Size([1, 256, 3, 1]))
        x = torch.flatten(x)
        x = x.view(1, x.shape[0], 1, 1)
        #print(x.shape) #torch.Size([1, 768, 1, 1])   
        return x

    def forward(self, x):
        x = self.convolution(x)
        return x

class QPCNN(nn.Module):
    def __init__(self):
        super(QPCNN, self).__init__()
        self.encoder_input_vector = Encoder_input_vector()#3->64
        self.encoder_qubit_vector = Encoder_qubit_vector()#3->64
        self.encoder_qubit_spin = Encoder_qubit_spin()#1->64
        self.cnn = CNN()
        self.fc_49_199 = nn.Linear(49*49, 199*199)
        self.decoder_input_spin = Decoder_input_spin()
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU()

    def forward(self, input_vector, qubit_vector, qubit_spin):
        #input_vector:(1, 3), qubit_vector:(1, 3), qubit_spin:(1, 1)
        input_vector = input_vector.view(1, input_vector.shape[0], input_vector.shape[1]) # (1, 3)->(1, 1, 3)
        qubit_vector = qubit_vector.view(1, qubit_vector.shape[0], qubit_vector.shape[1]) # (1, 3)->(1, 1, 3)
        qubit_spin = qubit_spin.view(1, qubit_spin.shape[0], qubit_spin.shape[1]) # (1, 1)->(1, 1, 1)

        input_vector = self.encoder_input_vector(input_vector) # (1, 1, 3)->(1, 1, 64)
        qubit_vector = self.encoder_qubit_vector(qubit_vector) # (1, 1, 3)->(1, 1, 64)
        qubit_spin = self.encoder_qubit_spin(qubit_spin) # (1, 1, 1)->(1, 1, 64)

        #print(qubit_spin.shape) #torch.Size([1, 1, 64])
        x = torch.cat((input_vector, qubit_vector, qubit_spin), 0).cuda()  # (3, 1, 64)
        #print(x.shape) #torch.Size([3, 3, 64])
        x = x.view(1, x.shape[2], x.shape[0], x.shape[1])  # (1, 64, 3, 1)
        # CNN
        x = self.cnn(x)  # (1, 768, 1, 1)
        x = x.view(x.shape[2], x.shape[1], x.shape[0])# (1, 1, 768)
        input_spin = self.decoder_input_spin(x)# (1, 1, 2)
        return input_spin