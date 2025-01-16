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
from functions_lib.earlystopping import EarlyStopping  # EarlyStopping
from to_npy import set_file

class HyperParameters:
    def __init__(self, date, hour, min):
        self.save_path = 'C:/Users/86136/Desktop/QPCNN'#!!You need to fill in the address of the file where the neural network is located
        self.output_save_path = '{}/output_Data/QPCNN_output'.format(self.save_path)  # Output data saving path
        self.model_save_path = '{}/output_Data/QPCNN_saved/{}-{}h{}min.pth'.format(self.save_path, date, hour, min)  # Save path for model parameters

        self.log_path = '{}/output_Data/QPCNN_log/{}-{}h{}min.log'.format(self.save_path,date, hour, min)  # Save path for logging
        self.logging_parameter = True # Whether to save data after training
        self.model_load_path = '{}/output_Data/QPCNN_saved/2024-08-07-1h21min.pth'.format(self.save_path)#
        self.training_resume = True   # Whether to use the previously saved model to continue training
        self.data_train = '{}/Data_train_correlated.npy'.format(self.save_path)
        self.data_test = '{}/Data_test_correlated.npy'.format(self.save_path)
        self.data_infer = '{}/Data_infer_correlated.npy'.format(self.save_path)

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

cur_date, cur_hour, cur_min = get_time()
hp = HyperParameters(cur_date, cur_hour, cur_min)

# Folder creation
for folder1 in ['QPCNN_log', 'QPCNN_saved', 'QPCNN_output']:#log,saved,output.
    os.makedirs('{}/output_Data/{}/'.format(hp.save_path,folder1), exist_ok=True)#exist_ok = True means the existence of this file is OK 
    if folder1 == 'QPCNN_output':#Inference means output, Validation is not used in our work. 
        for folder2 in ['validation', 'inference']:
            os.makedirs('{}/output_Data/{}/{}/'.format(hp.save_path,folder1,folder2), exist_ok=True)

# Log Setting 
logging.basicConfig(
    filename=hp.log_path,#Create a FileHandler with a file name called hp.log_path, that is, log information to the file hp.log_path
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',#Time - Level - Information
    datefmt='%m-%d %H:%M',#Month - Date Hour: minute
    level=logging.INFO#info
)

# Check whether the system can use a GPU for calculation. If no GPU is available, use a CPU
hp.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.info('Iteration times in this training: {}'.format(hp.iteration))  # Record the number of iterations for this training
logging.info('Computational Device: {}'.format(hp.device))  # The processor (CPU or GPU) used in the calculation is recorded in the log
logging.info('Learning parameters: lr-{}, lr_decay-{}, lr_patience-{}, early_patience-{}'.format(hp.lr, hp.lr_decay, hp.lr_patience, hp.early_patience))
logging.info('#' * 50)

# Model
qpcnn = QPCNN()  # 
qpcnn.to(hp.device) #CPU or GPU
optimizer = optim.Adam(qpcnn.parameters(), lr=hp.lr)  # Adam optimizer,
# optimizer = optim.SGD(spircnet.parameters(), lr=1e-3, momentum=0.99)  # SGD optimizer

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=hp.lr_decay, patience=hp.lr_patience,
                                                    verbose=True, threshold=hp.threshold, min_lr=hp.min_lr)  # learning rate scheduler
early_stopping = EarlyStopping(patience=hp.early_patience, delta=hp.min_delta, verbose=False, path=hp.model_save_path)

#=================================================================================pt and pm for entropy
p = pd.read_excel("{}/unfaked_data_correlated.xlsx".format(hp.save_path))
p = p.to_numpy()#recieve theta in unfaked data
pt_0 = p[:,7]
pt_1 = p[:,8]
pm_0 = p[:,9]
pm_1 = p[:,10]

#============================================================================================================
if hp.training_resume == True:  # inheriting the model
    checkpoint = torch.load(hp.model_load_path)  # loading
    qpcnn = torch.nn.DataParallel(qpcnn).cuda() #Multi-card parallel nn.DataParallel data previously reserved for multi-card parallel data
    qpcnn.load_state_dict(checkpoint['net'])  
    optimizer.load_state_dict(checkpoint['optimizer'])  
    #lr_scheduler.load_state_dict(checkpoint['lr_scheduler']) 
    logging.info('Load and continue training the existing model: {}'.format(hp.model_load_path))
    for param_group in optimizer.param_groups:
        param_group['lr'] = hp.lr #Go back to the original learning rate
else:  
    logging.info('New created model')
#============================================================================================================

since = time.time()  
set_file(hp.save_path)#Convert excel data in a file package to numpy for easy operation
data_input_train = np.load(hp.data_train)  # Loading training data
input_vector_train, qubit_vector_train, qubit_spin_train, input_spin_train = data_prepartion_trainandtest(data_input_train) # 处理训练集数据
instance_train= qubit_vector_train.shape[0]

data_input_test = np.load(hp.data_test)  # Loading testing data
input_vector_test, qubit_vector_test, qubit_spin_test,input_spin_test = data_prepartion_trainandtest(data_input_test) # 处理训练集数据
instance_test = input_vector_test.shape[0]

qpcnn = torch.nn.DataParallel(qpcnn)    #Multi-GPU acceleration
criterion_MSE_train = nn.MSELoss()  # Loss Function MAE(L1)
criterion_MAE_train = nn.L1Loss()  # Loss Function MSE
MSE_plot = []   # Draw an empty array and load MSE, MAE and Shannon entropy data
MAE_plot = []   
entropy_plot = [] 
epoch=0
#print(instance_train)
for iter in range(hp.iteration):
    #print(batches)
    for each_instance_train in range(0,instance_train):#batch = 1
        #=============================================Calculate MSE, MAE before each training===============================================
        if each_instance_train % hp.test_log == 0:
            loss_test_summary_MAE = 0
            loss_test_summary_MSE = 0
            summary_entropy = 0
            criterion_MAE_test = nn.L1Loss()
            criterion_MSE_test = nn.MSELoss()
            for each_instance_test in range(instance_test):
                output_spin_test = qpcnn(input_vector_test[each_instance_test,:,:].to(hp.device), qubit_vector_test[each_instance_test,:,:].to(hp.device), qubit_spin_test[each_instance_test,:,:].to(hp.device))
                loss_test_MSE = criterion_MSE_test(output_spin_test[0,:,:].to(hp.device), input_spin_test[each_instance_test,:,:].to(hp.device)).data  
                loss_test_summary_MSE += loss_test_MSE
                loss_test_MAE = criterion_MAE_test(output_spin_test[0,:,:].to(hp.device), input_spin_test[each_instance_test,:,:].to(hp.device)).data  
                loss_test_summary_MAE += loss_test_MAE
            loss_test_summary_MSE = loss_test_summary_MSE/instance_test
            loss_test_summary_MAE = loss_test_summary_MAE/instance_test
            MSE_plot.append(loss_test_summary_MSE.cpu().numpy())#MSE
            MAE_plot.append(loss_test_summary_MAE.cpu().numpy())#MAE
            lr_scheduler.step(loss_test_summary_MSE)  # Based on loss_test_summary_MAE to adjust Learning rate
            #=============================================output phase:===============================================
            data_input_infer = np.load(hp.data_infer)  
            input_vector_infer, qubit_vector_infer, qubit_spin_infer = data_prepartion_infer(data_input_infer) 
            instance_infer = input_vector_infer.shape[0]
            output_summary_infer = np.zeros((instance_infer,2))
            for each_instance_infer in range(instance_infer):
                output_spin_infer = qpcnn(input_vector_infer[each_instance_infer,:,:].to(hp.device), qubit_vector_infer[each_instance_infer,:,:].to(hp.device), qubit_spin_infer[each_instance_infer,:,:].to(hp.device)) #shape(1,1,8)
                output_spin_infer = output_spin_infer.view(1,2)
                output_spin_infer = output_spin_infer.detach().cpu().numpy() 
                
                po_0 = output_spin_infer[0,1]*pt_0[each_instance_infer]*pm_0[each_instance_infer] / (output_spin_infer[0,0]*pt_1[each_instance_infer]*pm_1[each_instance_infer] + output_spin_infer[0,1]*pt_0[each_instance_infer]*pm_0[each_instance_infer])
                po_0_clipped = np.clip(po_0, 1e-10, 1 - 1e-10)#ensure p != 0
                summary_entropy = summary_entropy - po_0_clipped * np.log(po_0_clipped) - (1 - po_0_clipped) * np.log(1 - po_0_clipped)
                #output_spin_infer[0,0] = po_0_clipped
                #output_spin_infer[0,1] = 1 - po_0_clipped
                output_summary_infer[each_instance_infer,:] = output_spin_infer#now we output pn, if you want to output po(n->infinite), use the last two lines of code

            summary_entropy = summary_entropy / instance_infer
            entropy_plot.append(summary_entropy)#entropy
            #print(output_spin_infer.shape)  #torch.Size([1, 1, 2])
            print('iter:{},epoch:{},validation number:{},MSE:{},MAE:{},entropy:{}'.format(iter,each_instance_train,instance_test,loss_test_summary_MSE,loss_test_summary_MAE,summary_entropy))
            
            output_summary_infer = np.concatenate((data_input_infer,output_summary_infer),axis=1)
            np.save(hp.output_save_path + '/inference/output.npy', output_summary_infer)  # save
            colname = ['input_a1_x','input_a1_y','input_a1_z',
                                                    'qubit_a1_x','qubit_a1_y','qubit_a1_z',
                                                    'qubit_a1_spin',
                                                    '0','1']
            
            pd.DataFrame(output_summary_infer,columns = colname).to_excel(hp.output_save_path + '/inference/output_iter{}_epoch{}.xlsx'.format(iter,epoch),index=True,
                                        columns = colname)
            epoch = epoch+1
            print('iter:{},epoch:{},output number:{}'.format(iter,each_instance_train,instance_infer))    

        qpcnn.train() 
        optimizer.zero_grad() # grading resetting
        output_spin_train = qpcnn(input_vector_train[each_instance_train,:,:].to(hp.device), qubit_vector_train[each_instance_train,:,:].to(hp.device), qubit_spin_train[each_instance_train,:,:].to(hp.device))
        loss_train = criterion_MSE_train(output_spin_train[0,:,:].to(hp.device), input_spin_train[each_instance_train,:,:].to(hp.device)) 
        loss_train.backward()  # (Calculate the gradient value of the parameter) by backpropagation
        optimizer.step()  # Update parameters (by gradient descent)

        if each_instance_train == instance_train-1:#last
            #sfwml.eval()    
            loss_test_summary_MAE = 0
            loss_test_summary_MSE = 0
            summary_entropy = 0
            criterion_MAE_test = nn.L1Loss()
            criterion_MSE_test = nn.MSELoss()
            for each_instance_test in range(instance_test):
                output_spin_test = qpcnn(input_vector_test[each_instance_test,:,:].to(hp.device), qubit_vector_test[each_instance_test,:,:].to(hp.device), qubit_spin_test[each_instance_test,:,:].to(hp.device))
                loss_test_MSE = criterion_MSE_test(output_spin_test[0,:,:].to(hp.device), input_spin_test[each_instance_test,:,:].to(hp.device)).data  
                loss_test_summary_MSE += loss_test_MSE
                loss_test_MAE = criterion_MAE_test(output_spin_test[0,:,:].to(hp.device), input_spin_test[each_instance_test,:,:].to(hp.device)).data 
                loss_test_summary_MAE += loss_test_MAE
            loss_test_summary_MSE = loss_test_summary_MSE/instance_test
            loss_test_summary_MAE = loss_test_summary_MAE/instance_test
            MSE_plot.append(loss_test_summary_MSE.cpu().numpy())#MSE
            MAE_plot.append(loss_test_summary_MAE.cpu().numpy())#MAE
            lr_scheduler.step(loss_test_summary_MSE)  
            #=============================================The same as previous work===============================================
            data_input_infer = np.load(hp.data_infer)  # 
            input_vector_infer, qubit_vector_infer, qubit_spin_infer = data_prepartion_infer(data_input_infer) # 
            instance_infer = input_vector_infer.shape[0]
            output_summary_infer = np.zeros((instance_infer,2))
            for each_instance_infer in range(instance_infer):
                output_spin_infer = qpcnn(input_vector_infer[each_instance_infer,:,:].to(hp.device), qubit_vector_infer[each_instance_infer,:,:].to(hp.device), qubit_spin_infer[each_instance_infer,:,:].to(hp.device)) #shape(1,1,8)
                output_spin_infer = output_spin_infer.view(1,2)
                output_spin_infer = output_spin_infer.detach().cpu().numpy() 
                
                po_0 = output_spin_infer[0,1]*pt_0[each_instance_infer]*pm_0[each_instance_infer] / (output_spin_infer[0,0]*pt_1[each_instance_infer]*pm_1[each_instance_infer] + output_spin_infer[0,1]*pt_0[each_instance_infer]*pm_0[each_instance_infer])
                po_0_clipped = np.clip(po_0, 1e-10, 1 - 1e-10)
                summary_entropy = summary_entropy - po_0_clipped * np.log(po_0_clipped) - (1 - po_0_clipped) * np.log(1 - po_0_clipped)
                #output_spin_infer[0,0] = po_0_clipped
                #output_spin_infer[0,1] = 1 - po_0_clipped
                output_summary_infer[each_instance_infer,:] = output_spin_infer

            summary_entropy = summary_entropy / instance_infer
            entropy_plot.append(summary_entropy)#entropy
            #print(output_spin_infer.shape)  #torch.Size([1, 1, 2])
            print('iter:{},epoch:{},validation number:{},MSE:{},MAE:{},entropy:{}'.format(iter,each_instance_train,instance_test,loss_test_summary_MSE,loss_test_summary_MAE,summary_entropy))
            output_summary_infer = np.concatenate((data_input_infer,output_summary_infer),axis=1)
            np.save(hp.output_save_path + '/inference/output.npy', output_summary_infer)  
            colname = ['input_a1_x','input_a1_y','input_a1_z',
                                                    'qubit_a1_x','qubit_a1_y','qubit_a1_z',
                                                    'qubit_a1_spin',
                                                    '0','1']
            
            pd.DataFrame(output_summary_infer,columns = colname).to_excel(hp.output_save_path + '/inference/output_iter{}_epoch{}.xlsx'.format(iter,epoch+1),index=True,
                                        columns = colname)
            print('iter:{},epoch:{},output number:{}'.format(iter,each_instance_train,instance_infer))

    if iter % hp.train_log == 0:# loss_train is recorded every hp.train_log iter
        print("---iter{} finished---".format(iter))
        logging.info('iter {}, train loss: {}'.format(iter, loss_train.data))
        logging.info('*' * 50)
#=================================================saving the model====================================================
if hp.logging_parameter == True:
    checkpoint = {'net': qpcnn.state_dict(), 'optimizer': optimizer.state_dict(),
              'lr_scheduler': lr_scheduler.state_dict()}
    torch.save(checkpoint, hp.model_save_path)  
#=================================================plotting and saving the data====================================================
plt.subplot(3,1,1)
plt.xlabel('batch',size = 5)
plt.ylabel('MSE',size = 5)
plt.plot(MSE_plot, color='#13533E', label='MSE_loss')
plt.subplot(3,1,2)
plt.xlabel('batch',size = 5)
plt.ylabel('MAE',size = 5)
plt.plot(MAE_plot, color='#13533E', label='MAE_loss')
plt.subplot(3,1,3)
plt.xlabel('batch',size = 15)
plt.ylabel('entropy',size = 15)
plt.plot(entropy_plot, color='#13533E', label='entropy')
plt.show()

data_PLOT = {'MSE_plot': MSE_plot,
        'MAE_plot': MAE_plot,
        'entropy_plot': entropy_plot}
df = pd.DataFrame(data_PLOT)
df.to_excel('{}/output_Data/QPCNN_output/error.xlsx'.format(hp.save_path), index=True)
#============================================================================================================
time_tv = time.time() - since
logging.info('training & validation time: {}h {}min {}s'.format(int(time_tv // 3600), int(time_tv % 3600 // 60), int(time_tv % 60)))
logging.info('#' * 50)

