# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Import libraries

# +
import os
import scipy.io as scio
import sys
import random as rd

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
# %matplotlib inline

from datetime import datetime
import time

import glob
from skimage import io, transform, color
# -

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision import datasets 

# # Set gpu

# + code_folding=[]
from py.load_plot import *
from py.losses import *

# + code_folding=[]
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    gpunum=torch.cuda.device_count()
    currentgpu=torch.cuda.current_device()
    currentgpuname=torch.cuda.get_device_name(device)
    print('The number of GPU is {}.\n\nThe using one is : \nIndex: {} \nName: {}'.format(gpunum,device.index,currentgpuname))
else:
    device = torch.device("cpu")
    print('There is not available GPU. Using CPU instead.')


# -

# # Functions

# + code_folding=[0, 8]
def wgn_zeromean(x, snr):
    if not torch.is_tensor(x):
        x=torch.tensor(x,dtype=torch.float32)
    P_signal = torch.sum(torch.abs(x)**2)/x.numel()
    P_noise = P_signal/(10**(snr/10.0))
    noise=torch.randn(x.shape[0],x.shape[1])*torch.sqrt(P_noise)
    return torch.abs(datanorm(x+noise))

def wgn_mean(x, snr):
    if not torch.is_tensor(x):
        x=torch.tensor(x,dtype=torch.float32)
    P_signal = torch.sum(torch.abs(x)**2)/x.numel()
    P_noise = P_signal/(10**(snr/10.0))
    noise=torch.max(x)+torch.randn(x.shape[0],x.shape[1])*torch.sqrt(P_noise)
    return torch.abs(datanorm(x+noise))


# + code_folding=[0]
class ulmDataset(Dataset):
    def __init__(self, img_folder,lbl_folder, transform=None):
        self.img_folder = img_folder
        self.lbl_folder = lbl_folder
        self.image_name_list = sorted(os.listdir(self.img_folder))
        self.label_name_list = sorted(os.listdir(self.lbl_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.img_folder+os.sep+self.image_name_list[idx])
        label = io.imread(self.lbl_folder+os.sep+self.label_name_list[idx])
        image =np.expand_dims(image,axis=-1)
        label =np.expand_dims(label,axis=-1)
        imidx = np.array([idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
#             label=torch.where(label>0,torch.ones_like(label),torch.zeros_like(label))
        return [datanorm0_1(image), datanorm0_1(label)]


# + code_folding=[0]
class ulmDatasetGS_zeromean(Dataset):
    def __init__(self, img_folder,lbl_folder):
        self.img_folder = img_folder
        self.lbl_folder = lbl_folder
        self.image_name_list = sorted(os.listdir(self.img_folder))
        self.label_name_list = sorted(os.listdir(self.lbl_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.img_folder+os.sep+self.image_name_list[idx])
        label = io.imread(self.lbl_folder+os.sep+self.label_name_list[idx])
        imidx = np.array([idx])
        
        snr1=torch.rand(1)*40+10
        image=wgn_zeromean(datanorm0_1(image),int(snr1))
        label=datanorm0_1(label)
#         label=torch.where(label>0.32,torch.ones_like(label),torch.zeros_like(label))

        return [torch.unsqueeze(image,0), torch.unsqueeze(label,0)]


# + code_folding=[0]
class ulmDatasetGS(Dataset):
    def __init__(self, img_folder,lbl_folder):
        self.img_folder = img_folder
        self.lbl_folder = lbl_folder
        self.image_name_list = sorted(os.listdir(self.img_folder))
        self.label_name_list = sorted(os.listdir(self.lbl_folder))
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        image = io.imread(self.img_folder+os.sep+self.image_name_list[idx])
        label = io.imread(self.lbl_folder+os.sep+self.label_name_list[idx])
        imidx = np.array([idx])
        
        snr1=torch.rand(1)*40+10
        image=wgn(datanorm0_1(image),int(snr1))
        label=datanorm0_1(label)

        return [torch.unsqueeze(image,0), torch.unsqueeze(label,0)]


# + code_folding=[0]
def getpath(folder):
    return folder+os.sep+os.listdir(folder)[0]


# + code_folding=[0]
def t_norm(t):
    t0=torch.max(t)
    t1=torch.min(t)
    return (t-t1)/(t0-t1)


# + code_folding=[0]
def printTime(time_elapsed_all):
    time_hour = time_elapsed_all//3600
    time_min = time_elapsed_all % 3600//60
    time_sec = time_elapsed_all-time_hour*3600-time_min*60
    print('Time elapse: {:.0f}h {:.0f}m {:.0f}s'.format(time_hour, time_min, time_sec))


# + code_folding=[]
def makefolder(datanum,epoch,modelname):
    today = datetime.now()
    model_path = 'output_model'+ os.sep + modelname+os.sep+today.strftime('%Y%m%d') + '_' + today.strftime('%H%M%S') \
    +'_TrainSize'+str(datanum)+'_epoch'+str(epoch)

    # Create results directory if not already existing
    os.makedirs(model_path)
    print('Directory ' + model_path + ' succesfully created! \n')
    return model_path


# + code_folding=[]
def savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,name):
    history={}
    history['train_loss']=train_loss
    history['val_loss']=val_loss
    history['train_ssim']=np.array(train_ssim)
    history['val_ssim']=np.array(val_ssim)
    history['LearningRate']=lrlist
#     epochnp=np.reshape(range(1,epoch+1),[epoch,1])
#     batchnp=np.ones([1,batch_num_train])
#     epochhistory=(np.dot(epochnp,batchnp)).flatten()
#     history['epoch']=list(epochhistory)

    dfhistory=pd.DataFrame(history)
    dfhistory.to_csv(model_path+os.sep+'dfhistory_'+name+'.csv')
#     print("Saved train history to dfhistory.csv")


# -

# # Load img data

# + [markdown] heading_collapsed=true
# ## Temp Try

# + hidden=true
# trdir0 = 'data/20230802_256/train'
# trdir0_input = trdir0+'_input'
# trdir0_label0 = trdir0+'_label0'
# trdir0_label1 = trdir0+'_label1'

# transform_img0 = transforms.Compose(
#     [transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0)])

# all_data = ulmDatasetGS_zeromean(getpath(trdir0_input), getpath(trdir0_label0))
# datanum = len(all_data)

# train_size = int(0.8 * datanum)
# val_size = datanum - train_size
# train_data, val_data = torch.utils.data.random_split(
#     all_data, [train_size, val_size], generator=torch.Generator().manual_seed(1))
# train_dataloader = data.DataLoader(
#     train_data, batch_size=32, num_workers=4, shuffle=True)
# val_dataloader = data.DataLoader(
#     val_data, batch_size=16, num_workers=4, shuffle=True)

# seed=1
# torch.manual_seed(seed)
# for img_val, label0_val in val_dataloader:
#     print(img_val.shape)
#     print(label0_val.shape)
# #     print(batch_train)
#     break
# # label_res=torch.where(label0_val>0.32,torch.ones_like(label0_val),torch.zeros_like(label0_val))

# plt.figure(dpi=100,facecolor='w',figsize=(12,3))
# i=0
# plt.subplot(131)
# plt.imshow(torch.squeeze(img_val[i,:,:]),cmap='gray');plt.colorbar()
# plt.subplot(132)
# plt.imshow(torch.squeeze(label0_val[i,:,:]),cmap='gray');plt.colorbar()
# # plt.subplot(133)
# # plt.imshow(torch.squeeze(label_res[i,0,:,:]),cmap='gray');plt.colorbar()

# + code_folding=[] hidden=true
# Reload presaved model
# model = Unet2(1, 1).to(device)

# model.load_state_dict(torch.load(model_path+os.sep+"model_next10epoch_lr5e-4.pth"))
# model = torch.nn.DataParallel(model, device_ids=[0])
# -

# # Train the model

from py import restormer
from py import Unet5

from py import restore_CNN
from py import SwinIR_convStem
from py import SwinIR

# model = restore_CNN.Restormer()
model = restormer.Restormer()
# model = SwinIR_convStem.SwinIR(img_size=(128,128))
# model = SwinIR.SwinIR(img_size=(128,128))
# model = SwinIR_NoNorm.SwinIR(img_size=(128,128))
# model = Unet5.Unet5(1,1)
model.to(device)

ModelName='Restormer'
# ModelName='Restormer_CNN'
# ModelName='SwinIR_convStem'
# ModelName='SwinIR'
# ModelName='Unet5'

# + code_folding=[]
alpha=.8
lossname='Charbonnier Loss.csv'
# 'Charbonnier Loss.csv'
taskname='Restore'

# def loss_fn(d0, labels_v,alpha=alpha):
#     loss = lossf0(d0, labels_v)
#     return loss
def loss_fn(d0, labels_v,alpha=alpha):
    loss = torch.sqrt(torch.pow(d0.reshape(-1)-labels_v.reshape(-1),2)+1e-3**2)
    return torch.mean(loss)


# + [markdown] heading_collapsed=true
# ## Progressive learning

# + code_folding=[] hidden=true
def load_data(input_dir,label_dir,batch_size_train,batch_size_val):
    transform_img0 = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0)])

    all_data = ulmDatasetGS_zeromean(getpath(input_dir), getpath(label_dir))
    datanum = len(all_data)
    
    train_size = int(0.8 * datanum)
    val_size = datanum - train_size
    train_data,val_data=torch.utils.data.random_split(all_data,[train_size,val_size],generator=torch.Generator().manual_seed(1))
    train_dataloader = data.DataLoader(train_data, batch_size=batch_size_train,num_workers=4,shuffle=True)
    val_dataloader =  data.DataLoader(val_data, batch_size=batch_size_val,num_workers=4,shuffle=True)
    IterNum=train_size/batch_size_train
    
    return train_dataloader,val_dataloader,train_size


# -

# ## 128*128

# + code_folding=[]
trdir0='data/20230802_128/train'
trdir0_input=trdir0+'_input'
trdir0_label0=trdir0+'_label0'
trdir0_label1=trdir0+'_label1'

batch_size_train,batch_size_val=64,32
# batch_size_train,batch_size_val=1,1
[train_dataloader,val_dataloader,train_size]=load_data(trdir0_input,trdir0_label0,batch_size_train,batch_size_val)

epoch = 50
model_path=makefolder(train_size,epoch,ModelName)
os.makedirs(model_path+os.sep+'Models')

# Adam optimizer
LearningRate=1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=LearningRate,betas=(0.9, 0.999), eps=1e-07)

dfinfofile=pd.DataFrame({})
dfinfofile.to_csv(model_path+os.sep+lossname)
dfinfofile.to_csv(model_path+os.sep+taskname)

train_loss,val_loss = [],[]
lrlist = []
train_ssim,val_ssim=[],[]

since0 = time.time()
batch_num = 0
batch_num_train = len(train_dataloader)

ite_num = 1
save_frq = int(batch_num_train*8)
lossF=loss_fn

seed = 32
torch.manual_seed(seed)
for i in range(epoch):
    since_each_epoch = time.time()
    train_loss_batch,val_loss_batch = [],[]
    train_ssim_batch,val_ssim_batch = [],[]
    model.train()
    batch_train = 0
    for img_train, label_train0 in train_dataloader:

        since_each_batch = time.time()
        img_train, label_train0 = img_train.to(device), label_train0.to(device)
        inputs_v = Variable(img_train, requires_grad=False)
        labels_v = Variable(label_train0, requires_grad=False)

        optimizer.zero_grad()
        d1 = model(inputs_v)
        loss = lossF(d1, labels_v)
        
        lr_batch = optimizer.param_groups[0]['lr']
        lrlist.append(lr_batch)

        pred_train = datanorm(d1)
        
        train_ssim.append(myssimF(pred_train, label_train0).item())
        train_ssim_batch.append(myssimF(pred_train, label_train0).item())

        loss.backward()
        optimizer.step()
#         scheduler.step()

        train_loss.append(loss.item())
        train_loss_batch.append(loss.item())

        del d1,loss

        model.eval()
        correct,correct_ssim = [],[]

        with torch.no_grad():  # val loss
            for img_val, label_val0 in val_dataloader:
                img_val, label_val0 = img_val.to(device), label_val0.to(device)
                inputs_val_v = Variable(img_val, requires_grad=False)
                labels_val_v = Variable(label_val0, requires_grad=False)
                d1_val = model(inputs_val_v)
                val_loss_t = lossF(d1_val, labels_val_v)

                correct.append(val_loss_t.item())
                pred_val = datanorm(d1_val)
                correct_ssim.append(myssimF(pred_val, label_val0).item())
                
        val_loss.append(np.mean(np.array(correct)))
        val_loss_batch.append(np.mean(np.array(correct)))
        val_ssim.append(np.mean(np.array(correct_ssim)))
        val_ssim_batch.append(np.mean(np.array(correct_ssim)))

        if ite_num % save_frq == 0: # save model medium state
            torch.save(model.state_dict(), model_path+'/Models/' +str('{:03d}').format(ite_num//batch_num_train)+".pth")
            dfname=str('{:03d}').format(ite_num//batch_num_train)
            savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,dfname)
            model.train()  # resume train
        ite_num = ite_num+1
        batch_train = batch_train + 1
        batch_num = batch_num + 1
        time_elapsed_each_batch = time.time() - since_each_batch

        print('\r' + f'Epoch: {i+1} / {epoch}\
        Progress: {batch_train}/{batch_num_train}\
        Time elapse: {time_elapsed_each_batch:.1f}s\
        LR: {lr_batch:.5f}\
              Train_loss: {np.mean(np.array(train_loss_batch)):.5f}\
              Val_loss: {np.mean(np.array(val_loss_batch)):.5f}\
              Train_ssim: {np.mean(np.array(train_ssim_batch)):.5f}\
              Val_ssim: {np.mean(np.array(val_ssim_batch)):.5f} ', end='', flush=True)

    train_loss_batch = np.array(train_loss_batch)
    val_loss_batch = np.array(val_loss_batch)
    train_ssim_batch=np.array(train_ssim_batch)
    val_ssim_batch=np.array(val_ssim_batch)
    print('')
    print('1 epoch Time elapse: {:.1f}s'.format(time.time() - since_each_epoch))
    printTime(time.time() - since0)
    print('')

torch.save(model.state_dict(), model_path+'/Models/000.pth')
savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,'000')

# + [markdown] heading_collapsed=true
# ## 256*256

# + code_folding=[] hidden=true
# trdir0='data/20230802_256/train'
# trdir0_input=trdir0+'_input'
# trdir0_label0=trdir0+'_label0'
# trdir0_label1=trdir0+'_label1'

# batch_size_train,batch_size_val=16,4
# # batch_size_train,batch_size_val=12,4
# [train_dataloader,val_dataloader,train_size]=load_data(trdir0_input,trdir0_label0,batch_size_train,batch_size_val)

# epoch = 30
# model_path=makefolder(train_size,epoch,ModelName)
# os.makedirs(model_path+os.sep+'Models')

# # Adam optimizer
# LearningRate=5e-4
# optimizer = torch.optim.Adam(model.parameters(),lr=LearningRate,betas=(0.9, 0.999), eps=1e-07)

# dfinfofile=pd.DataFrame({})
# dfinfofile.to_csv(model_path+os.sep+lossname)
# dfinfofile.to_csv(model_path+os.sep+taskname)


# train_loss,val_loss = [],[]
# lrlist = []
# train_ssim,val_ssim=[],[]

# since0 = time.time()
# batch_num = 0
# batch_num_train = len(train_dataloader)

# ite_num = 1
# save_frq = int(batch_num_train*4)
# lossF=loss_fn

# seed = 3
# torch.manual_seed(seed)
# for i in range(epoch):
#     since_each_epoch = time.time()
#     train_loss_batch,val_loss_batch = [],[]
#     train_ssim_batch,val_ssim_batch = [],[]
#     model.train()
#     batch_train = 0
#     for img_train, label_train0 in train_dataloader:

#         since_each_batch = time.time()
#         img_train, label_train0 = img_train.to(device), label_train0.to(device)
#         inputs_v = Variable(img_train, requires_grad=False)
#         labels_v = Variable(label_train0, requires_grad=False)

#         optimizer.zero_grad()
#         d1 = model(inputs_v)
#         loss = lossF(d1, labels_v)
        
#         lr_batch = optimizer.param_groups[0]['lr']
#         lrlist.append(lr_batch)

#         pred_train = datanorm(d1)
        
#         train_ssim.append(myssimF(pred_train, label_train0).item())
#         train_ssim_batch.append(myssimF(pred_train, label_train0).item())

#         loss.backward()
#         optimizer.step()
# #         scheduler.step()

#         train_loss.append(loss.item())
#         train_loss_batch.append(loss.item())

#         del d1,loss

#         model.eval()
#         correct,correct_ssim = [],[]

#         with torch.no_grad():  # val loss
#             for img_val, label_val0 in val_dataloader:
#                 img_val, label_val0 = img_val.to(device), label_val0.to(device)
#                 inputs_val_v = Variable(img_val, requires_grad=False)
#                 labels_val_v = Variable(label_val0, requires_grad=False)
#                 d1_val = model(inputs_val_v)
#                 val_loss_t = lossF(d1_val, labels_val_v)

#                 correct.append(val_loss_t.item())
#                 pred_val = datanorm(d1_val)
#                 correct_ssim.append(myssimF(pred_val, label_val0).item())
                
#         val_loss.append(np.mean(np.array(correct)))
#         val_loss_batch.append(np.mean(np.array(correct)))
#         val_ssim.append(np.mean(np.array(correct_ssim)))
#         val_ssim_batch.append(np.mean(np.array(correct_ssim)))

#         if ite_num % save_frq == 0: # save model medium state
#             torch.save(model.state_dict(), model_path+'/Models/' +str('{:03d}').format(ite_num//batch_num_train)+".pth")
#             dfname=str('{:03d}').format(ite_num//batch_num_train)
#             savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,dfname)
#             model.train()  # resume train
#         ite_num = ite_num+1
#         batch_train = batch_train + 1
#         batch_num = batch_num + 1
#         time_elapsed_each_batch = time.time() - since_each_batch

#         print('\r' + f'Epoch: {i+1} / {epoch}\
#         Progress: {batch_train}/{batch_num_train}\
#         Time elapse: {time_elapsed_each_batch:.1f}s\
#         LR: {lr_batch:.5f}\
#               Train_loss: {np.mean(np.array(train_loss_batch)):.5f}\
#               Val_loss: {np.mean(np.array(val_loss_batch)):.5f}\
#               Train_ssim: {np.mean(np.array(train_ssim_batch)):.5f}\
#               Val_ssim: {np.mean(np.array(val_ssim_batch)):.5f} ', end='', flush=True)

#     train_loss_batch = np.array(train_loss_batch)
#     val_loss_batch = np.array(val_loss_batch)
#     train_ssim_batch=np.array(train_ssim_batch)
#     val_ssim_batch=np.array(val_ssim_batch)
#     print('')
#     print('1 epoch Time elapse: {:.1f}s'.format(time.time() - since_each_epoch))
#     printTime(time.time() - since0)
#     print('')

# torch.save(model.state_dict(), model_path+'/Models/000.pth')
# savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,'000')

# + [markdown] heading_collapsed=true
# ## 512*512

# + code_folding=[] hidden=true
# trdir0='data/20230802_512/train'
# trdir0_input=trdir0+'_input'
# trdir0_label0=trdir0+'_label0'
# trdir0_label1=trdir0+'_label1'

# batch_size_train,batch_size_val=4,2
# [train_dataloader,val_dataloader,train_size]=load_data(trdir0_input,trdir0_label0,batch_size_train,batch_size_val)


# epoch = 20
# model_path=makefolder(train_size,epoch,ModelName)
# os.makedirs(model_path+os.sep+'Models')

# # Adam optimizer
# LearningRate=2e-4
# optimizer = torch.optim.Adam(model.parameters(),lr=LearningRate,betas=(0.9, 0.999), eps=1e-07)

# dfinfofile=pd.DataFrame({})
# dfinfofile.to_csv(model_path+os.sep+lossname)
# dfinfofile.to_csv(model_path+os.sep+taskname)



# train_loss,val_loss = [],[]
# lrlist = []
# train_ssim,val_ssim=[],[]

# since0 = time.time()
# batch_num = 0
# batch_num_train = len(train_dataloader)

# ite_num = 1
# save_frq = int(batch_num_train*3)
# lossF=loss_fn

# seed = 3
# torch.manual_seed(seed)
# for i in range(epoch):
#     since_each_epoch = time.time()
#     train_loss_batch,val_loss_batch = [],[]
#     train_ssim_batch,val_ssim_batch = [],[]
#     model.train()
#     batch_train = 0
#     for img_train, label_train0 in train_dataloader:

#         since_each_batch = time.time()
#         img_train, label_train0 = img_train.to(device), label_train0.to(device)
#         inputs_v = Variable(img_train, requires_grad=False)
#         labels_v = Variable(label_train0, requires_grad=False)

#         optimizer.zero_grad()
#         d1 = model(inputs_v)
#         loss = lossF(d1, labels_v)
        
#         lr_batch = optimizer.param_groups[0]['lr']
#         lrlist.append(lr_batch)

#         pred_train = datanorm(d1)
        
#         train_ssim.append(myssimF(pred_train, label_train0).item())
#         train_ssim_batch.append(myssimF(pred_train, label_train0).item())

#         loss.backward()
#         optimizer.step()
# #         scheduler.step()

#         train_loss.append(loss.item())
#         train_loss_batch.append(loss.item())

#         del d1,loss

#         model.eval()
#         correct,correct_ssim = [],[]

#         with torch.no_grad():  # val loss
#             for img_val, label_val0 in val_dataloader:
#                 img_val, label_val0 = img_val.to(device), label_val0.to(device)
#                 inputs_val_v = Variable(img_val, requires_grad=False)
#                 labels_val_v = Variable(label_val0, requires_grad=False)
#                 d1_val = model(inputs_val_v)
#                 val_loss_t = lossF(d1_val, labels_val_v)

#                 correct.append(val_loss_t.item())
#                 pred_val = datanorm(d1_val)
#                 correct_ssim.append(myssimF(pred_val, label_val0).item())
                
#         val_loss.append(np.mean(np.array(correct)))
#         val_loss_batch.append(np.mean(np.array(correct)))
#         val_ssim.append(np.mean(np.array(correct_ssim)))
#         val_ssim_batch.append(np.mean(np.array(correct_ssim)))

#         if ite_num % save_frq == 0: # save model medium state
#             torch.save(model.state_dict(), model_path+'/Models/' +str('{:03d}').format(ite_num//batch_num_train)+".pth")
#             dfname=str('{:03d}').format(ite_num//batch_num_train)
#             savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,dfname)
#             model.train()  # resume train
#         ite_num = ite_num+1
#         batch_train = batch_train + 1
#         batch_num = batch_num + 1
#         time_elapsed_each_batch = time.time() - since_each_batch

#         print('\r' + f'Epoch: {i+1} / {epoch}\
#         Progress: {batch_train}/{batch_num_train}\
#         Time elapse: {time_elapsed_each_batch:.1f}s\
#         LR: {lr_batch:.5f}\
#               Train_loss: {np.mean(np.array(train_loss_batch)):.5f}\
#               Val_loss: {np.mean(np.array(val_loss_batch)):.5f}\
#               Train_ssim: {np.mean(np.array(train_ssim_batch)):.5f}\
#               Val_ssim: {np.mean(np.array(val_ssim_batch)):.5f} ', end='', flush=True)

#     train_loss_batch = np.array(train_loss_batch)
#     val_loss_batch = np.array(val_loss_batch)
#     train_ssim_batch=np.array(train_ssim_batch)
#     val_ssim_batch=np.array(val_ssim_batch)
#     print('')
#     print('1 epoch Time elapse: {:.1f}s'.format(time.time() - since_each_epoch))
#     printTime(time.time() - since0)
#     print('')

# torch.save(model.state_dict(), model_path+'/Models/000.pth')
# savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,'000')

# + [markdown] heading_collapsed=true
# ## 1024*1024

# + hidden=true
# trdir0='data/20230802_1024/train'
# trdir0_input=trdir0+'_input'
# trdir0_label0=trdir0+'_label0'
# trdir0_label1=trdir0+'_label1'

# batch_size_train,batch_size_val=1,1
# [train_dataloader,val_dataloader,train_size]=load_data(trdir0_input,trdir0_label0,batch_size_train,batch_size_val)

# epoch = 20
# model_path=makefolder(train_size,epoch,ModelName)
# os.makedirs(model_path+os.sep+'Models')

# # Adam optimizer
# LearningRate=1e-4
# optimizer = torch.optim.Adam(model.parameters(),lr=LearningRate,betas=(0.9, 0.999), eps=1e-07)

# dfinfofile=pd.DataFrame({})
# dfinfofile.to_csv(model_path+os.sep+lossname)
# dfinfofile.to_csv(model_path+os.sep+taskname)


# train_loss,val_loss = [],[]
# lrlist = []
# train_ssim,val_ssim=[],[]

# since0 = time.time()
# batch_num = 0
# batch_num_train = len(train_dataloader)

# ite_num = 1
# save_frq = int(batch_num_train*3)
# lossF=loss_fn

# seed = 3
# torch.manual_seed(seed)
# for i in range(epoch):
#     since_each_epoch = time.time()
#     train_loss_batch,val_loss_batch = [],[]
#     train_ssim_batch,val_ssim_batch = [],[]
#     model.train()
#     batch_train = 0
#     for img_train, label_train0 in train_dataloader:

#         since_each_batch = time.time()
#         img_train, label_train0 = img_train.to(device), label_train0.to(device)
#         inputs_v = Variable(img_train, requires_grad=False)
#         labels_v = Variable(label_train0, requires_grad=False)

#         optimizer.zero_grad()
#         d1 = model(inputs_v)
#         loss = lossF(d1, labels_v)
        
#         lr_batch = optimizer.param_groups[0]['lr']
#         lrlist.append(lr_batch)

#         pred_train = datanorm(d1)
        
#         train_ssim.append(myssimF(pred_train, label_train0).item())
#         train_ssim_batch.append(myssimF(pred_train, label_train0).item())

#         loss.backward()
#         optimizer.step()
# #         scheduler.step()

#         train_loss.append(loss.item())
#         train_loss_batch.append(loss.item())

#         del d1,loss

#         model.eval()
#         correct,correct_ssim = [],[]

#         with torch.no_grad():  # val loss
#             for img_val, label_val0 in val_dataloader:
#                 img_val, label_val0 = img_val.to(device), label_val0.to(device)
#                 inputs_val_v = Variable(img_val, requires_grad=False)
#                 labels_val_v = Variable(label_val0, requires_grad=False)
#                 d1_val = model(inputs_val_v)
#                 val_loss_t = lossF(d1_val, labels_val_v)

#                 correct.append(val_loss_t.item())
#                 pred_val = datanorm(d1_val)
#                 correct_ssim.append(myssimF(pred_val, label_val0).item())
                
#         val_loss.append(np.mean(np.array(correct)))
#         val_loss_batch.append(np.mean(np.array(correct)))
#         val_ssim.append(np.mean(np.array(correct_ssim)))
#         val_ssim_batch.append(np.mean(np.array(correct_ssim)))

#         if ite_num % save_frq == 0: # save model medium state
#             torch.save(model.state_dict(), model_path+'/Models/' +str('{:03d}').format(ite_num//batch_num_train)+".pth")
#             dfname=str('{:03d}').format(ite_num//batch_num_train)
#             savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,dfname)
#             model.train()  # resume train
#         ite_num = ite_num+1
#         batch_train = batch_train + 1
#         batch_num = batch_num + 1
#         time_elapsed_each_batch = time.time() - since_each_batch

#         print('\r' + f'Epoch: {i+1} / {epoch}\
#         Progress: {batch_train}/{batch_num_train}\
#         Time elapse: {time_elapsed_each_batch:.1f}s\
#         LR: {lr_batch:.5f}\
#               Train_loss: {np.mean(np.array(train_loss_batch)):.5f}\
#               Val_loss: {np.mean(np.array(val_loss_batch)):.5f}\
#               Train_ssim: {np.mean(np.array(train_ssim_batch)):.5f}\
#               Val_ssim: {np.mean(np.array(val_ssim_batch)):.5f} ', end='', flush=True)

#     train_loss_batch = np.array(train_loss_batch)
#     val_loss_batch = np.array(val_loss_batch)
#     train_ssim_batch=np.array(train_ssim_batch)
#     val_ssim_batch=np.array(val_ssim_batch)
#     print('')
#     print('1 epoch Time elapse: {:.1f}s'.format(time.time() - since_each_epoch))
#     printTime(time.time() - since0)
#     print('')

# torch.save(model.state_dict(), model_path+'/Models/000.pth')
# savehistory(train_loss,val_loss,train_ssim,val_ssim,epoch,batch_num_train,'000')
# -

# # Save loss & metric

history = pd.read_csv(model_path+'/dfhistory_000.csv')


# + code_folding=[]
def plot_metric(metric_history, metric, epoch, train_size, batch_size_train):

    train_metrics = metric_history['train_'+metric]
    val_metrics = metric_history['val_'+metric]
#     x=history['epoch']
    x = range(len(train_metrics))
    plt.plot(x, train_metrics, '-')
    plt.plot(x, val_metrics, '-')
    plt.ylim([0,1])
    plt.title(metric)
    plt.xlabel('Batchnum, epoch = '+str(epoch)+', train_data_size=' +
               str(train_size)+', batch_size='+str(batch_size_train))
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])

metriclist = ['loss','ssim']
plt.figure(dpi=100, facecolor='w', figsize=(10,6))
plt.subplot(121)
plot_metric(history, metriclist[0], epoch, train_size, batch_size_train)
plt.xlabel('Batchnum, epoch = '+str(epoch)+', train_data_size=' +str(train_size)+', batch_size='+str(batch_size_train))
plt.grid(linestyle='--', linewidth=1);plt.xlabel('')

plt.subplot(122)
plot_metric(history, metriclist[1], epoch, train_size, batch_size_train)
plt.grid(linestyle='--', linewidth=1)
plt.ylim([0,1])
# plt.subplot(224)
# plot_metric(history, metriclist[2], epoch, train_size, batch_size_train)
# plt.grid(linestyle='--', linewidth=1)
# plt.ylim([0,1])
plt.savefig(model_path + '/metrics.jpg', dpi=200)
# -

lrlist=history['LearningRate']
x = range(len(lrlist))
plt.figure(facecolor='w',dpi=100)
plt.plot(x, lrlist, '-')
plt.xlabel('Iteration = '+str(len(lrlist)))
plt.ylabel('Learning Rate')
plt.savefig(model_path + '/Learning Rate.jpg',dpi=200)

# + code_folding=[]
num=1 if img_val.shape[0]//2<1 else img_val.shape[0]//2

for ibatch in range(num):
    plt.figure(figsize=(12,3),facecolor='w')
    plt.subplot(131)
    plt.imshow(img_val[ibatch,0,:,:].cpu(),cmap='gray')
    plt.title('Input');plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(pred_val[ibatch,0,:,:].cpu(),cmap='gray')
    plt.title('Pred');plt.colorbar()

    plt.subplot(133)
    plt.imshow(label_val0[ibatch,0,:,:].cpu(),cmap='gray')
    plt.title('Label0');plt.colorbar()
    
    plt.savefig(model_path + '/val_example'+str(ibatch)+'.jpg',dpi=200)
