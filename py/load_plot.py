import os
import scipy.io as scio
import random as rd

import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import torch

from torch.utils import data
from torchvision import transforms


def fill_nan(matrix):
    if np.isnan(np.min(matrix)):
        for i in range(matrix.shape[1]):
            temp_col = matrix[i, :]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
        return matrix
    else:
        return matrix


def datanorm(t):
    if not torch.is_tensor(t):
        t=torch.tensor(t,dtype=torch.float32)
    if torch.max(t):
        return t/torch.max(t)
    else:
        return t


def datanorm0_1(t):
    if not torch.is_tensor(t):
        t=torch.tensor(t,dtype=torch.float32)
    tmin=torch.min(t)
    tmax=torch.max(t)
    if (tmax-tmin):
        return (t-tmin)/(tmax-tmin)
    else:
        return t


# +
## Mat stack
class TrainDataGeneratorStep1(data.Dataset):
    def __init__(self, file_folder):
        self.img_folder = file_folder
        self.img_path = os.listdir(self.img_folder)

    def __getitem__(self, index):
        data = scio.loadmat(self.img_folder+os.sep+self.img_path[int(index)])
        x0 = fill_nan(data['im_X'])
#         try data['im_Y']
        y0 = data['im_Y']
        y1=data['im_YY']
        
        xx = datanorm(x0)
        yy0 = datanorm(y0)
        yy1 = datanorm(y1)
        
#         img = torch.from_numpy(xx).unsqueeze(0)
#         label0 = torch.from_numpy(yy0).unsqueeze(0)        
        img = xx.unsqueeze(0)
        label0 = yy0.unsqueeze(0)
        label1 = yy1.unsqueeze(0)
        return [img, label0, label1]
    
    def __len__(self):
        return len(self.img_path)


# -

class TrainDataGeneratorStep1_half(data.Dataset):
    def __init__(self, file_folder):
        self.img_folder = file_folder
        self.img_path = os.listdir(self.img_folder)

    def __getitem__(self, index):
        data = scio.loadmat(self.img_folder+os.sep+self.img_path[int(index)])
        x0 = fill_nan(data['im_X'])
        y0 = data['im_Y']
        xx = (x0-np.min(x0))/(np.max(x0)-np.min(x0))
        yy = (y0-np.min(y0))/(np.max(y0)-np.min(y0))
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        
        img = torch.from_numpy(xx).unsqueeze(0)
        label = torch.from_numpy(yy).unsqueeze(0)
        img=img.half()
        label=label.half()
        return [img, label]
    
    def __len__(self):
        return len(self.img_path)

class TrainDataGeneratorStep2(data.Dataset):
    def __init__(self, file_folder):
        self.img_folder = file_folder
        self.img_path = os.listdir(self.img_folder)

    def __getitem__(self, index):
        data = scio.loadmat(self.img_folder+os.sep+self.img_path[int(index)])
        x0 = data['im_X']
        y0 = data['im_Y']
        xx = (x0-np.min(x0))/(np.max(x0)-np.min(x0))
        yy = (y0-np.min(y0))/(np.max(y0)-np.min(y0))
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        
        img = torch.from_numpy(xx).unsqueeze(0)
        label = torch.from_numpy(yy).unsqueeze(0)
        return [img, label]
    
    def __len__(self):
        return len(self.img_path)

class TrainDataGeneratorStep2_half(data.Dataset):
    def __init__(self, file_folder):
        self.img_folder = file_folder
        self.img_path = os.listdir(self.img_folder)

    def __getitem__(self, index):
        data = scio.loadmat(self.img_folder+os.sep+self.img_path[int(index)])
        x0 = data['im_X']
        y0 = data['im_Y']
        xx = (x0-np.min(x0))/(np.max(x0)-np.min(x0))
        yy = (y0-np.min(y0))/(np.max(y0)-np.min(y0))
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        
        img = torch.from_numpy(xx).unsqueeze(0)
        label = torch.from_numpy(yy).unsqueeze(0)
        img=img.half()
        label=label.half()
        return [img, label]
    
    def __len__(self):
        return len(self.img_path)


class TestDataGenerator(data.Dataset):
    
    def __init__(self, file_folder):
        self.img_folder = file_folder
        self.img_path = os.listdir(self.img_folder)

    def __getitem__(self, index):
        data = scio.loadmat(self.img_folder+os.sep+self.img_path[index])
        x0 = fill_nan(data['data'])
        xx = (x0-np.min(x0))/(np.max(x0)-np.min(x0))
        xx = xx.astype(np.float16)
        img = torch.from_numpy(xx).unsqueeze(0)
        return img

    def __len__(self):
        return len(self.img_path)


def plot_metric(metric_history, metric,epoch,train_size,batch_size_train):
    """
    This function is used for plotting the following metrics of both train and validation:
    -- loss
    -- precision
    -- recall
    -- iou
    
    Arguments:
    metric_history -- a pd.dataFrame storing metric history
    metric -- a str indicating the metric need to be plotted
    
    Returns:
    --
                    
    """
    train_metrics = metric_history['train_'+metric]
    val_metrics = metric_history['val_'+metric]
    x = range(len(train_metrics))
#     plt.figure(facecolor='w',dpi=100)
    plt.plot(x, train_metrics,'-')
    plt.plot(x, val_metrics, '-')
    plt.ylim([0, 1])
    plt.title(metric)
    plt.xlabel('epoch = '+str(epoch)+', train_data_size='+str(train_size)+', batch_size='+str(batch_size_train))
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
#     plt.show()

# class TestDataGenerator(data.Dataset):

#     def __init__(self, file_folder,times_crop=1):

#         self.img_folder = file_folder
#         self.img_path = os.listdir(self.img_folder)

#         rd.seed(10)
#         self.cropx_index=rd.sample(set(np.linspace(0,384,4)),times_crop)
#         self.cropy_index=rd.sample(set(np.linspace(0,384,4)),times_crop)
#         self.times_crop=times_crop

#     def __getitem__(self, index):

#         data = scio.loadmat(self.img_folder+os.sep+self.img_path[index])
#         x0 = data['data']

#         xx = (x0-np.min(x0))/(np.max(x0)-np.min(x0))
#         xx = xx.astype(np.float32)

#         cropx=int(self.cropx_index[0])
#         cropy=int(self.cropy_index[0])

#         x_crop=xx[cropx:cropx+128,cropy:cropy+128]
#         img = torch.from_numpy(x_crop).unsqueeze(0)

#         return img

#     def __len__(self):
#         return len(self.img_path)*self.times_crop
