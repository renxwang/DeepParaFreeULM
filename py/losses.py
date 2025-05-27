import torch
import torch.nn.functional as F
import torch.jit
from torch import nn
import torchvision


# # SSIM

# ## MySSIM

# +
def mycov(a1,a2):
    a1=a1-torch.mean(a1)
    a2=a2-torch.mean(a2)
    return torch.mean(F.relu(a1*a2))

class myssimLoss(nn.Module):
    def __init__(self):
        super(myssimLoss, self).__init__()
        
    def forward(self, im1, im2):
        # im1, im2: N,C,H,W tensor
        num = im1.size(0)
        im1 = im1.reshape(num, -1)
        im2 = im2.reshape(num, -1)
    
        mu1=torch.mean(im1)
        mu2=torch.mean(im2)
        sigma1=torch.var(im1,unbiased=False)
        sigma2=torch.var(im2,unbiased=False)
        ssim1=(2*mu1*mu2+1e-4)*(2*mycov(im1,im2)+9e-4)
        ssim2=(mu1**2+mu2**2+1e-4)*(sigma1+sigma2+9e-4)
        
        if ssim1/ssim2 >1:
            ssim=1-ssim1/ssim2+ssim1/ssim2
            return torch.mean(ssim)
        else:
            ssim=1-ssim1/ssim2
        return torch.mean(ssim)

def myssimF(im1,im2):
    # im1, im2: N,C,H,W tensor
    num = im1.size(0)
    im1 = im1.reshape(num, -1)
    im2 = im2.reshape(num, -1)
    
    mu1=torch.mean(im1)
    mu2=torch.mean(im2)
    sigma1=torch.var(im1,unbiased=False)
    sigma2=torch.var(im2,unbiased=False)
    ssim1=(2*mu1*mu2+1e-4)*(2*mycov(im1,im2)+9e-4)
    ssim2=(mu1**2+mu2**2+1e-4)*(sigma1+sigma2+9e-4)
    ssim=ssim1/ssim2
    return torch.mean(ssim)


# -

# ## MS_SSIM

def create_window(window_size: int, sigma: float, channel: int):
    '''
    Create 1-D gauss kernel
    :param window_size: the size of gauss kernel
    :param sigma: sigma of normal distribution
    :param channel: input channel
    :return: 1D kernel
    '''
    coords = torch.arange(window_size, dtype=torch.float)
    coords -= window_size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    g = g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)
    return g


def _gaussian_filter(x, window_1d, use_padding: bool):
    '''
    Blur input with 1-D kernel
    :param x: batch of tensors to be blured
    :param window_1d: 1-D gauss kernel
    :param use_padding: padding image before conv
    :return: blured tensors
    '''
    C = x.shape[1]
    padding = 0
    if use_padding:
        window_size = window_1d.shape[3]
        padding = window_size // 2
    out = F.conv2d(x, window_1d, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window_1d.transpose(2, 3), stride=1, padding=(padding, 0), groups=C)
    return out


def ssim(X, Y, window, data_range: float, use_padding: bool=False):
    '''
    Calculate ssim index for X and Y
    :param X: images
    :param Y: images
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param use_padding: padding image before conv
    :return:
    '''

    K1 = 0.01
    K2 = 0.03
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mu1 = _gaussian_filter(X, window, use_padding)
    mu2 = _gaussian_filter(Y, window, use_padding)
    sigma1_sq = _gaussian_filter(X * X, window, use_padding)
    sigma2_sq = _gaussian_filter(Y * Y, window, use_padding)
    sigma12 = _gaussian_filter(X * Y, window, use_padding)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (sigma1_sq - mu1_sq)
    sigma2_sq = compensation * (sigma2_sq - mu2_sq)
    sigma12 = compensation * (sigma12 - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    # Fixed the issue that the negative value of cs_map caused ms_ssim to output Nan.
    cs_map = F.relu(cs_map)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_val = ssim_map.mean(dim=(1, 2, 3))  # reduce along CHW
    cs = cs_map.mean(dim=(1, 2, 3))

    return ssim_val, cs


def ms_ssim(X, Y, window, data_range: float, weights, use_padding: bool=False, eps: float=1e-8):
    '''
    interface of ms-ssim
    :param X: a batch of images, (N,C,H,W)
    :param Y: a batch of images, (N,C,H,W)
    :param window: 1-D gauss kernel
    :param data_range: value range of input images. (usually 1.0 or 255)
    :param weights: weights for different levels
    :param use_padding: padding image before conv
    :param eps: use for avoid grad nan.
    :return:
    '''
    weights = weights[:, None]

    levels = weights.shape[0]
    vals = []
    for i in range(levels):
        ss, cs = ssim(X, Y, window=window, data_range=data_range, use_padding=use_padding)

        if i < levels-1:
            vals.append(ss)
            X = F.avg_pool2d(X, kernel_size=2, stride=2, ceil_mode=True)
            Y = F.avg_pool2d(Y, kernel_size=2, stride=2, ceil_mode=True)
        else:
            vals.append(ss)

    vals = torch.stack(vals, dim=0)
    # Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
    vals = vals.clamp_min(eps)
    # The origin ms-ssim op.
    ms_ssim_val = torch.prod(vals[:-1] ** weights[:-1] * vals[-1:] ** weights[-1:], dim=0)
    # The new ms-ssim op. But I don't know which is best.
    # ms_ssim_val = torch.prod(vals ** weights, dim=0)
    # In this file's image training demo. I feel the old ms-ssim more better. So I keep use old ms-ssim op.
    return torch.mean(ms_ssim_val)


class SSIM(nn.Module):
    __constants__ = ['data_range', 'use_padding']

    def __init__(self, window_size=11, window_sigma=1.5, data_range=1., channel=1, use_padding=False):
        '''
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels (default: 3)
        :param use_padding: padding image before conv
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)
        self.data_range = data_range
        self.use_padding = use_padding

    def forward(self, X, Y):
        r = ssim(X, Y, window=self.window, data_range=self.data_range, use_padding=self.use_padding)
        return r[0]


class MS_SSIM(nn.Module):
    __constants__ = ['data_range', 'use_padding', 'eps']

    def __init__(self, window_size=5, window_sigma=1.5, data_range=1.,\
                 channel=1, use_padding=False, weights=None, levels=None, eps=1e-8):
        '''
        class for ms-ssim
        :param window_size: the size of gauss kernel
        :param window_sigma: sigma of normal distribution
        :param data_range: value range of input images. (usually 1.0 or 255)
        :param channel: input channels
        :param use_padding: padding image before conv
        :param weights: weights for different levels. (default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        :param levels: number of downsampling
        :param eps: Use for fix a issue. When c = a ** b and a is 0, c.backward() will cause the a.grad become inf.
        '''
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        self.data_range = data_range
        self.use_padding = use_padding
        self.eps = eps

        window = create_window(window_size, window_sigma, channel)
        self.register_buffer('window', window)

        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        weights = torch.tensor(weights, dtype=torch.float)

        if levels is not None:
            weights = weights[:levels]
            weights = weights / weights.sum()

        self.register_buffer('weights', weights)

    def forward(self, X, Y):
        return ms_ssim(X, Y, window=self.window, data_range=self.data_range, weights=self.weights,
                       use_padding=self.use_padding, eps=self.eps)





# # Dice Loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1
        
    def forward(self, predict, label):
        assert predict.size() == label.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        y_pre = predict.view(num, -1)
        y_true = label.view(num, -1)
        intersection = torch.abs((y_pre * y_true).sum(-1))
        union = (torch.abs(y_pre) + torch.abs(y_true)).sum(-1)

        score = 1 - (2 * intersection + self.epsilon) / (union + self.epsilon)
        return score.mean()


class DiceLoss_logis(nn.Module):
    def __init__(self):
        super(DiceLoss_logis, self).__init__()
        self.epsilon = 1

    def forward(self, predict, label):
        assert predict.size() == label.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        y_pre = torch.sigmoid(predict).view(num, -1)
        y_true = label.view(num, -1)
        intersection = torch.abs((y_pre * y_true).sum(-1))
        union = (torch.abs(y_pre) + torch.abs(y_true)).sum(-1)

        score = 1 - (2 * intersection + self.epsilon) / (union + self.epsilon)
        return score.mean()


class TverskyLoss(nn.Module):
    def __init__(self,alpha,beta):
        super(TverskyLoss, self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.epsilon = 1

    def forward(self, predict, label):
#         assert predict.size() == label.size(), "the size of predict and target must be equal."
        
        num = predict.size(0)
        y_pre = predict.view(num, -1)
        y_true = label.view(num, -1)
        
        TP=torch.abs((y_pre * y_true).sum(-1))
        FP=( torch.abs(y_pre)*(1-torch.abs(y_true)) ).sum(-1)
        FN=( torch.abs(y_true)*(1-torch.abs(y_pre)) ).sum(-1)
        score = 1 - (TP + self.epsilon) / (TP + FP * self.alpha + FN * self.beta + self.epsilon)

        return score.mean()


class Tver_L1Loss(nn.Module):
    def __init__(self,alpha,beta):
        super(Tver_L1Loss, self).__init__()
        self.alpha=alpha
        self.beta=beta

    def forward(self, predict, label):
        Tvfn=TverskyLoss(self.alpha,self.beta)
        TvScore=Tvfn(predict,label)
        
        FWHM_x,FWHM_z=10,10
        sigma_x = FWHM_x/2.355;
        sigma_z = FWHM_z/2.355;
        kernel_x=FWHM_x*2+1
        kernel_z=FWHM_z*2+1
        gauss_model=torchvision.transforms.GaussianBlur(kernel_size=(kernel_x,kernel_z), sigma=(sigma_x,sigma_z))
#         gauss_model=(gauss_model-torch.min(gauss_model)) / (torch.max(gauss_model)-torch.min(gauss_model))
        label_G=gauss_model(label)
        L1fn=nn.L1Loss()
        L1Score=L1fn(predict,label_G)

        return TvScore.mean()+L1Score.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self,alpha,beta,gamma):
        super(FocalTverskyLoss, self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.epsilon = 1

    def forward(self, predict, label):
        assert predict.size() == label.size(), "the size of predict and target must be equal."
        
        num = predict.size(0)
        y_pre = predict.view(num, -1)
        y_true = label.view(num, -1)
        
        TP=torch.abs((y_pre * y_true).sum(-1))
        FP=( torch.abs(y_pre)*(1-torch.abs(y_true)) ).sum(-1)
        FN=( torch.abs(y_true)*(1-torch.abs(y_pre)) ).sum(-1)
        TveScore = 1 - (TP + self.epsilon) / (TP + FP * self.alpha + FN * self.beta + self.epsilon)
        score=TveScore**self.gamma

        return score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1) 
        pred = pred.view(-1,1)
        target = target.view(-1,1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1-pred,pred),dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0],pred.shape[1])
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor. 
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0],pred.shape[1])
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

         # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


## Recall+L1 loss
class Model1Loss(nn.Module):
    def __init__(self):
        super(Model1Loss, self).__init__()
        self.epsilon = 1

    def forward(self, predict, label0, label1):
        assert predict.size() == label1.size(), "the size of predict and target must be equal."
        num = predict.size(0)
#         fn=nn.Softmax()
#         y_pre = torch.sigmoid(predict.view(num, -1))
        y_pre = predict.view(num, -1)
        y_true = label1.view(num, -1)
#         if torch.max(y_pre)==torch.min(y_pre):
#             y_pre=(y_pre-torch.min(y_pre))/(torch.max(y_pre)-torch.min(y_pre))
#         else:
#             y_pre=torch.rand_like(y_pre)

        TP = torch.abs((y_pre * y_true).sum(-1))
        FP = (torch.abs(y_pre)*(1-torch.abs(y_true))).sum(-1)
        FN = (torch.abs(y_true)*(1-torch.abs(y_pre))).sum(-1)

        RecallScore = 1 - (TP + self.epsilon) / (TP + FN + self.epsilon)

        L1fn = nn.L1Loss()
        score = RecallScore+L1fn(predict, label0)

        return RecallScore.mean()


## Tversky+BCE
class myLoss(nn.Module):
    def __init__(self,alpha,beta):
        super(myLoss, self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.epsilon = 1

    def forward(self, predict, label):
        assert predict.size() == label.size(), "the size of predict and target must be equal."
        
        num = predict.size(0)
        y_pre = predict.view(num, -1)
        y_true = label.view(num, -1)
        
        TP=torch.abs((y_pre * y_true).sum(-1))
        FP=( torch.abs(y_pre)*(1-torch.abs(y_true)) ).sum(-1)
        FN=( torch.abs(y_true)*(1-torch.abs(y_pre)) ).sum(-1)
        TvScore=1 - (TP + self.epsilon) / (TP + FP * self.alpha + FN * self.beta + self.epsilon)
        
        BCEfn=nn.BCELoss()
        
        score = TvScore+BCEfn(predict, label)

        return score.mean()


# Precision metric
def my_precisionbw(predict, label):
    predict = torch.where(predict>0.5,torch.ones_like(predict),torch.zeros_like(predict))
    num = predict.size(0)
    y_pre = predict.view(num, -1)
    y_true = label.view(num, -1)
    
    TP=torch.abs((y_pre * y_true).sum(-1))
    FP=( torch.abs(y_pre)*(1-torch.abs(y_true)) ).sum(-1)
    
    if (TP+FP).mean:
        score = TP/(TP+FP)
    else:
        score = 0
    return score.mean()

# Recall metric
def my_recallbw(predict, label):
    predict = torch.where(predict>0.5,torch.ones_like(predict),torch.zeros_like(predict))
    num = predict.size(0)
    y_pre = predict.view(num, -1)
    y_true = label.view(num, -1)
    
    TP=torch.abs((y_pre * y_true).sum(-1))
    FN=( torch.abs(y_true)*(1-torch.abs(y_pre)) ).sum(-1)
    
    score = TP/(TP+FN)
    return score.mean()

# IoU metric
def my_ioubw(predict, label):
    predict = torch.where(predict>0.5,torch.ones_like(predict),torch.zeros_like(predict))
    num = predict.size(0)
    y_pre = predict.view(num, -1)
    y_true = label.view(num, -1)
    
    TP=torch.abs((y_pre * y_true).sum(-1))
    FP=( torch.abs(y_pre)*(1-torch.abs(y_true)) ).sum(-1)
    FN=( torch.abs(y_true)*(1-torch.abs(y_pre)) ).sum(-1)
    
    score = TP/(TP+FP+FN)
    return score.mean()
