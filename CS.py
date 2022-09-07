# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import numpy as np
from scipy.fftpack import fft,ifft



def OMP(Z, A, K=None):
    # Z : class_num*sparse_dim,   [10, 2]
    # A : sparse_dim*feature_dim,  [2, 512]
    # indexs : K*batch_size,  K*10
    # A_new :  sparse_dim*K, 2*K
    # theta :  K*num_class,  K*10
    Z = torch.from_numpy(Z).clone().detach()
    A = torch.from_numpy(A).clone().detach()
    # A = torch.where(torch.isnan(A), torch.full_like(A,0), A)
    
    sparse_n, feat_n = A.shape  # 32*512
    class_num, _ = Z.shape  # 10*32
    x_rec = torch.zeros([class_num, feat_n]) # 10*512
    if K==None:
        K = sparse_n
    residual = Z   #  初始化残差值为Z
    indexs = []
    
    for i in range(K):
        products = torch.sum(torch.mul(A.T, residual[:, None, :]), dim=2, keepdim=False)  #  10*512
        # print(products.shape)
        val, ind = torch.max(torch.abs(products), dim=1)  #  10
        indexs.append(ind)  #  K*10
        inv_indexs = torch.stack(indexs,0).T  #  10*K
        A_new = A[:,inv_indexs[1]]  #  32*K
       
        theta = torch.tensor(np.linalg.pinv(A_new).dot(Z.T)) # 最小二乘估计 K*10
        residual = Z- torch.mm(A_new, theta).T # 更新残差  10*32（sparse_dim*K)*（K*batch_size）
        
    x_rec[:,inv_indexs[1]] = theta.T.to(torch.float) # 迭代K次,X_rec中放入非零解值K个 10*512
    return x_rec
    
def OMP_Separate(Y, A, labels): # K=None
    # Y : [10, 2]
    # A : [10, 2, 512]
    # labels: [64, 1]
    
    Y = torch.from_numpy(Y[labels.T.cpu().numpy().tolist()]) # [64, 2]
    A = torch.from_numpy(A[labels.T.cpu().numpy().tolist()]) # [64, 2, 512]
    # A = torch.where(torch.isnan(A), torch.full_like(A,0), A)
    # print('A: ',A.shape, A.dtype)
    batch, sparse_n, feat_n = A.shape  # [64, 2, 512]
    x_rec = torch.zeros([batch, feat_n]) # [64, 512]
    # if K==None:
    #     K = sparse_n
    K = sparse_n
    residual = Y   #  初始化残差r0 值为y
    
    # indexs :  [K, 64]
    # A_new :   [64, 2, K]
    # theta :   [64, K]
    indexs = []
    A_new =  torch.zeros(A.shape)
    for i in range(K):
        products = torch.bmm(torch.transpose(A, dim0=1, dim1=2),residual[:, :, None]).squeeze(dim=2) #  [64, 512]
        val, ind = torch.max(torch.abs(products), dim=1)  #  64
        indexs.append(ind)  #  K*64
        inv_indexs = torch.stack(indexs,0).T  #  [64, K]
        # print('inv_indexs: ',inv_indexs.shape, inv_indexs[2])
        A_new = A[:,:,inv_indexs[2]]
        # print('A_new: ',A_new, A_new.shape)
        
        #  A(10, 2, 512), A_new(10, 2), A_pinv(2, 10), Y(10, 2)
        theta = torch.tensor(np.linalg.pinv(A_new).dot(Y.T)) # 最小二乘估计 计算一次θ  # 10*K*10
        # print('theta.shape: ',theta.shape)
        residual = Y- torch.sum(torch.bmm(A_new, theta).T , dim=-1)# 更新残差   # [64, 2] （sparse_dim*K)*（K*batch_size）
        # print('residual: ', residual.shape, residual)
        
    # 迭代K次,放入X_rec K个解值
    x_rec[:,inv_indexs[1]] = torch.sum(theta, dim=2).to(torch.float32) # 10*512
    
    return x_rec

 

if __name__ == '__main__':
    # A = torch.randn(10,2,512)
    # Z = torch.ones(10,2)
    # labels = torch.zeros(64,1)
    # x_rec = OMP_Separate(Z,A,labels) # [64,512]
    
    A = torch.randn(2,5)
    Z = torch.ones(10,2)
    x_rec = OMP(Z.numpy(),A.numpy())
    
    print('x_rec.shape: ',x_rec.shape)