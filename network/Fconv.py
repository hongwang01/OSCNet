# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:58:33 2021

@author: XieQi
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as  F 
import math
from PIL import Image

class Fconv_PCA(nn.Module):

    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True, iniScale = 1.0):
       
        super(Fconv_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis, Rank, weight = GetBasis_PCA(sizeP,tranNum,inP, Smooth = Smooth)        
        self.register_buffer("Basis", Basis)#.cuda())        
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand, weight)*iniScale
        self.weights = nn.Parameter(iniw, requires_grad=True)  # 4*1*1*25  every rotation share the same weight
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        self.c = nn.Parameter(torch.zeros(1,inNum,1,1), requires_grad=bias)

    def forward(self, input):
        if self.training:
            tranNum = self.tranNum  # num_rot
            outNum = self.outNum    # num_M
            inNum = self.inNum      #  1
            expand = self.expand    #  1
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights) # num_M * num_rot * 1 *1 *sizeP *sizeP
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]
            tempW = torch.cat(tempWList, dim = 1)
            _filter = tempW.reshape([inNum*self.expand, outNum*tranNum, self.sizeP, self.sizeP])
            _bias = self.c.repeat([1,1,inNum,1]).reshape([1,inNum*self.expand,1,1])
        else:
            _filter = self.filter
            _bias   = self.bias
        output = F.conv2d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output + _bias, _filter
        
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
                del self.bias
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
            _filter = tempW.reshape([inNum*self.expand, outNum*tranNum, self.sizeP, self.sizeP])
            _bias = self.c.repeat([1,1,inNum,1]).reshape([1,inNum*self.expand,1,1])
            self.register_buffer("filter", _filter)
            self.register_buffer("bias", _bias)
        return super(Fconv_PCA, self).train(mode)  
    

def Getini_reg(nNum, inNum, outNum,expand, weight = 1): # sqrt(6)
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)
    return torch.FloatTensor(A)


def GetBasis_PCA(sizeP, tranNum=8, inP=None, Smooth = True):
    if inP==None:
        inP = sizeP
    inX, inY, Mask = MaskC(sizeP)
    X0 = np.expand_dims(inX,2)
    Y0 = np.expand_dims(inY,2)
    Mask = np.expand_dims(Mask,2)
    theta = np.arange(tranNum)/tranNum*2*np.pi
    theta = np.expand_dims(np.expand_dims(theta,axis=0),axis=0)
#    theta = torch.FloatTensor(theta)
    X = np.cos(theta)*X0-np.sin(theta)*Y0
    Y = np.cos(theta)*Y0+np.sin(theta)*X0
#    X = X.unsqueeze(3).unsqueeze(4)
    X = np.expand_dims(np.expand_dims(X,3),4)
    Y = np.expand_dims(np.expand_dims(Y,3),4)
    v = np.pi/inP*(inP-1)
    p = inP/2
    
    k = np.reshape(np.arange(inP),[1,1,1,inP,1])
    l = np.reshape(np.arange(inP),[1,1,1,1,inP])
    
    
    BasisC = np.cos((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    BasisS = np.sin((k-inP*(k>p))*v*X+(l-inP*(l>p))*v*Y)
    
    BasisC = np.reshape(BasisC,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)
    BasisS = np.reshape(BasisS,[sizeP, sizeP, tranNum, inP*inP])*np.expand_dims(Mask,3)

    BasisC = np.reshape(BasisC,[sizeP*sizeP*tranNum, inP*inP])
    BasisS = np.reshape(BasisS,[sizeP*sizeP*tranNum, inP*inP])

    BasisR = np.concatenate((BasisC, BasisS), axis = 1)
    
    U,S,VT = np.linalg.svd(np.matmul(BasisR.T,BasisR))

    Rank   = np.sum(S>0.0001)
    BasisR = np.matmul(np.matmul(BasisR,U[:,:Rank]),np.diag(1/np.sqrt(S[:Rank]+0.0000000001))) 
    BasisR = np.reshape(BasisR,[sizeP, sizeP, tranNum, Rank])
    
    temp = np.reshape(BasisR, [sizeP*sizeP, tranNum, Rank])
    var = (np.std(np.sum(temp, axis = 0)**2, axis=0)+np.std(np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0))/np.mean(np.sum(temp, axis = 0)**2+np.sum(temp**2*sizeP*sizeP, axis = 0),axis = 0)
    Trod = 1
    Ind = var<Trod
    Rank = np.sum(Ind)
    Weight = 1/np.maximum(var, 0.04)/25
    if Smooth: # after svd, freq
        BasisR = np.expand_dims(np.expand_dims(np.expand_dims(Weight,0),0),0)*BasisR

    return torch.FloatTensor(BasisR), Rank, Weight

def MaskC(SizeP):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        C    =X**2+Y**2
        
        Mask = np.ones([SizeP,SizeP])
#        Mask[C>(1+1/(4*p))**2]=0
        Mask = np.exp(-np.maximum(C-1,0)/0.2)
        
        return X, Y, Mask
    

