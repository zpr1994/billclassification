# -*- coding: utf-8 -*-
"""
Created on Wed May  2 07:02:54 2018

@author: zpr
"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MyDataset
from model import Net

use_gpu = torch.cuda.is_available()

# =============================================================================
# 输出每一个类别的可能性
# =============================================================================
def test_5_possibility(root):

    test_data = MyDataset(txt=root+'label/simple_test.txt', transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=64)
    
    model = Net()
    model.load_state_dict(torch.load('weight/result.pkl'))
    if(use_gpu):
        model = model.cuda()      
    model.eval()
    
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        if(use_gpu):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)     
        if(use_gpu):
            out = out.cpu()
        out_array = out.data.numpy()
        for i in range(len(out_array)):
            for j in range(len(out_array[i])):
                if out_array[i][j] < 0:
                    out_array[i][j] = 0                           
            total = sum(out_array[i])
            for j in range(len(out_array[i])):                      
                out_array[i][j] /= total
                
            print(out_array[i])

# =============================================================================
# 输出每一个类别的准确率
# =============================================================================     
def test_5_class(root):     
    
    correct = [0, 0, 0, 0, 0]
    
    test_data = MyDataset(txt=root+'label/mytest.txt', transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=64)
    
    test_num_list = [0,0,0,0,0]
    file_read = open(root + 'label/mytest.txt', 'r')
    file_list = file_read.readlines()
    file_read.close()
    for i in range(len(file_list)):
        index = int(file_list[i].split(' ')[1])
        test_num_list[index] += 1
    
    model = Net()
    model.load_state_dict(torch.load('weight/result.pkl'))
    if(use_gpu):
        model = model.cuda()
    model.eval()
    
    for batch_x, batch_y in test_loader:    
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        if(use_gpu):
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)
        pred = torch.max(out, 1)[1]
        for i in range(len(pred)):
            if(pred[i].data[0] == batch_y[i].data[0]):
                correct[pred[i].data[0]] += 1          
    
    for i in range(len(correct)):
        correct[i] /= test_num_list[i]
    print(correct)

if __name__ == "__main__":
    
    root=""
    #test_5_possibility(root)
    test_5_class(root)
    
    