# -*- coding: utf-8 -*-
"""
Created on Wed May  2 01:00:27 2018

@author: zpr
"""

from torch.utils.data import Dataset
from PIL import Image

# =============================================================================
# 图像格式转化    
# =============================================================================
def default_loader(path):
    return Image.open(path).convert('RGB')

# =============================================================================
# 数据加载
# =============================================================================
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)       
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)