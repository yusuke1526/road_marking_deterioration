import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

class MyDataset(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
    
    def __init__(self, data_dir, img_dir, mask_dir, transform1=None, transform2=None):
        self.img_paths = self._get_img_paths(data_dir + '/' + img_dir)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path.replace(self.img_dir, self.mask_dir)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)[:, :, -1:]
        # road marking is 6
        mask = np.where(mask == 6, 1.0, 0.0)
        if self.transform1:
            img = self.transform1(img)
        if self.transform2:
            mask = self.transform2(mask)
        return img, mask[:1].to(torch.float), img_path
    
    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [
            str(p) for p in img_dir.iterdir() if p.suffix in MyDataset.IMG_EXTENSIONS
        ]

        return sorted(img_paths)
    
    def __len__(self):
        return len(self.img_paths)
    
class DAVIDDataset(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
    
    def __init__(self, data_dir, transform1=None, transform2=None):
        self.img_dir = data_dir + '/Images'
        self.mask_dir = data_dir + '/Labels'
        self.img_paths = self._get_img_paths(self.img_dir)
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = img_path.replace('Images', 'Labels')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        # road marking is [157, 234, 50]
        mask = np.where(mask == [157, 234, 50], 1.0, 0.0)[:, :, 1:2]
        if self.transform1:
            img = self.transform1(img)
        if self.transform2:
            mask = self.transform2(mask)
        return img, mask.to(torch.float), img_path
    
    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [
            str(p) for p in img_dir.iterdir() if p.suffix in DAVIDDataset.IMG_EXTENSIONS
        ]

        return sorted(img_paths)
    
    def __len__(self):
        return len(self.img_paths)