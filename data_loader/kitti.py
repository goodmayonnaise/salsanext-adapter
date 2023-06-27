import os, cv2
import numpy as np 
from einops import rearrange

import torch
from torch.utils.data import Dataset

class KITTI(Dataset):
    def __init__(self, path, shape, nclasses, mode) -> None:
        self.path = self.path_loader(path, mode)

        self.x = self.path[0]
        self.y = self.path[1]

        self.shape = shape
        self.nclasses = nclasses

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        x = cv2.imread(x)
        x = np.array(x)
        x = cv2.resize(x, (self.shape[1], self.shape[0]))
        x = torch.FloatTensor(x)
        x = rearrange(x, 'h w c -> c h w')
        
        y = self.y[idx]
        y = cv2.imread(y)
        y = np.array(y)
        y = cv2.resize(y, (self.shape[1], self.shape[0]))
        y = torch.FloatTensor(y)
        y = rearrange(y, 'h w c -> c h w')

        _y = y[0]
        h, w = _y.size()
        target = torch.zeros(self.nclasses, h, w)
        for c in range(self.nclasses):
            target[c] = (_y==c).type(torch.int32).clone().detach()

        return x, y, target
    
    def path_loader(self, path, mode):
        xs, ys= [], []

        x_dir = os.path.join(path, 'img_dir', mode)
        x = os.listdir(x_dir)
        for png in x:
            xs.append(os.path.join(x_dir, png))

        y_dir = os.path.join(path, 'ann_dir', mode)
        y = os.listdir(y_dir)
        for png in y :
            ys.append(os.path.join(y_dir, png))

        return xs, ys

if __name__ == "__main__":
    path = '/vit-adapter-kitti/data/kitti'
    dataset = KITTI(path, (256,1024), 20, 'train')
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, 1)
    
    for i,j in enumerate(loader):
        # print(i)
        input = j[0]
        target = j[-1]
        print()


    
