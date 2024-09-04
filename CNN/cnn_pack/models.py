import torch
import torch.nn as nn
import torch.nn.functional as F

class NetBatchNorm(nn.Module):
    def __init__(self, n_chans1, ten_size, poolten_size):
        super().__init__()
        self.poolten_size = poolten_size
        self.ten_size = ten_size
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(5, n_chans1, kernel_size=3, padding='same')
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, 
                               padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1 // 2)
        self.fc1 = nn.Linear( self.poolten_size*(n_chans1 // 2), 15000)
        self.fc2 = nn.Linear(15000, self.ten_size)
        
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2, ceil_mode=True)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2, ceil_mode=True)
        out = out.view(-1,self.poolten_size* (self.n_chans1 // 2))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

class NetDropout(nn.Module):
    def __init__(self, n_chans1, ten_size, poolten_size):
        super().__init__()
        self.poolten_size = poolten_size
        self.ten_size = ten_size
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(5, n_chans1, kernel_size=3, padding='same')
        self.conv1_dropout = nn.Dropout2d(p=0.4)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.conv2_dropout = nn.Dropout2d(p=0.4)
        self.fc1 = nn.Linear(self.poolten_size*(n_chans1 // 2), 15000)
        self.fc2 = nn.Linear(15000, self.ten_size)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2, ceil_mode=True)
        out = self.conv1_dropout(out)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2, ceil_mode=True)
        out = self.conv2_dropout(out)
        out = out.view(-1,self.poolten_size* (self.n_chans1 // 2))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out    
    
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                              padding=1, bias=False)  
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class NetResDeep(nn.Module):
    def __init__(self, n_chans1, ten_size, poolten_size, n_blocks=10):
        super().__init__()
        self.poolten_size = poolten_size
        self.ten_size = ten_size
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(5, n_chans1, kernel_size=3, padding='same')
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(self.poolten_size * n_chans1, 15000)
        self.fc2 = nn.Linear(15000, self.ten_size)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2, ceil_mode=True)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = out.view(-1, self.poolten_size * self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out