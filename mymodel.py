import torch
import torch.nn as nn
import torch.nn.functional as F


class dilateattention(nn.Module):
    def __init__(self, in_channel, depth):
        super(dilateattention, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        # self.mean = nn.AdaptiveAvgPool2d((1, 1))
        # k=1 s=1 no pad
        self.batch_norm = nn.BatchNorm3d(in_channel)
        self.relu = nn.ReLU()
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 1, 1)
        self.atrous_block2 = nn.Conv3d(in_channel, depth, 3, 1, padding=2, dilation=2, groups=in_channel)
        self.atrous_block3 = nn.Conv3d(in_channel, depth, 3, 1, padding=4, dilation=4, groups=in_channel)
        self.conv_1x1_output = nn.Conv3d(depth * 5, depth, 1, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        v = self.atrous_block1(x)
        q = self.atrous_block2(x)
        k = self.atrous_block3(x)
        temp = k * q
        # dk = torch.std(temp)
        output = v * self.softmax(temp)
        output += self.relu(self.batch_norm(x))
        return output

class mymodel(nn.Module):
    def __init__(self, in_channel,classnum=1):
        super(mymodel, self).__init__()
        self.dim = 8
        self.conv3d1 = nn.Conv3d(in_channel, self.dim, kernel_size=(7,7,7), padding=3,stride=1)
        self.maxpooling1 = nn.MaxPool3d(kernel_size=2,stride=2)
        self.dialatiaon1 = dilateattention(self.dim,self.dim)
        self.conv3d2 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon2 = dilateattention(self.dim, self.dim)
        self.conv3d3 = nn.Conv3d(self.dim, self.dim, kernel_size=(7, 7, 7), padding=3, stride=1)
        self.maxpooling3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dialatiaon3 = dilateattention(self.dim, self.dim)
        self.transpose1 = nn.ConvTranspose3d(self.dim,self.dim,kernel_size=2,stride=2)
        self.conv2d1 = nn.Conv3d(self.dim,self.dim, kernel_size=(7,7,7),padding=(3,3,3),stride=1)
        self.transpose2 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv3d(self.dim, self.dim, kernel_size=7, padding=3, stride=1)
        self.transpose3 = nn.ConvTranspose3d(self.dim * 2, self.dim, kernel_size=2, stride=2)
        self.conv2d3 = nn.Conv3d(self.dim, 1, kernel_size=7, padding=3, stride=1)
        self.final = nn.Conv2d(152, classnum, 3, 1, 1)
    def forward(self,x):
        x1 = self.conv3d1(x)
        x1 = self.maxpooling1(x1)
        x1 = self.dialatiaon1(x1)
        x2 = self.conv3d2(x1)
        x2 = self.maxpooling2(x2)
        x2 = self.dialatiaon2(x2)
        x3 = self.conv3d3(x2)
        x3 = self.maxpooling3(x3)
        x3 = self.dialatiaon3(x3)
        x4 = self.transpose1(x3)
        x4 = self.conv2d1(x4)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.transpose2(x4)
        x5 = self.conv2d2(x5)
        x6 = torch.cat([x5,x1], dim=1)
        x6 = self.transpose3(x6)
        x6 = self.conv2d3(x6)
        x6 = torch.squeeze(x6,dim=1)
        x6 = self.final(x6)
        return x6






