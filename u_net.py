import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        # Insert your code here
        self.in_planes=in_planes
        self.out_planes=out_planes

        self.conv1=nn.Conv2d(in_planes, out_planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_planes)
        self.conv2=nn.Conv2d(out_planes, out_planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_planes)


    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))

        return out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.block1 = BasicBlock(3, 64)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = BasicBlock(512, 512)
        self.maxpool = nn.MaxPool2d(2)

        self.block6 = BasicBlock(2*512, 256)
        self.block7 = BasicBlock(2*256, 128)
        self.block8 = BasicBlock(2*128, 64)
        self.block9 = BasicBlock(2*64, 64)

        self.conv = nn.Conv2d(64,3,kernel_size=1)

    def forward(self, x, return_last_layer=False, return_both = False,second_head=False):
        out11 = self.block1(x)
        out21 = self.block2(self.maxpool(out11))
        out31 = self.block3(self.maxpool(out21))
        out41 = self.block4(self.maxpool(out31))
        out5 = self.maxpool(out41)

        out42 = F.interpolate(self.block5(out5),size=out41.size()[2:])
        out4 = torch.cat((out41,out42),1)

        out32 = F.interpolate(self.block6(out4),size=out31.size()[2:])
        out3 = torch.cat((out31,out32),1)

        out22 = F.interpolate(self.block7(out3),size=out21.size()[2:])
        out2 = torch.cat((out21,out22),1)

        out12 = F.interpolate(self.block8(out2),size=out11.size()[2:])
        out1 = torch.cat((out11,out12),1)

        out = self.block9(out1)
        out = torch.sigmoid(self.conv(out))

        return out
