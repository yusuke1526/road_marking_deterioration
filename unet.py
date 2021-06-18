import torch
import torch.nn as nn

# https://tarovel4842.hatenablog.com/entry/2019/11/10/165426

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        # contracting path 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # contracting path 2
        self.pool2 = nn.MaxPool2d(2, stride=2) # 1/2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        # contracting path 3
        self.pool3 = nn.MaxPool2d(2, stride=2) # 1/4
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)

        # contracting path 4
        self.pool4 = nn.MaxPool2d(2, stride=2) # 1/8
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)

        # contracting path 5
        self.pool5 = nn.MaxPool2d(2, stride=2) # 1/16
        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        # expansive path 1
        self.dconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)

        # expansive path 2
        self.dconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu7_2 = nn.ReLU(inplace=True)

        # expansive path 3
        self.dconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu8_2 = nn.ReLU(inplace=True)

        # expansive path 4
        self.dconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu9_2 = nn.ReLU(inplace=True)
        self.conv9_3 = nn.Conv2d(64, self.n_classes, 1)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        output1 = self.relu1_2(self.conv1_2(h))

        h = self.pool2(output1)
        h = self.relu2_1(self.conv2_1(h))
        output2 = self.relu2_2(self.conv2_2(h))

        h = self.pool3(output2)
        h = self.relu3_1(self.conv3_1(h))
        output3 = self.relu3_2(self.conv3_2(h))

        h = self.pool4(output3)
        h = self.relu4_1(self.conv4_1(h))
        output4 = self.relu4_2(self.conv4_2(h))

        h = self.pool5(output4)
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))

        upsample1 = self.dconv1(h)
        h = torch.cat((output4, upsample1), dim=1)
        h = self.relu6_1(self.conv6_1(h))
        h = self.relu6_2(self.conv6_2(h))

        upsample2 = self.dconv2(h)
        h = torch.cat((output3, upsample2), dim=1)
        h = self.relu7_1(self.conv7_1(h))
        h = self.relu7_2(self.conv7_2(h))

        upsample3 = self.dconv3(h)
        h = torch.cat((output2, upsample3), dim=1)
        h = self.relu8_1(self.conv8_1(h))
        h = self.relu8_2(self.conv8_2(h))

        upsample4 = self.dconv4(h)
        h = torch.cat((output1, upsample4), dim=1)
        h = self.relu9_1(self.conv9_1(h))
        h = self.relu9_2(self.conv9_2(h))
        h = self.conv9_3(h)
        
        # Binary classification
        h = torch.sigmoid(h)

        return h