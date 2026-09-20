import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, norm='batch', residual=False, activation='leakyrelu',
                 transpose=False):
        super(ConvBlock, self).__init__()
        self.dropout = dropout
        self.residual = residual  # 
        self.activation = activation  # 
        self.transpose = transpose

        if self.dropout:
            self.dropout1 = nn.Dropout2d(p=0.05) # dropout防止过拟合
            self.dropout2 = nn.Dropout2d(p=0.05)

        self.norm1 = None
        self.norm2 = None
        if norm == 'batch':  # 默认采用 batch normalization
            self.norm1 = nn.BatchNorm2d(out_channels)  # 输出处进行batch norm
            self.norm2 = nn.BatchNorm2d(out_channels)
        elif norm == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'mixed':
            self.norm1 = nn.BatchNorm2d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)

        # [2025-11-06] Unet Double Conv block
        # padding=1, 两均为1进行padding, 不改变像素数目
        if self.transpose: # Upsampling
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, padding=1)
        else: # Downsampling
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if self.activation == 'relu':
            self.actfun1 = nn.ReLU()
            self.actfun2 = nn.ReLU()
        elif self.activation == 'leakyrelu':
            self.actfun1 = nn.LeakyReLU()
            self.actfun2 = nn.LeakyReLU()
        elif self.activation == 'elu':
            self.actfun1 = nn.ELU()
            self.actfun2 = nn.ELU()
        elif self.activation == 'selu':
            self.actfun1 = nn.SELU()
            self.actfun2 = nn.SELU()

    def forward(self, x):
        ox = x

        # 1. 先进行一次卷积
        x = self.conv1(x)

        # 2. dropout weight
        if self.dropout:
            x = self.dropout1(x)

        # 3. batch norm
        if self.norm1:
            x = self.norm1(x)

        # 4. activation 激活函数
        x = self.actfun1(x)


        # 567 进行第二次卷积+drop+batch
        x = self.conv2(x)

        if self.dropout:
            x = self.dropout2(x)

        if self.norm2:
            x = self.norm2(x)

        # 8. 若块中包含了residual, 则直接用input和刚刚第7步的结果相加
        # [batch, channel, height, width] 对共有的channel进行相加
        # [2025-11-06] 相当于是block内做了一个residual
        if self.residual:
            x[:, 0:min(ox.shape[1], x.shape[1]), :, :] += ox[:, 0:min(ox.shape[1], x.shape[1]), :, :]

        x = self.actfun2(x)

        # print("shapes: x:%s ox:%s " % (x.shape,ox.shape))

        return x