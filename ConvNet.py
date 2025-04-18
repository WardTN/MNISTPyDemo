from torch import nn


class ConvNet(nn.Module):
    #卷积层 对图像进行卷积操作,提取图像特征点
    #卷积层中的卷积核可以共享参数,即在卷积操作中
    #池化层 降低特征图的大小 从而减少模型参数

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            # [BATCH_SIZE, 1, 28, 28]
            # 输入通道为 1 输出通道为32 卷积核为 5 padding 为 2
            nn.Conv2d(1, 32, 5, 1, 2),
            # [BATCH_SIZE, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [BATCH_SIZE, 32, 14, 14]
            # 输入通道为 32 输出通道为 64 卷积核为 5 padding 为2
            nn.Conv2d(32, 64, 5, 1, 2),
            # [BATCH_SIZE, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [BATCH_SIZE, 64, 7, 7]
        )

        #全连接层 将卷积层获得的特征向量映射到类别概率

        self.fc = nn.Linear(64 * 7 * 7, 10)
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y
