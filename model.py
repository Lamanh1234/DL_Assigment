import torch
from torch import nn
import timm



class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, skip_layer):
        x = torch.cat([x, skip_layer], axis=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class bottleneck_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


# UNet model
class UNet(nn.Module):
    def __init__(self, n_class=3):
        super(UNet, self).__init__()
        # Encoder blocks

        ## backbone:
        self.backbone = timm.create_model("resnet101", pretrained=True, features_only=True)

        # Bottleneck block
        self.bottleneck = bottleneck_block(2048, 1024)

        # Decoder blocks
        self.dec1 = decoder_block(1024 + 1024, 512)
        self.dec2 = decoder_block(512 + 512, 256)
        self.dec3 = decoder_block(256 + 256, 128)
        self.dec4 = decoder_block(128 + 64, 64)

        # upsampling
        self.transpose_conv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2)
        self.transpose_conv2 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.transpose_conv3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.transpose_conv4 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.transpose_conv5 = nn.ConvTranspose2d(n_class, n_class, kernel_size=2, stride=2)

        # 1x1 convolution
        self.out = nn.Conv2d(64, n_class, kernel_size=1, padding='same')

    def forward(self, image):
        n1, n2, n3, n4, n5 = self.backbone(image)

        n6 = self.bottleneck(n5)

        n7 = self.dec1(self.transpose_conv1(n6), n4)
        n8 = self.dec2(self.transpose_conv2(n7), n3)
        n9 = self.dec3(self.transpose_conv3(n8), n2)
        n10 = self.dec4(self.transpose_conv4(n9), n1)

        output = self.out(n10)

        return self.transpose_conv5(output)