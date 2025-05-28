import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3  , num_filters=8):
        super(UNetGenerator, self).__init__()
        self.encoder1 = self.conv_block(in_channels, num_filters)
        self.encoder2 = self.conv_block(num_filters, num_filters * 2)
        self.encoder3 = self.conv_block(num_filters * 2, num_filters * 4)
        self.encoder4 = self.conv_block(num_filters * 4, num_filters * 8)
        
        self.decoder3 = self.upconv_block(num_filters * 8 + 3, num_filters * 4)
        self.decoder2 = self.upconv_block(num_filters * 4, num_filters * 2)
        self.decoder1 = self.upconv_block(num_filters * 2, num_filters)
        self.decoder0 = nn.ConvTranspose2d(num_filters, out_channels, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 ,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, img, label):
        enc1 = self.encoder1(img)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        B, C, H, W = enc4.shape
        mix = torch.cat((enc4, label.view(B, 3, 1, 1).expand(B, 3, H, W)), dim=1)
        
        dec3 = self.decoder3(mix)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)
        dec0 = self.decoder0(dec1)

        return dec0