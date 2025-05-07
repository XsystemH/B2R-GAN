import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3*2+6, out_channels=3, num_filters=64):
        super(UNetGenerator, self).__init__()
        self.encoder1 = self.conv_block(in_channels, num_filters)
        self.encoder2 = self.conv_block(num_filters, num_filters * 2)
        self.encoder3 = self.conv_block(num_filters * 2, num_filters * 4)
        self.encoder4 = self.conv_block(num_filters * 4, num_filters * 8)
        self.bottleneck = self.conv_block(num_filters * 8, num_filters * 8)

        self.decoder4 = self.upconv_block(num_filters * 8, num_filters * 8)
        self.decoder3 = self.upconv_block(num_filters * 8, num_filters * 4)
        self.decoder2 = self.upconv_block(num_filters * 4, num_filters * 2)
        self.decoder1 = nn.ConvTranspose2d(num_filters * 2, out_channels, kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck) + enc4
        dec3 = self.decoder3(dec4) + enc3
        dec2 = self.decoder2(dec3) + enc2
        dec1 = self.decoder1(dec2)

        return dec1