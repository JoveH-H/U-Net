import paddle
import paddle.nn as nn


class PetNet(nn.Layer):
    def __init__(self, num_classes):
        super(PetNet, self).__init__()

        self.in_channel_layer_1 = 3
        self.out_channel_layer_1 = 16
        self.out_channel_layer_2 = 32
        self.out_channel_layer_3 = 64
        self.out_channel_layer_4 = 128
        self.out_channel_layer_5 = 256
        self.in_channel_layer_2 = self.out_channel_layer_1
        self.in_channel_layer_3 = self.out_channel_layer_2
        self.in_channel_layer_4 = self.out_channel_layer_3
        self.in_channel_layer_5 = self.out_channel_layer_4
        self.in_channel_layer_up_4 = self.out_channel_layer_4 * 2
        self.in_channel_layer_up_3 = self.out_channel_layer_3 * 2
        self.in_channel_layer_up_2 = self.out_channel_layer_2 * 2
        self.in_channel_layer_up_1 = self.out_channel_layer_1 * 2
        self.out_channel_layer_up_4 = self.out_channel_layer_4
        self.out_channel_layer_up_3 = self.out_channel_layer_3
        self.out_channel_layer_up_2 = self.out_channel_layer_2
        self.out_channel_layer_up_1 = self.out_channel_layer_1

        # Encoder
        self.conv_1 = nn.Sequential(
            nn.Conv2D(self.in_channel_layer_1, self.out_channel_layer_1, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_1),
            nn.ReLU(),
            nn.Conv2D(self.out_channel_layer_1, self.out_channel_layer_1, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_1),
            nn.ReLU())
        self.pool_1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.conv_2 = nn.Sequential(
            nn.Conv2D(self.in_channel_layer_2, self.out_channel_layer_2, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_2),
            nn.ReLU(),
            nn.Conv2D(self.out_channel_layer_2, self.out_channel_layer_2, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_2),
            nn.ReLU())
        self.pool_2 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.conv_3 = nn.Sequential(
            nn.Conv2D(self.in_channel_layer_3, self.out_channel_layer_3, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_3),
            nn.ReLU(),
            nn.Conv2D(self.out_channel_layer_3, self.out_channel_layer_3, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_3),
            nn.ReLU())
        self.pool_3 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.conv_4 = nn.Sequential(
            nn.Conv2D(self.in_channel_layer_4, self.out_channel_layer_4, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_4),
            nn.ReLU(),
            nn.Conv2D(self.out_channel_layer_4, self.out_channel_layer_4, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_4),
            nn.ReLU())
        self.pool_4 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.conv_5 = nn.Sequential(
            nn.Conv2D(self.in_channel_layer_5, self.out_channel_layer_5, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_5),
            nn.ReLU(),
            nn.Conv2D(self.out_channel_layer_5, self.out_channel_layer_5, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_5),
            nn.ReLU())

        # Decoder
        self.upsample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2D(self.in_channel_layer_up_4, self.out_channel_layer_up_4, kernel_size=1, padding='same'))
        self.conv_up_4 = nn.Sequential(
            nn.Conv2DTranspose(self.out_channel_layer_up_4, self.out_channel_layer_up_4, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_4),
            nn.ReLU(),
            nn.Conv2DTranspose(self.out_channel_layer_up_4, self.out_channel_layer_up_4, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_4),
            nn.ReLU())

        self.upsample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2D(self.in_channel_layer_up_3, self.out_channel_layer_up_3, kernel_size=1, padding='same'))
        self.conv_up_3 = nn.Sequential(
            nn.Conv2DTranspose(self.out_channel_layer_up_3, self.out_channel_layer_up_3, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_3),
            nn.ReLU(),
            nn.Conv2DTranspose(self.out_channel_layer_3, self.out_channel_layer_up_3, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_3),
            nn.ReLU())

        self.upsample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2D(self.in_channel_layer_up_2, self.out_channel_layer_up_2, kernel_size=1, padding='same'))
        self.conv_up_2 = nn.Sequential(
            nn.Conv2DTranspose(self.out_channel_layer_up_2, self.out_channel_layer_up_2, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_2),
            nn.ReLU(),
            nn.Conv2DTranspose(self.out_channel_layer_up_2, self.out_channel_layer_up_2, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_2),
            nn.ReLU())

        self.upsample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2D(self.in_channel_layer_up_1, self.out_channel_layer_up_1, kernel_size=1, padding='same'))
        self.conv_up_1 = nn.Sequential(
            nn.Conv2DTranspose(self.out_channel_layer_up_1, self.out_channel_layer_up_1, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_1),
            nn.ReLU(),
            nn.Conv2DTranspose(self.out_channel_layer_up_1, self.out_channel_layer_up_1, kernel_size=3, padding='same'),
            nn.BatchNorm2D(self.out_channel_layer_up_1),
            nn.ReLU())

        self.output_conv = nn.Conv2D(self.out_channel_layer_up_1, num_classes, kernel_size=1, padding='same')

    def forward(self, inputs):
        # Encoder
        conv_1 = self.conv_1(inputs)
        y = self.pool_1(conv_1)

        conv_2 = self.conv_2(y)
        y = self.pool_2(conv_2)

        conv_3 = self.conv_3(y)
        y = self.pool_3(conv_3)

        conv_4 = self.conv_4(y)
        y = self.pool_4(conv_4)

        y = self.conv_5(y)

        # Decoder
        y = self.upsample_4(y)
        y = paddle.add(y , conv_4)
        y = self.conv_up_4(y)
    
        y = self.upsample_3(y)
        y = paddle.add(y , conv_3)
        y = self.conv_up_3(y)
    
        y = self.upsample_2(y)
        y = paddle.add(y , conv_2)
        y = self.conv_up_2(y)

        y = self.upsample_1(y)
        y = paddle.add(y , conv_1)
        y = self.conv_up_1(y)

        y = self.output_conv(y)
        return y
