import torch
from torch import nn, cat
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, output_channels=None, iterations=3):
        self.iterations = iterations
        self.alpha = 0.5
        super(Net, self).__init__()
        # keras
        # params = {'kernel_size': 3, 'activation': 'relu',
        #           'padding': 'same', 'kernel_regularizer': l2(l2_lambda)}
        self.conv1_1 = nn.Conv3d(in_channels=18, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                 padding=(1, 1, 1))  # (N,Cin,D,H,W)
        self.conv1_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                 padding=(1, 1, 1))  # (N,Cin,D,H,W)
        # self.bn3d1_1 = nn.BatchNorm3d(32)
        # self.bn3d1_2 = nn.BatchNorm3d(32)
        self.pool1 = nn.AvgPool3d(kernel_size=2)

        self.conv2_1 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.bn3d2_1 = nn.BatchNorm3d(64)
        # self.bn3d2_2 = nn.BatchNorm3d(64)
        self.pool2 = nn.AvgPool3d(kernel_size=2)

        self.conv3_1 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3_2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.bn3d3_1 = nn.BatchNorm3d(128)
        # self.bn3d3_2 = nn.BatchNorm3d(128)
        self.pool3 = nn.AvgPool3d(kernel_size=3)

        self.conv4_1 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4_2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.bn3d4_1 = nn.BatchNorm3d(256)
        # self.bn3d4_2 = nn.BatchNorm3d(256)
        self.pool4 = nn.AvgPool3d(kernel_size=3)

        self.conv5_1 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5_2 = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.bn3d5_1 = nn.BatchNorm3d(512)
        # self.bn3d5_2 = nn.BatchNorm3d(512)
        self.up6 = nn.Upsample(scale_factor=3)
        self.conv6_1 = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv6_2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.up7 = nn.Upsample(scale_factor=3)
        self.conv7_1 = nn.Conv3d(in_channels=256, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv7_2 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8_1 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv8_2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9_1 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv9_2 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.output = nn.Sequential(
            nn.Conv3d(in_channels=32*self.iterations, out_channels=1, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Sigmoid())

    def forward(self, x_ori, is_test=False):
        feedback = [None, None, None, None]
        outputs = []
        # print('input=', x_ori.shape)
        for i in range(self.iterations):
            x = x_ori
            x = F.relu(self.conv1_1(x))
            x = sum([self.alpha * x, (1 - self.alpha) * feedback[0] if feedback[0] is not None
                    else (1 - self.alpha) * x])
            x = F.relu(self.conv1_2(x))
            skip_1 = x
            x = self.pool1(x)
            # print('conv1:', x.shape)

            x = F.relu(self.conv2_1(x))
            x = sum([self.alpha * x, (1 - self.alpha) * feedback[1] if feedback[1] is not None
                    else (1 - self.alpha) * x])
            x = F.relu(self.conv2_2(x))
            skip_2 = x
            x = self.pool2(x)
            # print('conv2:', x.shape)

            x = F.relu(self.conv3_1(x))
            x = sum([self.alpha * x, (1 - self.alpha) * feedback[2] if feedback[2] is not None
                    else (1 - self.alpha) * x])
            x = F.relu(self.conv3_2(x))
            skip_3 = x
            x = self.pool3(x)
            # print('conv3:', x.shape)

            x = F.relu(self.conv4_1(x))
            x = sum([self.alpha * x, (1 - self.alpha) * feedback[3] if feedback[3] is not None
                    else (1 - self.alpha) * x])
            x = F.relu(self.conv4_2(x))
            skip_4 = x
            x = self.pool4(x)
            # print('conv4:', x.shape)

            x = F.relu(self.conv5_1(x))
            x = F.relu(self.conv5_2(x))
            # print('conv5:', x.shape)

            x = self.up6(x)

            # print(x.shape, skip_4.shape)
            #
            x = sum([self.alpha * x, (1 - self.alpha) * cat([skip_4, skip_4], dim=1)])
            x = F.relu(self.conv6_1(x))
            x = F.relu(self.conv6_2(x))
            feedback[-1] = x

            x = self.up7(x)

            # print(x.shape, skip_3.shape)
            x = sum([self.alpha * x, (1 - self.alpha) * cat([skip_3, skip_3], dim=1)])
            x = F.relu(self.conv7_1(x))
            x = F.relu(self.conv7_2(x))
            feedback[-2] = x

            # print('1', x.shape)  # [10, 128, 9, 9, 9]
            x = self.up8(x)

            # print(x.shape, skip_2.shape)
            x = sum([self.alpha * x, (1 - self.alpha) * cat([skip_2, skip_2], dim=1)])
            x = F.relu(self.conv8_1(x))
            x = F.relu(self.conv8_2(x))
            feedback[-3] = x

            # print('2', x.shape) # [10, 64, 18, 18, 18]
            x = self.up9(x)

            # print(x.shape, skip_1.shape)
            x = sum([self.alpha * x, (1 - self.alpha) * cat([skip_1, skip_1], dim=1)])
            x = F.relu(self.conv9_1(x))
            x = F.relu(self.conv9_2(x))
            feedback[-4] = x
            outputs.append(x)

            # print('3', x.shape) # [10, 32, 36, 36, 36]
        x = self.output(cat(outputs, dim=1))  # [0, 1]

        # print('out=', x.shape)
        if is_test:
            return x, outputs

        return x


if __name__ == '__main__':
    data = torch.randn(size=(1, 18, 36, 36, 36))
    model = Net(iterations=1)
    out = model(data)
    print('out.shape=', out.shape)