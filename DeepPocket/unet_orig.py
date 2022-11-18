import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from os.path import join

def save_density_as_cmap(density, origin, step, fname='pockets.cmap', mode='w', name='protein'):
    """Save predcited pocket density as .cmap file (which can be opened in
    UCSF Chimera or ChimeraX)
    """
    if len(density) != 1:
        raise ValueError('saving more than one prediction at a time is not'
                         ' supported')
    density = density[0].transpose([3, 2, 1, 0])

    with h5py.File(fname, mode) as cmap:
        g1 = cmap.create_group('Chimera')
        for i, channel_dens in enumerate(density):
            g2 = g1.create_group('image%s' % (i + 1))
            g2.attrs['chimera_map_version'] = 1
            g2.attrs['name'] = name.encode() + b' binding sites'
            g2.attrs['origin'] = origin
            g2.attrs['step'] = step
            g2.create_dataset('data_zyx', data=channel_dens,
                              shape=channel_dens.shape,
                              dtype='float32')


def channelPool(input):
    n, c, w, h, z = input.size()
    input = input.view(n, c, w * h * z).permute(0, 2, 1)
    pooled = nn.functional.max_pool1d(
        input,
        c,
    )
    _, _, c = pooled.size()
    pooled = pooled.permute(0, 2, 1)
    return pooled.view(n, c, w, h, z)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(),
                                   nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU())

    def forward(self, x):
        out = self.block(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_pad, stride=2):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(kernel_size_pad, stride=stride), DoubleConv(in_channels, out_channels, 3))

    def forward(self, x):
        out = self.block(x)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_up,padding=0,stride=2, out_pad=0, upsample=None):
        super().__init__()
        if upsample:
            self.up_s = nn.Upsample(scale_factor=2, mode=upsample, align_corners=True)
        else:
            self.up_s = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size_up, stride=stride, padding=padding,
                                           output_padding=out_pad)

        self.convT = DoubleConv(in_channels, out_channels, 3)

    def forward(self, x1, x2):
        out = self.up_s(x1)
        out = self.convT(torch.cat((x2, out), dim=1))
        return out


class Unet(nn.Module):
    def __init__(self, n_classes=1, upsample=None):
        super().__init__()
        self.n_classes = n_classes

        self.in1 = DoubleConv(14, 32, 3)
        self.down1 = Down(32, 64, 3)
        self.down2 = Down(64, 128, 3)
        self.down3 = Down(128, 256, 3)
        factor = 2 if upsample else 1
        self.down4 = Down(256, 512 // factor, 3)
        self.up1 = Up(512, 256 // factor, 3, upsample=upsample,stride=2,out_pad=0)
        self.up2 = Up(256, 128 // factor, 3, upsample=upsample)
        self.up3 = Up(128, 64 // factor, 3, upsample=upsample,out_pad=1)
        self.up4 = Up(64, 32, 3, upsample=upsample)
        self.conv = nn.Conv3d(32, self.n_classes, 1)

    def forward(self, x, name=None):
        x1 = self.in1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x11 = self.up1(x5, x4)
        x22 = self.up2(x11, x3)
        x33 = self.up3(x22, x2)
        x44 = self.up4(x33, x1)
        logits = self.conv(x44)

        # skip_list = [x1, x2, x3, x4]
        # up_list = [x44, x33, x22, x11]
        # save_root = '/cmach-data/lipeiying/program/_Drug_/deep-pocket-ours/visual'
        #
        # visualx = channelPool(x).detach()
        # visualx = (visualx).cpu().numpy()
        # visualx = np.where(visualx > 0, 1, 0).transpose((1, 2, 3, 4, 0))
        # print(visualx.shape)
        # save_density_as_cmap(visualx, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], fname=join(save_root, '{}_input.cmap'.format(name)))
        #
        # for i in range(4):
        #     skip, up = skip_list[i], up_list[i]
        #     visualx = channelPool(skip).detach()
        #     visualx = F.sigmoid(visualx)
        #     visualx = (visualx).cpu().numpy()
        #     print('down-{}'.format(i+1), np.max(visualx))
        #     print('min', np.min(visualx))
        #     print('median', np.median(visualx))
        #     visualx = np.where(visualx > np.median(visualx), 1, 0).transpose((1, 2, 3, 4, 0))
        #     print('ite-{}'.format(i+1), visualx.shape)
        #     save_density_as_cmap(visualx, [0.0, 0.0, 0.0], [65 / visualx.shape[2], 65 / visualx.shape[2], 65 / visualx.shape[2]],
        #                          fname=join(save_root, '{}_{}{}{}_skip.cmap'.format(name, i+1, i+1, i+1)))
        #
        #     visualx = channelPool(up).detach()
        #     visualx = F.sigmoid(visualx)
        #     visualx = (visualx).cpu().numpy()
        #     print('up-{}'.format(i+1), np.max(visualx))
        #     print('min', np.min(visualx))
        #     print('median', np.median(visualx))
        #     visualx = np.where(visualx > np.median(visualx), 1, 0).transpose((1, 2, 3, 4, 0))
        #     print('ite-{}'.format(i+1), visualx.shape)
        #     save_density_as_cmap(visualx, [0.0, 0.0, 0.0], [65 / visualx.shape[2], 65 / visualx.shape[2], 65 / visualx.shape[2]],
        #                          fname=join(save_root, '{}_{}{}{}_up.cmap'.format(name, i+1, i+1, i+1)))
        #
        # visualx = F.sigmoid(logits)
        # visualx = visualx.detach().cpu().numpy()
        # visualx = np.where(visualx > 0.5, 1, 0).transpose((1, 2, 3, 4, 0))
        # print('final_out', visualx.shape)
        # save_density_as_cmap(visualx, [0.0, 0.0, 0.0], [1, 1, 1],
        #                      fname=join(save_root, '{}_final_out.cmap'.format(name)))


        return logits


