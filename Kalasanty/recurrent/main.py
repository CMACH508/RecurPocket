# _date_:2021/8/27 13:25
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
from torch.nn import DataParallel
from torch import nn as nn
import argparse
from os.path import join
import os, sys

# print()
torch.backends.cudnn.bencmark = True

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(101)
BASE_DIR = os.path.abspath('../../')
print(BASE_DIR)
sys.path += [BASE_DIR]
from Kalasanty import dataset
from Kalasanty.recurrent.criterion import Ovl, Dice, DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=70, help='')
parser.add_argument('--test_batch_size', type=int, default=30, help='')
parser.add_argument('--base_lr', type=float, default=1e-5, help='')
parser.add_argument('--lr_adjust', nargs='+', type=int, default=[20000], help='')
parser.add_argument('--print_freq', type=int, default=300, help='')
# parser.add_argument('--gpu', type=str, default='0,1', help='')
parser.add_argument('--gpu', type=str, default='2,3', help='')
parser.add_argument('--save_dir', type=str, default=None, help='')
parser.add_argument('--max_epoch', type=int, default=1500, help='')
parser.add_argument('--iterations', type=int, default=3, help='')
parser.add_argument('--is_mask', type=int, default=0, help='')
parser.add_argument('--DATA_ROOT', type=str, default=None, help='')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if args.save_dir is None:
    args.save_dir = 'checkpoint'

args.save_dir = os.path.abspath(args.save_dir)
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def validate(model, device, val_loader, dice_loss):
    model.eval()
    loss_list = []
    for ite, (protien, label) in enumerate(val_loader, start=1):
        protien, label = protien.to(device), label.to(device)
        with torch.no_grad():
            predy = model(protien)
        tot_loss = dice_loss(y_pred=predy, y_true=label)
        tot_loss = tot_loss.item()
        loss_list.append(tot_loss)
    return np.mean(loss_list)


def main():
    device = torch.device('cuda')
    if args.is_mask:
        from Kalasanty.recurrent.network_mask import Net
        print('------- mask -------')
    else:
        from Kalasanty.recurrent.network import Net
        print('--------- no mask ---------')

    print('args.iterations=', args.iterations)
    print('args.is_mask=', args.is_mask)
    model = Net(iterations=args.iterations).to(device)
    model = DataParallel(model)

    bce = nn.BCELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-3)  # l2_lambda

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_adjust, gamma=0.1)

    dataset.DATA_ROOT = args.DATA_ROOT

    train_dataset = dataset.TrainscPDB(subset='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=10,
                              pin_memory=False)

    val_dataset = dataset.TrainscPDB(subset='validation')
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=True, num_workers=10,
                            pin_memory=False)
    min_val_loss = 100

    resume = False
    if resume:
        checkpoint = torch.load('path of pretrained model')
        model.load_state_dict(checkpoint)
        print('-------- load model successfully --------')

    dice_loss = DiceLoss().to(device)
    dice = Dice().to(device)
    ovl = Ovl().to(device)
    epo = 0
    for epo in range(0, args.max_epoch):
        scheduler.step()
        model.train()
        a_time = datetime.datetime.now()

        for ite, (protien, label) in enumerate(train_loader, start=0):
            # print('protien:', protien.shape)
            # print('label:', label.shape)
            protien, label = protien.to(device), label.to(device)

            predy = model(protien)
            # print(predy.shape)  # [10, 1, 36, 36, 36]

            tot_loss = dice_loss(y_pred=predy, y_true=label)
            optimizer.zero_grad()
            tot_loss.backward()
            tot_loss = tot_loss.item()
            optimizer.step()

            if ite % 100 == 0:
                metric_dice = dice(y_pred=predy, y_true=label).item()
                metric_ovl = ovl(y_pred=predy, y_true=label).item()
                metric_bce = bce(predy, label).item()

                print('epoch : %2d|%2d, iter:%3d|%3d,loss:%.4f,dice:%.3f,ovl:%.3f,bce:%.3f,lr={%.6f}' %
                      (epo, args.max_epoch, ite, len(train_loader), tot_loss, metric_dice, metric_ovl, metric_bce,
                       optimizer.param_groups[0]['lr']))
        b_time = datetime.datetime.now()

        if epo % 20 == 0:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(epo)))

        val_loss = validate(model, device, val_loader, dice_loss)
        print('val_loss={}'.format(val_loss))
        c_time = datetime.datetime.now()
        if min_val_loss > val_loss:
            min_val_loss = val_loss

            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print('------- sota found, epo={} loss={}'.format(epo, val_loss))

        print(get_time(), 'train:{}s val:{}s'.format((b_time - a_time).seconds, (c_time - b_time).seconds))
    torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(epo)))


if __name__ == '__main__':
    print('start time:', get_time())
    main()
    print('end time:', get_time())
