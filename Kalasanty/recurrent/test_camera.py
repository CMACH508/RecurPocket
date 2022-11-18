# _date_:2021/8/29 16:10
import numpy as np
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
from scipy.spatial.distance import cdist
import os, sys
import argparse
from os.path import join

BASE_DIR = os.path.abspath('../')
sys.path += [BASE_DIR, join(BASE_DIR, 'recurrent')]
import dataset

def get_pockets_segmentation(density, threshold=0.5, min_size=50, scale=0.5, max_n = None):
    """Predict pockets using specified threshold on the probability density.
    Filter out pockets smaller than min_size A^3
    """
    if len(density) != 1:
        raise ValueError('segmentation of more than one pocket is not'
                         ' supported')
    voxel_size = (1 / scale) ** 3  # scale? # 8
    # get a general shape, without distinguishing output channels

    bw = closing((density[0] > threshold).any(axis=-1))

    cleared = clear_border(bw)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)

    # for i in range(1, num_labels + 1):
    #     pocket_idx = (label_image == i)
    #     pocket_size = pocket_idx.sum() * voxel_size
    #     if pocket_size < min_size:
    #         label_image[np.where(pocket_idx)] = 0
    # return label_image

    if max_n is None:
        for i in range(1, num_labels + 1):
            pocket_idx = (label_image == i)
            pocket_size = pocket_idx.sum() * voxel_size
            if pocket_size < min_size:
                label_image[np.where(pocket_idx)] = 0
        return label_image
    else:
        size_list = []
        for i in range(1, num_labels + 1):
            pocket_idx = (label_image == i)
            pocket_size = pocket_idx.sum() * voxel_size
            # print(i, pocket_size, min_size)
            if pocket_size >= min_size:
                size_list.append(pocket_size)
        indexs = np.argsort(-np.array(size_list))  # da -- xiao
        indexs = indexs[:max_n] + 1
        label_list = indexs
        new_label_image = np.zeros_like(label_image)
        # print(label_list)
        for ii, lab in enumerate(label_list):
            pocket_idx = (label_image == lab)
            new_label_image[np.where(pocket_idx)] = ii+1
        return new_label_image


def test_model(model, device, data_loader, scale, Threshold_dist):
    model.eval()
    succ = 0
    total = 0
    dvo = 0
    DVO_list = []
    with torch.no_grad():
        for ite, (protien_x, label, centerid, real_num) in enumerate(data_loader, start=1):
            print('Processing {}/{}'.format(ite, len(data_loader)))
            protien_x, label = protien_x.to(device), label.cpu().numpy()  # (bs, 18, 36, 36, 36) # (bs, 36, 36, 36)
            predy_density = model(protien_x)  # (bs, 1, 36, 36, 36)

            predy_density = predy_density.data.cpu().numpy()

            for i, density in enumerate(predy_density):  # (1, 36, 36, 36)
                total += real_num[i]
                density = np.expand_dims(density, 4)  # (1, 36, 36, 36, 1)
                truth_labels = label[i]
                num_cavity = int(truth_labels.max())
                # print('num_cavity={}'.format(num_cavity))
                if num_cavity == 0:
                    # print('--------========xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                    continue

                if is_dca:
                    max_n = real_num[i] + top_n
                else:
                    max_n = None
                predict_labels = get_pockets_segmentation(density, scale=scale, max_n=max_n)


                for target_num in range(1, num_cavity + 1):
                    truth_indices = np.argwhere(truth_labels == target_num).astype('float32')
                    label_center = truth_indices.mean(axis=0)
                    min_dist = 1e6
                    match_label = 0
                    for pocket_label in range(1, predict_labels.max() + 1):
                        indices = np.argwhere(predict_labels == pocket_label).astype('float32')
                        if len(indices) == 0:
                            continue
                        center = indices.mean(axis=0)
                        dist = np.linalg.norm(label_center - center)

                        if dist < min_dist:
                            min_dist = dist
                            match_label = pocket_label

                    if min_dist <= Threshold_dist:
                        succ += 1
                        indices = np.argwhere(predict_labels == match_label).astype('float32')
                        if test_set == 'pdbbind':
                            protien_array = protien_x[i].data.cpu().numpy()  # (18,36,36,36)
                            protien_array = protien_array.transpose((1, 2, 3, 0))  # (36,36,36,18)
                            protein_coord = []
                            for k1 in range(36):
                                for k2 in range(36):
                                    for k3 in range(36):
                                        # print(protien_array[k1, k2, k3].shape)  #(18)
                                        # if not np.all(protien_array[k1, k2, k3]):
                                        if np.any(protien_array[k1, k2, k3]):
                                            protein_coord.append(np.asarray([k1, k2, k3]))
                            protein_coord = np.asarray(protein_coord)
                            ligand_dist = cdist(indices, protein_coord)
                            distance = 3
                            binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
                            indices = protein_coord[binding_indices]
                            ligand_dist = cdist(truth_indices, protein_coord)
                            distance = 1
                            binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
                            truth_indices = protein_coord[binding_indices]

                        indices_set = set([tuple(x) for x in indices])
                        truth_indices_set = set([tuple(x) for x in truth_indices])

                        dvo = len(indices_set & truth_indices_set) / len(indices_set | truth_indices_set)
                        DVO_list.append(dvo)

            if is_dca:
                print('now: succ={} total={} succ/total={}'.format(succ, total, succ / total))
            else:
                print('now: succ={} total={} succ/total={} dvo={} ({})'.format(succ, total, succ / total, np.sum(DVO_list) / total, np.mean(DVO_list)))
        if is_dca:
            res = '| succ={} | total={} | succ/total={}'.format(succ, total, succ / total)
        else:
            res = '| succ={} | total={} | succ/total={} | dvo={} ({})'.format(succ, total, succ / total, np.sum(DVO_list) / total, np.mean(DVO_list))
        return res


parser = argparse.ArgumentParser(description='Test Kalasanty')
parser.add_argument('--model_path', type=str, help="path of Kalasanty checkpoint")
parser.add_argument('--DATA_ROOT', type=str, help="the path of DATA_ROOT")
parser.add_argument('--test_set', type=str, help="the name of test data set")
parser.add_argument('--is_dca', type=int, help="calculate DCA or (DCC and DVO)")
parser.add_argument('-n', '--top_n', type=int, default=0, help="for dca test, top-n if n=0, top-(n+2) if n=2")
parser.add_argument('--gpu', type=str, default='0', help="gpu device")
parser.add_argument('--ite', type=int, default=3, help="iteration")
parser.add_argument('--is_mask', type=int, default=0, help="is_mask")
args = parser.parse_args()

# ICME RecurPocket-kalasanty
dataset.DATA_ROOT = args.DATA_ROOT
model_path = args.model_path
is_dca = args.is_dca
top_n = args.top_n
test_set = args.test_set
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# test_set = 'scpdb'
# test_set = 'pdbbind'
# test_set = 'apo_holo'
# test_set = 'coach420'
# test_set = 'holo4k'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# is_dca = 1
# is_dca = 0

if __name__ == '__main__':
    device = torch.device('cuda')
    _dataset = None

    mask = False
    one_channel = False

    if test_set == 'scpdb':
        _dataset = dataset.TestscPDB(one_channel=False, mask=mask)
    elif test_set == 'pdbbind':
        _dataset = dataset.TestPDBbind(one_channel=False, mask=mask, is_dca=is_dca)
    elif test_set == 'apo_holo':
        _dataset = dataset.TestApoHolo(one_channel=False, mask=mask, is_dca=is_dca)
    elif test_set == 'coach420' or test_set == 'holo4k':
        _dataset = dataset.Test_coach420_holo4k(set=test_set, is_dca=is_dca)

    data_loader = DataLoader(_dataset, batch_size=40, shuffle=False, drop_last=False, num_workers=6)

    if args.is_mask:
        from recurrent.network_mask import Net
    else:
        from recurrent.network import Net
    model = Net(iterations=args.ite).to(device)
    model = DataParallel(model)
    print('Restoring model from path: ' + model_path)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    res = test_model(model, device, data_loader, scale=0.5, Threshold_dist=2)
    print('|', test_set, res)
    print(model_path, '\n')
    print('is_dca=', is_dca)

    # paths.sort()
    # for model_path in paths:
    #     checkpoint = torch.load(model_path)
    #     print('Restoring model from path: ' + model_path)
    #     model.load_state_dict(checkpoint)
    #
    #     res = test_model(model, device, data_loader, scale=0.5, Threshold_dist=2)
    #     print('|', test_set, res)
    #     print(model_path, '\n')
    #


