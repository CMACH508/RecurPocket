# _date_:2021/8/29 16:10
import numpy as np
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
import os, sys
import argparse
from scipy.spatial.distance import cdist
import dataset
from pybel import readfile


def get_pockets_segmentation(density, threshold=0.5, min_size=50, scale=0.5, max_n = None):
    """Predict pockets using specified threshold on the probability density.
    Filter out pockets smaller than min_size A^3
    """
    if len(density) != 1:
        raise ValueError('segmentation of more than one pocket is not'
                         ' supported')
    voxel_size = (1 / scale) ** 3  # scale?
    bw = closing((density[0] > threshold).any(axis=-1))
    cleared = clear_border(bw)
    label_image, num_labels = label(cleared, return_num=True)

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

def _get_binary_features(mol):
    coords = []
    for a in mol.atoms:
        coords.append(a.coords)
    coords = np.array(coords)
    features = np.ones((len(coords), 1))
    return coords, features

def get_label_grids(cavity_paths):
    pocket_number = 0
    pocket_coords_list = []
    cavity_suffix = cavity_paths[0].split('.')[-1]
    for n, cavity_path in enumerate(cavity_paths, start=1):
        print('cavity_path=', cavity_path)
        mol = next(readfile(cavity_suffix, cavity_path))
        pocket_coords, pocket_features = _get_binary_features(mol)
        pocket_coords_list.append(pocket_coords)
        pocket_number += 1
    return pocket_coords_list, pocket_number

def test_model(model, device, data_loader, scale, Threshold_dist, test_set=None, is_dca=0, top_n=0):
    model.eval()
    succ = 0
    total = 0
    dvo = 0
    DVO_list = []
    max_n = None
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
                        if is_dca:
                            dist = 1e6
                            for c in truth_indices:
                                d = np.linalg.norm(center - np.array(c))
                                if d < dist:
                                    dist = d
                        else:
                            dist = np.linalg.norm(label_center - center)
                        if dist < min_dist:
                            min_dist = dist
                            match_label = pocket_label

                    if min_dist <= Threshold_dist:
                        # print(min_dist, Threshold_dist)
                        succ += 1
                        indices = np.argwhere(predict_labels == match_label).astype('float32')

                        if test_set == 'pdbbind':
                            protien_array = protien_x[i].data.cpu().numpy()  # (18,36,36,36)
                            protien_array = protien_array.transpose((1, 2, 3, 0))  # (36,36,36,18)
                            protein_coord = []
                            for k1 in range(36):
                                for k2 in range(36):
                                    for k3 in range(36):
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
                print('now: succ={} total={} succ/total={} dvo={} ({})'.format(succ, total, succ / total, np.sum(DVO_list)/total, np.mean(DVO_list)))
        if is_dca:
            res = '| succ={} | total={} | succ/total={}'.format(succ, total, succ / total)
        else:
            res = '| succ={} | total={} | succ/total={} | dvo={} ({})'.format(succ, total, succ / total, np.sum(DVO_list)/total, np.mean(DVO_list))
        return res

parser = argparse.ArgumentParser(description='Test Kalasanty')
parser.add_argument('--model_path', type=str, help="path of Kalasanty checkpoint")
parser.add_argument('--DATA_ROOT', type=str, help="the path of DATA_ROOT")
parser.add_argument('--test_set', type=str, help="the name of test data set")
parser.add_argument('--is_dca', type=int, help="calculate DCA or (DCC and DVO)")
parser.add_argument('-n', '--top_n', type=int, default='0', help="for dca test, top-n if n=0, top-(n+2) if n=2")
parser.add_argument('--gpu', type=str, default='0', help="gpu device")
args = parser.parse_args()

# ICME kalasanty
dataset.DATA_ROOT = args.DATA_ROOT
model_path = args.model_path
is_dca = args.is_dca
top_n = args.top_n
test_set = args.test_set
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


if __name__ == '__main__':
    from network import Net
    device = torch.device('cuda')
    _dataset = None

    mask = False
    one_channel = False

    if test_set == 'scpdb':
        _dataset = dataset.TestscPDB(one_channel=False, mask=mask)
    elif test_set == 'pdbbind':
        _dataset = dataset.TestPDBbind(one_channel=False, mask=mask, is_dca=is_dca)
    elif test_set == 'apoholo':
        _dataset = dataset.TestApoHolo(one_channel=False, mask=mask, is_dca=is_dca)
    elif test_set == 'coach420' or test_set == 'holo4k':
        _dataset = dataset.Test_coach420_holo4k(set=test_set, is_dca=is_dca)

    data_loader = DataLoader(_dataset, batch_size=40, shuffle=False, drop_last=False, num_workers=6)

    print('Restoring model from path: ' + model_path)
    model = Net(one_channel=one_channel).to(device)
    model = DataParallel(model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    res = test_model(model, device, data_loader, scale=0.5, Threshold_dist=2, test_set=test_set, is_dca=is_dca, top_n=top_n)
    print('T={} | {} {}'.format(2, test_set, res))
    print(model_path)
    print(is_dca)
    print(test_set)

    # file = open('baseline_{}_dcc_dvo.txt'.format(test_set), 'w')
    # file.write(model_path+'\n')
    # for T in range(1, 21):
    #     T = T * 0.5
    #     res = test_model(model, device, data_loader, scale=0.5, Threshold_dist=T)
    #     print('T={} | {} {}'.format(T, test_set, res))
    #     print(model_path)
    #     file.write('T={} | {} {}\n'.format(T, test_set, res))
    # print('------------ Finish -----------')
