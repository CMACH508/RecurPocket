import datetime
from os.path import join
import torch
import numpy as np
import argparse
import os
import molgrid
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label
from scipy.spatial.distance import cdist
from skimage.morphology import binary_dilation
from skimage.morphology import cube
import prody
from pybel import readfile
import glob
import pybel

pybel.ob.obErrorLog.SetOutputLevel(0)
prody.confProDy(verbosity='none')

def preprocess_output(input, threshold):
    input[input >= threshold] = 1
    input[input != 1] = 0
    input = input.numpy()
    bw = closing(input).any(axis=0)
    # remove artifacts connected to border
    cleared = clear_border(bw)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)
    largest = 0
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size > largest:
            largest = pocket_size
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum()
        if pocket_size < largest:
            label_image[np.where(pocket_idx)] = 0
    label_image[label_image > 0] = 1
    return torch.tensor(label_image, dtype=torch.float32)


def get_model_gmaker_eproviders(args):
    # test example provider
    eptest = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, iteration_scheme=molgrid.IterationScheme.LargeEpoch,
                                     default_batch_size=1)
    eptest.populate(args.test_types)
    # gridmaker with defaults
    gmaker_img = molgrid.GridMaker(dimension=32)

    return gmaker_img, eptest


def Output_Coordinates(tensor, center, dimension=16.25, resolution=0.5):
    # get coordinates of mask from predicted mask
    tensor = tensor.numpy()
    indices = np.argwhere(tensor > 0).astype('float32')
    indices *= resolution
    center = np.array([float(center[0]), float(center[1]), float(center[2])])
    indices += center
    indices -= dimension
    return indices

def output_pocket_pdb(pocket_name, prot_prody, pred_AA):
    # output pocket pdb
    # skip if no amino acids predicted
    if len(pred_AA) == 0:
        # print('Return')
        return
    sel_str = 'resindex '
    for i in pred_AA:
        sel_str += str(i) + ' or resindex '
    sel_str = ' '.join(sel_str.split()[:-2])
    pocket = prot_prody.select(sel_str)
    prody.writePDB(pocket_name, pocket)


def _get_binary_features(mol):
    coords = []
    for a in mol.atoms:
        coords.append(a.coords)
    coords = np.array(coords)
    features = np.ones((len(coords), 1))
    return coords, features


def get_label_grids(cavity_paths):
    # cavitys = glob.glob(join(dir_path, cavity_keyword))
    # print(dir_path, self.suffix)
    # print('len_cavity=', len(cavitys), dir_path)
    # label_grids = np.zeros(shape=(1, 36, 36, 36, 1))
    pocket_number = 0
    pocket_coords_list = []

    # print('cavity_paths=', cavity_paths)

    cavity_suffix = cavity_paths[0].split('.')[-1]
    for n, cavity_path in enumerate(cavity_paths, start=1):
        # print('cavity_path=', cavity_path)
        mol = next(readfile(cavity_suffix, cavity_path))
        pocket_coords, pocket_features = _get_binary_features(mol)
        pocket_coords_list.append(pocket_coords)
        pocket_number += 1
    #     x = x * pocket_number
    #     label_grids += x
    #     label_grids = np.where(label_grids > n, n, label_grids)
    #
    # return label_grids.astype(int), pocket_number
    return pocket_coords_list, pocket_number


def get_model_gmaker_eprovider(test_types, batch_size, dims=None):
    eptest_large = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0, balanced=False,
                                           iteration_scheme=molgrid.IterationScheme.LargeEpoch, default_batch_size=batch_size)
    eptest_large.populate(test_types)
    if dims is None:
        gmaker = molgrid.GridMaker()
    else:
        gmaker = molgrid.GridMaker(dimension=dims)
    return gmaker, eptest_large


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def union(lst1, lst2):
    return list(set().union(lst1, lst2))


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def binding_site_AA(ligand, prot_prody, distance):
    # amino acids from ligand distance threshold
    c = ligand.GetConformer()
    ligand_coords = c.GetPositions()

    prot_coords = prot_prody.getCoords()
    ligand_dist = cdist(ligand_coords, prot_coords)
    binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
    # Get protein residue indices involved in binding site
    prot_resin = prot_prody.getResindices()
    prot_binding_indices = prot_resin[binding_indices]
    prot_binding_indices = sorted(list(set(prot_binding_indices)))
    return prot_binding_indices


def predicted_AA(indices, prot_prody, distance):
    # amino acids from mask distance thresholds
    prot_coords = prot_prody.getCoords()
    ligand_dist = cdist(indices, prot_coords)
    binding_indices = np.where(np.any(ligand_dist <= distance, axis=0))
    # get predicted protein residue indices involved in binding site
    prot_resin = prot_prody.getResindices()
    prot_binding_indices = prot_resin[binding_indices]
    prot_binding_indices = sorted(list(set(prot_binding_indices)))
    return prot_binding_indices


def coors2grid(coords, box_size=80):
    grid = np.zeros(shape=(box_size, box_size, box_size))
    center = coords.mean(axis=0)
    coords -= center
    coords += (box_size / 2)
    coords = coords.round().astype(int)
    in_box = ((coords >= 0) & (coords < box_size)).all(axis=1)
    for (x, y, z) in coords[in_box]:
        grid[x, y, z] = 1
    return grid, center


def get_coach420_or_holo4k(set, DATA_ROOT):
    protein_root = join(DATA_ROOT, '{}/protein/'.format(set))
    cavity_root = join(DATA_ROOT, '{}/cavity/'.format(set))
    ligand_root = None
    if set == 'coach420':
        ligand_root = join(DATA_ROOT, '{}/ligand_T2_cavity/'.format(set))
    elif set == 'holo4k':
        ligand_root = join(DATA_ROOT, '{}/ligand/'.format(set))
    exist_id = os.listdir(cavity_root)
    exist_id.sort()
    protein_paths = [join(protein_root, '{}.pdb'.format(id_)) for id_ in exist_id]
    cavity_paths = []
    ligand_paths = []
    for id_ in exist_id:
        # print(id_)
        tmp_cavity_paths = glob.glob(join(cavity_root, id_, '*', 'CAVITY*'))
        tmp_ligand_paths = glob.glob(join(ligand_root, id_, 'ligand*'))
        # assert len(cavities) == len(tmp_cavity_paths)
        cavity_paths.append(tmp_cavity_paths)
        ligand_paths.append(tmp_ligand_paths)
    print(len(protein_paths), len(cavity_paths), len(ligand_paths))
    return protein_paths, cavity_paths, ligand_paths


def test(is_dca, protein_path, label_paths, model, test_loader, gmaker_img, device, dx_name, args):
    label_coords_list, num_cavity = get_label_grids(label_paths)  # (1, 36, 36, 36, 1)

    if args.rank == 0:
        return
    count = 0
    model.eval()

    dims = gmaker_img.grid_dimensions(test_loader.num_types())
    # print(test_loader.num_types(), dims)  # 28 (28, 65, 65, 65)
    tensor_shape = (1,) + dims
    # print('tensor_shape=', tensor_shape)  # (1, 28, 65, 65, 65)
    # create tensor for input, centers and indices
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device, requires_grad=True)
    float_labels = torch.zeros((1, 4), dtype=torch.float32, device=device)
    prot_prody = prody.parsePDB(protein_path)
    # print('=== prot_prody ==', prot_prody.getCoords().shape)
    pred_aa_list = []
    pred_pocket_coords_list = []
    for ii, batch in enumerate(test_loader):
        # print('count=', count)
        # update float_labels with center and index values
        batch.extract_labels(float_labels)
        centers = float_labels[:, 1:]
        # print("float_labels=", float_labels)  # [[15.0000, 39.7082, 14.9926, 23.8298]]
        for b in range(1):
            center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
            # print('11', np.array([float(center[0]), float(center[1]), float(center[2])]))

            # Update input tensor with b'th datapoint of the batch
            gmaker_img.forward(center, batch[b].coord_sets[0], input_tensor[b])
        # Take only the first 14 channels as that is for proteins, other 14 are ligands and will remain 0.
        # print('input shape=', input_tensor[:, :14].shape)  # torch.Size([1, 14, 65, 65, 65])
        with torch.no_grad():
            masks_pred = model(input_tensor[:, :14])
        # print('output shape=', masks_pred.shape)  # torch.Size([1, 1, 65, 65, 65])

        masks_pred = masks_pred.detach().cpu()
        masks_pred = preprocess_output(masks_pred[0], args.threshold)
        # print('processed shape=', masks_pred.shape)  # torch.Size([65, 65, 65])

        # predict binding site residues
        pred_coords = Output_Coordinates(masks_pred, center)  # 预测的cavity坐标，float,小数
        pred_aa = predicted_AA(pred_coords, prot_prody, args.mask_dist)  # protein距离预测cavity比较近的原子的索引
        if not len(pred_aa) == 0:
            pred_pocket_coords_list.append(pred_coords)
            count += 1

        max_n = num_cavity
        # max_n = num_cavity + 2
        if is_dca and count >= max_n:  # DCA
            break

    ''' dcc dvo '''
    # predict_labels = get_pockets_segmentation(density, scale=scale)
    DVO_list = []
    succ = 0

    for k, label_coords in enumerate(label_coords_list):
        min_dist = 1e6
        match_pred_pocket_coords = None
        for j, pred_pocket_coords in enumerate(pred_pocket_coords_list):
            pred_center_coords = pred_pocket_coords.mean(axis=0)  # (3,)
            # print(np.array([pred_center_coords]).shape)  # (1,3)
            # print(label_coords.shape)  # (63, 3)
            # print('--------------')
            # ligand_dist = cdist(np.array([pred_center_coords]), label_coords)[0]
            # # print(ligand_dist.shape)  # (63,)
            # dist = np.min(ligand_dist)

            if is_dca:
                dist = 1e6
                for c in label_coords:
                    d = np.linalg.norm(pred_center_coords - np.array(c))
                    if d < dist:
                        dist = d
            else:
                dist = np.linalg.norm(pred_center_coords - label_coords.mean(axis=0))

            # print('dist=', dist)
            if dist < min_dist:
                min_dist = dist
                match_pred_pocket_coords = pred_pocket_coords

        if is_dca:
            if min_dist <= 4:
                succ += 1
        else:
            if min_dist <= 4:
                succ += 1
                # cavity: coord->numpy, dilation, numpy->
            dvo = 0
            if args.is_dvo:
                if min_dist <= 4:
                    box_size = 80
                    label_grid, label_center = coors2grid(label_coords, box_size=box_size)
                    grid_np = binary_dilation(label_grid, cube(3))
                    grid_indices = np.argwhere(grid_np == 1)
                    label_coords = grid_indices - (box_size / 2)
                    label_coords += label_center

                    pred_coords_set = set([tuple(x) for x in (match_pred_pocket_coords / 2).astype(int)])
                    truth_coords_set = set([tuple(x) for x in (label_coords / 2).astype(int)])
                    dvo = len(pred_coords_set & truth_coords_set) / len(pred_coords_set | truth_coords_set)
            DVO_list.append(dvo)

    # print(DVO_list)
    return succ, num_cavity, DVO_list


def get_acc(seg_model, DATA_ROOT, args, test_set='coach420', is_dca=0):
    protein_paths, cavity_paths, ligand_paths, label_paths = None, None, None, None
    if test_set in ['coach420', 'holo4k']:
        protein_paths, cavity_paths, ligand_paths = get_coach420_or_holo4k(test_set, DATA_ROOT=DATA_ROOT)

    if is_dca:
        label_paths = ligand_paths
    else:
        label_paths = cavity_paths
    save_dir = os.path.join(DATA_ROOT, 'test_types', test_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = 0
    succ = 0
    dvo_list = []
    length = len(protein_paths)

    for i, protein_path in enumerate(protein_paths):
        # if i == 10:
        #     break
        if args.is_debug:
            print('{} [{}/{}] {}'.format(get_time(), i, length, protein_path))
        protein_nowat_file = protein_path.replace('.pdb', '_nowat.pdb')
        protein_name = os.path.basename(protein_path)
        id = protein_name.rsplit('.', 1)[0]
        # # print(id)
        tmp_dir = join(save_dir, id)
        # segmentation
        seg_types = '{}/{}_nowat_out/pockets/bary_centers_ranked.types'.format(tmp_dir, id)
        if not os.path.exists(seg_types):
            print('seg_types not exist path=', seg_types)
            # print('seg_types not exist')
            break

        lines = open(seg_types).readlines()
        if len(lines) == 0:
            total += len(label_paths[i])
            continue

        if args.rank != 0:
            seg_gmaker, seg_eptest = get_model_gmaker_eprovider(seg_types, 1, dims=32)
            dx_name = protein_nowat_file.replace('.pdb', '')

            tmp_succ, num_cavity, DVO_list = test(is_dca, protein_path, label_paths[i], seg_model, seg_eptest, seg_gmaker,
                                                  device, dx_name, args)
            total += num_cavity
            succ += tmp_succ
            dvo_list += DVO_list
            total = max(total, 0.0000001)
            if args.is_debug:
                print('tmp {}: succ={}, total={}, dcc={}/{}={:.4f}, dvo={:.4f}'.format(
                    test_set, succ, total, succ, total, succ / total, np.mean(DVO_list)))
                # print('----------- Finish ------------')

    total = max(total, 0.0000001)
    print('{}: succ={}, total={}, dcc={}/{}={:.4f}, dvo={:.4f}'.format(test_set, succ, total, succ, total, succ / total, np.mean(dvo_list)))
    # print('----------- Finish ------------')
    return succ / total


# print(args.seg_checkpoint)
# print(args.test_set)
if __name__ == '__main__':
    from torch import nn
    import argparse, sys
    sys.path.append(os.path.abspath('../'))

    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('--rank', type=int, help="training types file", default=1)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mask_dist', type=float, default=3.5)
    # todo modify
    parser.add_argument('--is_dvo', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--gpu', type=str, default='2,3')
    parser.add_argument('--is_mask', type=int, default=1)
    parser.add_argument('--is_seblock', type=int, default=0)
    parser.add_argument('--iteration', type=int, default=3)
    parser.add_argument('--is_debug', type=int, default=1)
    parser.add_argument('--is_dca', type=int, default=1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # todo modify
    # from unet import Unet
    # model = Unet(n_classes=1, upsample=False)
    from unet_recurrent import Unet
    model = Unet(in_channels=14, iterations=args.iteration, is_seblock=args.is_seblock, is_mask=args.is_mask)

    model.to(device)
    model = nn.DataParallel(model)

    # TODO modify
    test_set = 'coach420'
    DATA_ROOT = os.path.abspath('./dataset/')
    # ckpt_path = './ckpt/eetsk-deep-i3-coach/seg0-0.88710-70.pth.tar'
    ckpt_path = './ckpt/eetsk-deep-i3-coach-mask/seg0-0.89919-42.pth.tar'


    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    print('load successfully,', ckpt_path)
    a = datetime.datetime.now()

    get_acc(model, DATA_ROOT, args, test_set=test_set, is_dca=args.is_dca)
    b = datetime.datetime.now()
    print(ckpt_path)
    print(args.iteration)
    print('time:', str(b - a))
    print('test_set={}'.format(test_set))
    print('is_dca={}'.format(args.is_dca))
