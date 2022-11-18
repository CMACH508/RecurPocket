# _date_:2021/8/27 12:18
from torch.utils.data import Dataset
import h5py
from random import shuffle, choice, sample
import numpy as np
from scipy import ndimage
import tfbio
import tfbio.data
from skimage.draw import ellipsoid
import glob
from os.path import join
from pybel import readfile
import os
from tfbio.data import Featurizer

DATA_ROOT = ''

class BaseTrainSet(Dataset):
    def __init__(self):
        super(BaseTrainSet, self).__init__()
        self.max_dist = 35
        self.scale = 0.5
        self.footprint = None
        self.max_translation = 5
        self.transform = True
        hdf_path = join(DATA_ROOT, 'scPDB/scpdb_dataset.hdf')
        self.data_handle = h5py.File(hdf_path, mode='r')
        footprint = ellipsoid(2, 2, 2)
        self.footprint = footprint.reshape((1, *footprint.shape, 1))
        pdbids = list(self.data_handle.keys())
        self.x_channels = self.data_handle[pdbids[0]]['features'].shape[1]
        self.y_channels = self.data_handle[pdbids[0]]['pocket_features'].shape[1]


class TrainscPDB(BaseTrainSet):
    def __init__(self, subset, one_channel=False):
        super(TrainscPDB, self).__init__()
        if subset == 'train':
            with open(join(DATA_ROOT, 'scPDB/ten_folds/train_ids_fold0')) as f:
                lines = f.readlines()
            self.pdbids = [line.strip() for line in lines]
        elif subset == 'validation':
            self.transform = False
            with open(join(DATA_ROOT, 'scPDB/ten_folds/test_ids_fold0')) as f:
                lines = f.readlines()
            self.pdbids = [line.strip() for line in lines]

        self.one_channel = one_channel
        print('dataset_len=', len(self.pdbids))

    def __getitem__(self, index):  # sample_generator
        if index == 0:
            shuffle(self.pdbids)
        pdbid = self.pdbids[index]

        if self.transform:
            rot = choice(range(24))
            tr = self.max_translation * np.random.rand(1, 3)
        else:
            rot = 0
            tr = (0, 0, 0)
        r, p = self.prepare_complex(pdbid, rotation=rot, translation=tr)  # (1, 36, 36, 36, 18) (1, 36, 36, 36, 1)
        r, p = np.squeeze(r, 0), np.squeeze(p, 0)  # (36, 36, 36, 18) (36, 36, 36, 1)
        r, p = r.transpose((3, 0, 1, 2)), p.transpose((3, 0, 1, 2))
        return r, p

    def __len__(self):
        return len(self.pdbids)

    def prepare_complex(self, pdbid, rotation=0, translation=(0, 0, 0), vmin=0, vmax=1):
        """Prepare complex with given pdbid.

        Parameters
        ----------
        pdbid: str
            ID of a complex to prepare
        rotation: int or np.ndarray (shape (3, 3)), optional (default=0)
            Rotation to apply. It can be either rotation matrix or ID of
            rotatation defined in `tfbio.data` (0-23)
        translation: tuple of 3 floats, optional (default=(0, 0, 0))
            Translation to apply
        vmin, vmax: floats, optional (default 0 and 1)
            Clip values generated for pocket to this range

        Returns
        -------
        rec_grid: np.ndarray
            Grid representing protein
        pocket_dens: np.ndarray
            Grid representing pocket
        """

        resolution = 1. / self.scale
        structure = self.data_handle[pdbid]
        rec_coords = tfbio.data.rotate(structure['coords'][:], rotation)
        rec_coords += translation
        if self.one_channel:
            features = np.ones((len(rec_coords), 1))
        else:
            features = structure['features'][:]
        rec_grid = tfbio.data.make_grid(rec_coords, features,
                                        max_dist=self.max_dist,
                                        grid_resolution=resolution)

        pocket_coords = tfbio.data.rotate(structure['pocket_coords'][:], rotation)
        pocket_coords += translation

        pocket_dens = tfbio.data.make_grid(pocket_coords,
                                           structure['pocket_features'][:],
                                           max_dist=self.max_dist)
        margin = ndimage.maximum_filter(pocket_dens, footprint=self.footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(vmin, vmax)

        zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i], zoom) for i in range(self.y_channels)], -1)
        pocket_dens = np.expand_dims(pocket_dens, 0)  # (1, 36, 36, 36, 1)
        pocket_dens = pocket_dens.clip(vmin, vmax)
        return rec_grid, pocket_dens


class BaseTestSet(Dataset):
    def __init__(self, scale=0.5, max_dist=35):
        super(BaseTestSet, self).__init__()
        self.scale = scale
        self.max_dist = max_dist
        self.featurizer = Featurizer(save_molecule_codes=False)
        footprint = ellipsoid(2, 2, 2)
        self.footprint = footprint.reshape((1, *footprint.shape, 1))
        self.y_channels = 1

        self.latent_data_handle = None
        self.resolution = 1. / self.scale
        self.centroid = None

    def get_prior_pocket(self, pdbid):
        ''' latent '''
        structure_latent = self.latent_data_handle[pdbid]
        # self.origin, self.step = structure_latent['origin'], structure_latent['step']
        latent_coords = np.array(structure_latent['index'])
        coords = latent_coords * 2 - 35
        features = np.ones((len(coords), 1))
        latent_grid = tfbio.data.make_grid(coords, features,
                                           max_dist=self.max_dist,
                                           grid_resolution=self.resolution)  # (1, 36, 36, 36, 1)
        return latent_grid

    def get_mol(self, path):
        suffix = path.split('.')[-1]
        mol = next(readfile(suffix, path))
        coords = np.array([a.coords for a in mol.atoms])
        centroid = coords.mean(axis=0)
        coords -= centroid
        return centroid, mol

    def density_form_mol(self, mol, one_channel):
        prot_coords, prot_features = self.featurizer.get_features(mol)
        self.centroid = prot_coords.mean(axis=0)
        prot_coords -= self.centroid
        resolution = 1. / self.scale
        if one_channel:
            features = np.ones((len(prot_coords), 1))
        else:
            features = prot_features
        x = tfbio.data.make_grid(prot_coords, features,
                                 max_dist=self.max_dist,
                                 grid_resolution=resolution)  # (1, 36, 36, 36, 18)
        x = np.squeeze(x, 0)
        x = x.transpose((3, 0, 1, 2))  # (18, 36, 36, 36)

        return x

    def density_form_mol_01(self, mol):
        prot_coords, prot_features = self._get_binary_features(mol)
        # print(len(prot_coords), prot_coords[:3])
        centroid = prot_coords.mean(axis=0)
        prot_coords -= centroid
        resolution = 1. / self.scale
        x = tfbio.data.make_grid(prot_coords, prot_features,
                                 max_dist=self.max_dist,
                                 grid_resolution=resolution)  # (1, 36, 36, 36, 1)
        x = np.squeeze(x)  # (36, 36, 36)
        origin = (centroid - self.max_dist)
        step = np.array([1.0 / self.scale] * 3)
        return x, origin, step

    def get_label_grids(self, cavity_paths, protein_centroid):
        label_grids = np.zeros(shape=(1, 36, 36, 36, 1))
        pocket_number = 0
        cavity_suffix = cavity_paths[0].split('.')[-1]
        for n, cavity_path in enumerate(cavity_paths, start=1):
            if not os.path.exists(cavity_path):
                continue
            mol = next(readfile(cavity_suffix, cavity_path))
            pocket_coords, pocket_features = self._get_binary_features(mol)
            pocket_coords -= protein_centroid
            resolution = 1. / self.scale
            x = tfbio.data.make_grid(pocket_coords, pocket_features,
                                     max_dist=self.max_dist,
                                     grid_resolution=resolution)  # (1, 36, 36, 36, 1)
            x = np.where(x > 0, 1, 0)
            if not (x == 1).any():  # pocket超出边界
                # print('###################################################')
                continue
            pocket_number += 1
            x = x * pocket_number
            label_grids += x
            label_grids = np.where(label_grids > pocket_number, pocket_number, label_grids).astype(int) # 含0，1，2，3...表示的矩阵

        old_pocket_number = pocket_number
        for p in range(1, pocket_number + 1):
            while not (p in label_grids) and p <= np.max(label_grids):
                label_grids = np.where(label_grids > p, label_grids - 1, label_grids)
                pocket_number -= 1
        # print('cavity_path=', cavity_paths, 'now=', pocket_number, 'old=', old_pocket_number)
        return label_grids.astype(int), pocket_number

    def get_label_01(self, cavity_paths, protein_centroid, vmin=0, vmax=1, size=36):
        cavity_suffix = cavity_paths[0].split('.')[-1]
        pocket_coords, pocket_features = None, None
        for n, cavity_path in enumerate(cavity_paths, start=1):
            mol = next(readfile(cavity_suffix, cavity_path))
            tmp_pocket_coords, tmp_pocket_features = self._get_binary_features(mol)  # (point_num, 3)(point_num, 1)
            tmp_pocket_coords -= protein_centroid

            if n == 1:
                pocket_coords = tmp_pocket_coords
                pocket_features = tmp_pocket_features
            else:
                pocket_coords = np.concatenate((pocket_coords, tmp_pocket_coords))
                pocket_features = np.concatenate((pocket_features, tmp_pocket_features))

        # resolution = 1. / self.scale
        pocket_dens = tfbio.data.make_grid(pocket_coords, pocket_features, max_dist=self.max_dist)
        margin = ndimage.maximum_filter(pocket_dens, footprint=self.footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(vmin, vmax)
        zoom = size / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i], zoom) for i in range(self.y_channels)], -1)
        pocket_dens = pocket_dens.transpose((3, 0, 1, 2))
        return pocket_dens

    def _get_binary_features(self, mol):
        coords = []
        for a in mol.atoms:
            coords.append(a.coords)
        coords = np.array(coords)
        features = np.ones((len(coords), 1))
        return coords, features

    def get_mask(self, pdbid):
        structure_latent = self.latent_data_handle[pdbid]
        self.origin, self.step = structure_latent['origin'], structure_latent['step']
        latent_coords = np.array(structure_latent['index'])
        coords = latent_coords * 2 - 35
        features = np.ones((len(coords), 1))
        latent_grid = tfbio.data.make_grid(coords, features,
                                           max_dist=self.max_dist,
                                           grid_resolution=self.resolution)  # (1, 36, 36, 36, 1)
        latent_grid = np.squeeze(latent_grid, 0)  # (36, 36, 36, 1)
        latent_grid = latent_grid.transpose((3, 0, 1, 2))  # (1, 36, 36, 36)
        return latent_grid


class TestscPDB(BaseTestSet):
    def __init__(self, one_channel, mask):
        super(TestscPDB, self).__init__()  # cavity6_1.mol2
        # ten folds
        txt_paths = glob.glob(join(DATA_ROOT, 'scPDB/ten_folds/test_ids_fold0'))
        data_dir = join(DATA_ROOT, 'scPDB/scPDB_protein_ligands/')
        cavity_match_word = 'cavity*'
        self.data_paths = []
        self.cavity_paths = []
        for path in txt_paths:
            lines = open(path).readlines()
            self.data_paths += [join(data_dir, line.strip(), 'protein.mol2') for line in lines]
            tmp_cavity_paths = glob.glob(join(data_dir, line.strip(), cavity_match_word) for line in lines)
            self.cavity_paths.append(tmp_cavity_paths)

        print('all_data=', len(self.data_paths))
        # print(self.data_paths[0])
        self.mask = mask
        if self.mask:
            latent_hdf_path = join(DATA_ROOT, 'scPDB/scpdb_latent_pockets_v2.hdf')
            self.latent_data_handle = h5py.File(latent_hdf_path, mode='r')
        self.one_channel = one_channel

    def __getitem__(self, index):
        path = self.data_paths[index]
        centroid, mol = self.get_mol(path)
        protein_x = self.density_form_mol(mol, self.one_channel)  # # (18, 36, 36, 36)
        truth_labels, num_cavity = self.get_label_grids(self.cavity_paths[index], centroid)  # (1, 36, 36, 36, 1)
        truth_labels = np.squeeze(truth_labels)  # # (36, 36, 36)
        # label_01 = self.get_label_01(os.path.dirname(path), centroid)  # (1, 36, 36, 36)
        if self.mask:
            pdbid = path.split('/')[-2]
            prior = self.get_prior_pocket(pdbid)  # (1, 36, 36, 36, 1)
            prior = np.squeeze(prior, 0)  # (36, 36, 36, 1)
            prior_pocket = prior.transpose((3, 0, 1, 2))  # (1, 36, 36, 36)
            return protein_x, truth_labels, prior_pocket

        print('\n{}'.format(path))
        print(protein_x.dtype, truth_labels.dtype)
        return protein_x, truth_labels, self.centroid

    def __len__(self):
        return len(self.data_paths)


class TestPDBbind(BaseTestSet):
    ''' ##### icme camera-ready '''
    def __init__(self, one_channel, mask, is_dca):
        super(TestPDBbind, self).__init__()
        self.is_dca = is_dca
        self.one_channel = one_channel
        self.mask = mask
        if mask:
            mask_path = join(DATA_ROOT, 'PDBbind_v2020_refined/pdbbind_latent_pockets_v2.hdf')
            self.latent_data_handle = h5py.File(mask_path, mode='r')

        path = join(DATA_ROOT, 'PDBbind_v2020_refined/refined-set-no-solvent/*/*_protein.pdb')
        # black_list: Too large to load
        black_list = ['3t0b_protein.pdb', '3t09_protein.pdb', '3mv0_protein.pdb', '3dyo_protein.pdb', '3vd4_protein.pdb',
                      '3vdb_protein.pdb', '3f34_protein.pdb', '3i3b_protein.pdb', '3k1j_protein.pdb', '3f37_protein.pdb',
                      '3f33_protein.pdb', '3t08_protein.pdb', '3vd9_protein.pdb', '3t0d_protein.pdb', '3muz_protein.pdb',
                      '3t2q_protein.pdb', '2f2h_protein.pdb', '1px4_protein.pdb']
        # repeat_list: appear in training set
        repeat_list = open(join(DATA_ROOT, 'PDBbind_v2020_refined/repeat_list_1405.txt')).readlines()
        repeat_list = [name.strip() for name in repeat_list]
        total_list = black_list + repeat_list

        protein_paths = glob.glob(path)
        protein_paths.sort()

        self.protein_paths = [p for p in protein_paths if os.path.basename(p) not in total_list]
        self.cavity_paths = [[path.replace('protein', 'pocket')] for path in self.protein_paths]
        self.ligand_paths = [[path.replace('protein.pdb', 'ligand.mol2')] for path in self.protein_paths]

        self.protein_format = 'pdb'
        self.protein_paths.sort()
        print('all_data=', len(self.protein_paths), len(self.cavity_paths), len(self.ligand_paths))

    def __getitem__(self, index):
        path = self.protein_paths[index]
        # print(path)
        centroid, mol = self.get_mol(path)
        protein_x = self.density_form_mol(mol, self.one_channel)  # # (18, 36, 36, 36)

        if self.is_dca:
            truth_labels, num_cavity = self.get_label_grids(self.ligand_paths[index], centroid)  # (1, 36, 36, 36, 1)
            label_num = len(self.ligand_paths[index])
        else:
            truth_labels, num_cavity = self.get_label_grids(self.cavity_paths[index], centroid)  # (1, 36, 36, 36, 1)
            label_num = len(self.cavity_paths[index])
        # print('num_cavity=', num_cavity)
        truth_labels = np.squeeze(truth_labels)  # # (36, 36, 36)
        if self.mask:
            # TODO add mask
            pdbid = os.path.basename(path).split('.')[0]
            latent_grid = self.get_mask(pdbid)  # (1, 36, 36, 36)
            return protein_x, truth_labels, latent_grid
        return protein_x, truth_labels, self.centroid, label_num

    def __len__(self):
        return len(self.protein_paths)


class Test_coach420_holo4k(BaseTestSet):
    def __init__(self, set, is_dca):
        super(Test_coach420_holo4k, self).__init__()
        # set = 'coach420' or 'holo4k'
        protein_root = join(DATA_ROOT, '{}/protein/'.format(set))
        cavity_root = join(DATA_ROOT, '{}/cavity/'.format(set))
        ligand_root = None
        if set == 'coach420':
            ligand_root = join(DATA_ROOT, '{}/ligand_T2_cavity/'.format(set))

        exist_id = os.listdir(cavity_root)
        exist_id.sort()
        self.protein_paths = [join(protein_root, '{}.pdb'.format(id_)) for id_ in exist_id]

        self.cavity_paths = []
        self.ligand_paths = []
        for id_ in exist_id:
            # print(id_)
            tmp_cavity_paths = glob.glob(join(cavity_root, id_, '*', 'CAVITY*'))
            if set == 'coach420':
                tmp_ligand_paths = glob.glob(join(ligand_root, id_, 'ligand*'))
            elif set == 'holo4k':
                tmp_ligand_paths = glob.glob(join(cavity_root, id_, '*', 'ligand*'))
            # print(tmp_cavity_paths)
            self.cavity_paths.append(tmp_cavity_paths)
            self.ligand_paths.append(tmp_ligand_paths)
        print(len(self.protein_paths), len(self.cavity_paths), len(self.ligand_paths))

        self.one_channel = False

        if is_dca:
            self.label_paths = self.ligand_paths
        else:
            self.label_paths = self.cavity_paths

    def __getitem__(self, index):
        protein_path = self.protein_paths[index]
        centroid, mol = self.get_mol(protein_path)
        protein_x = self.density_form_mol(mol, self.one_channel)  # # (18, 36, 36, 36)
        truth_labels, num_cavity = self.get_label_grids(self.label_paths[index], centroid)  # (1, 36, 36, 36, 1)
        truth_labels = np.squeeze(truth_labels)  # # (36, 36, 36)
        return protein_x, truth_labels, centroid, len(self.cavity_paths[index])

    def __len__(self):
        return len(self.protein_paths)


class TestApoHolo(BaseTestSet):
    def __init__(self, one_channel, mask, is_dca):
        self.is_dca =is_dca
        super(TestApoHolo, self).__init__()
        ApoHolo = join(DATA_ROOT, 'ApoHolo')
        protein_paths = []
        cavity_paths = []
        ligand_paths = []
        key = ['*unbound.pdb', '*protein.pdb']
        for ii, set in enumerate(['apo', 'holo']):
            dirs = os.listdir(join(ApoHolo, set))
            dirs.sort()
            for jj, dd in enumerate(dirs):
                d = join(ApoHolo, set, dd)
                prot = glob.glob(join(d, key[ii]))[0]
                ligand_dirs = glob.glob(join(d, 'volsite/ligand*'))
                tmp_ligand_path, tmp_cavity_path = [], []
                for ligand_d in ligand_dirs:
                    ligand_dir = join(d, 'volsite', ligand_d)
                    files = os.listdir(ligand_dir)
                    if len(files) < 5:
                        continue
                    tmp_ligand_path.append(glob.glob(join(ligand_dir, '*ligand*'))[0])
                    tmp_cavity_path.append(glob.glob(join(ligand_dir, 'CAVITY_N1_ALL.mol2'))[0])
                if len(tmp_ligand_path) > 0:
                    protein_paths.append(prot)
                    ligand_paths.append(tmp_ligand_path)
                    cavity_paths.append(tmp_cavity_path)
        self.protein_paths = protein_paths
        self.cavity_paths = cavity_paths
        self.ligand_paths = ligand_paths
        self.mask = mask
        print(len(self.protein_paths), len(self.cavity_paths), len(self.ligand_paths), 'is_dca={}'.format(is_dca))
        self.one_channel = one_channel

    def __getitem__(self, index):
        path = self.protein_paths[index]
        centroid, mol = self.get_mol(path)
        protein_x = self.density_form_mol(mol, self.one_channel)  # # (18, 36, 36, 36)
        if self.is_dca:
            truth_labels, num_cavity = self.get_label_grids(self.ligand_paths[index], centroid)  # (1, 36, 36, 36, 1)
            real_num = len(self.ligand_paths[index])
        else:
            truth_labels, num_cavity = self.get_label_grids(self.cavity_paths[index], centroid)  # (1, 36, 36, 36, 1)
            real_num = len(self.cavity_paths[index])
        truth_labels = np.squeeze(truth_labels)  # # (36, 36, 36)
        if self.mask:
            pdbid = os.path.basename(path).split('.')[0]
            latent_grid = self.get_mask(pdbid)  # (1, 36, 36, 36)
            return protein_x, truth_labels, latent_grid
        return protein_x, truth_labels, self.centroid, real_num

    def __len__(self):
        return len(self.protein_paths)

