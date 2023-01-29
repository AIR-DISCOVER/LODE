import os

import numpy as np

import torch
from torch.utils.data import Dataset

import MinkowskiEngine as ME

import math
import yaml
import open3d as o3d
import random
import json
import scipy.linalg
from scipy import spatial
from scipy import ndimage
from PIL import Image

from IPython import embed


SPLIT_SEQUENCES = {
    'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
    'valid': ['08'],
    'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
}
SPLIT_SEQUENCES_SPOSS = {
    "train": ["00", "01", "03", "04", "05"],
    "valid": ["02"],
    "test": ["02"]
}

SPLIT_FILES = {
    'train': ['.bin', '.label', '.invalid'],
    'valid': ['.bin', '.label', '.invalid'],
    'test': ['.bin']
}

EXT_TO_NAME = {'.bin': 'input', '.label': 'label', '.invalid': 'invalid'}

VOXEL_SCALE = [256,256,32]

config_file = os.path.join('semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config['learning_map']


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed


def get_eval_mask(labels, invalid_voxels):
  '''
  Ignore labels set to 255 and invalid voxels (the ones never hit by a laser ray, probed using ray tracing)
  :param labels: input ground truth voxels
  :param invalid_voxels: voxels ignored during evaluation since the lie beyond the scene that was captured by the laser
  :return: boolean mask to subsample the voxels to evaluate
  '''
  masks = np.ones_like(labels, dtype=np.bool)
  masks[labels == 255] = False
  masks[invalid_voxels == 1] = False

  return masks


def data_augmentation(raw, label, invalid, config):
    '''flip & rotate'''
    if config['DATA_IO']['augmentation']:
        angle = config['DATA_IO']['augmentation_angle']
        theta = random.randint(0, angle * 2) - angle
        if config['DATA_IO']['augmentation_flip'] and np.random.rand(1) > 0.5:
            raw = np.flip(raw, axis=1)
            label = np.flip(label, axis=1)
            invalid = np.flip(invalid, axis=1)
    else:
        theta = 0

    raw = ndimage.rotate(raw, theta, reshape=False, order=0, mode='constant', cval=0)
    label = ndimage.rotate(label, theta, reshape=False, order=0, mode='constant', cval=0)
    invalid = ndimage.rotate(invalid, theta, reshape=False, order=0, mode='constant', cval=1)

    # raw points
    raw_pos = raw.nonzero()
    raw_points = np.transpose(raw_pos)

    # label points
    label_pos = label.nonzero()
    label_value = label[label_pos]
    label_points = np.transpose(label_pos)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(label_points[:, :3])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.,0.,1.]))
    occupancy_normal = np.concatenate((np.asarray(pcd.points),np.asarray(pcd.normals)),axis=1)

    return raw_points, occupancy_normal, label, invalid


class DG_Dataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.files = {}
        self.filenames = []

        for ext in SPLIT_FILES[split]:
            self.files[EXT_TO_NAME[ext]] = []

        for sequence in SPLIT_SEQUENCES[split]:
            voxels_path = os.path.join(config['GENERAL']['dataset_dir'], 'sequences', sequence, 'voxels')
            if not os.path.exists(voxels_path): raise RuntimeError('Voxel directory missing: ' + voxels_path)

            files = os.listdir(voxels_path)
            for ext in SPLIT_FILES[split]:
                comletion_data = sorted([os.path.join(voxels_path, f) for f in files if f.endswith(ext)])
                if len(comletion_data) == 0: raise RuntimeError('Missing data for ' + EXT_TO_NAME[ext])
                self.files[EXT_TO_NAME[ext]].extend(comletion_data)

            # filename
            self.filenames.extend(sorted([(sequence, os.path.splitext(f)[0]) for f in files if f.endswith('.bin')]))

        self.num_files = len(self.filenames)
        config['DATA_IO']['file_num'] = self.num_files
        # print('Files num: %d' % self.num_files)

        # sanity check:
        for k, v in self.files.items():
            assert (len(v) == self.num_files)

        remapdict = kitti_config['learning_map']
        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        self.comletion_remap_lut = remap_lut

        complt_num_per_class = np.array(config['DATA_IO']['complt_num_per_class'])
        compl_labelweights = complt_num_per_class / np.sum(complt_num_per_class)
        self.compl_labelweights = np.power(np.amax(compl_labelweights) / compl_labelweights, 1 / 3.0)

    def __len__(self):
        return self.num_files

    def getitem_train(self, idx):
        # input, label, invalid, mask
        raw_data = unpack(np.fromfile(self.files['input'][idx], dtype=np.uint8))
        raw_data = raw_data.reshape(VOXEL_SCALE)

        label = np.fromfile(self.files['label'][idx], dtype=np.uint16)
        label = self.comletion_remap_lut[label]
        label = label.reshape(VOXEL_SCALE)

        invalid = unpack(np.fromfile(self.files['invalid'][idx], dtype=np.uint8))
        invalid = invalid.reshape(VOXEL_SCALE)

        # augmentation
        raw_data_sparse, occupancy_normal, label_volume, invalid = data_augmentation(raw_data, label, invalid, self.config)

        # points
        point_cloud_size = occupancy_normal.shape[0]

        occupancy_sparse = np.zeros([point_cloud_size,3])
        occupancy_sparse = occupancy_normal[:,:3]

        # on&off surface points coordinates and normals
        on_surface_size = self.config['TRAIN']['G_TRAIN']['on_surface_size']
        rand_idcs = np.random.choice(point_cloud_size, size=on_surface_size)

        on_surface_coords = np.zeros([on_surface_size,3])
        on_surface_coords = occupancy_normal[rand_idcs, :3]

        on_surface_labels = label_volume[tuple(np.transpose(on_surface_coords.astype(np.int)))].reshape(-1)

        on_surface_coords += 0.5
        on_surface_coords = on_surface_coords / 256.0
        on_surface_coords -= 0.5
        on_surface_coords *= 2
        
        on_surface_normals = occupancy_normal[rand_idcs, 3:]
        on_surface_occ = np.zeros(on_surface_size)

        off_count_scale = self.config['DATA_IO']['off_count_scale']

        off_surface_size = on_surface_size // off_count_scale
        off_surface_xy = np.random.uniform(-1, 1, size=(off_surface_size, 2))
        off_surface_z  = np.random.uniform(-1, -0.75, size=(off_surface_size, 1))   # -1,-0.75
        off_surface_coords = np.concatenate((off_surface_xy, off_surface_z), axis=1)

        off_surface_normals = np.ones((off_surface_size, 3)) * -1
        if self.config['DATA_IO']['ignore_off_label']:
            off_surface_labels = np.ones(off_surface_size) * 255
        else:
            off_surface_labels = np.ones(off_surface_size) * 0
        off_surface_occ = np.ones(off_surface_size)

        # off_surface vertex
        if self.config['DATA_IO']['use_off_vertex']:
            off_surface_v_size = on_surface_size // off_count_scale

            empty = np.ones_like(label_volume)
            empty[label_volume!=0] = 0
            empty_vertex = np.transpose(empty.nonzero())

            rand_idcs = np.random.choice(empty_vertex.shape[0], size=off_surface_v_size)

            off_surface_v_coords = empty_vertex[rand_idcs, :3]
            off_surface_v_coords = off_surface_v_coords * 1.0 + 0.5
            off_surface_v_coords = off_surface_v_coords / 256.0
            off_surface_v_coords -= 0.5
            off_surface_v_coords *= 2

            off_surface_v_normals = np.ones((off_surface_v_size, 3)) * -1
            off_surface_v_labels = np.zeros(off_surface_v_size)
            off_surface_v_occ = np.ones(off_surface_v_size)
        else:
            off_surface_v_size = 0
            off_surface_v_coords = np.array([]).reshape(-1,3)
            off_surface_v_normals = np.array([]).reshape(-1,3)
            off_surface_v_labels = np.array([]).reshape(-1)
            off_surface_v_occ = np.array([]).reshape(-1)

        # output
        coords = np.concatenate((on_surface_coords, off_surface_coords, off_surface_v_coords), axis=0)
        # coords = np.concatenate((on_surface_coords, off_surface_v_coords), axis=0)
        out_coords = torch.from_numpy(coords).float()

        sdf = np.zeros((on_surface_size + off_surface_size + off_surface_v_size, 1))  # on-surface = 0
        sdf[on_surface_size:, :] = -1  # off-surface = -1
        normals = np.concatenate((on_surface_normals, off_surface_normals, off_surface_v_normals), axis=0)
        labels = np.concatenate((on_surface_labels, off_surface_labels, off_surface_v_labels), axis=0)
        occ = np.concatenate((on_surface_occ, off_surface_occ, off_surface_v_occ), axis=0)

        out_sdf = torch.from_numpy(sdf).float()
        out_normals = torch.from_numpy(normals).float()
        out_labels_point = torch.from_numpy(labels).long()
        out_occ = torch.from_numpy(occ).float()

        out_label = torch.from_numpy(label_volume).long()
        out_invalid = torch.from_numpy(invalid).long()

        out_raw_data = torch.from_numpy(raw_data_sparse).float()

        if self.config['TRAIN']['D_TRAIN']['D_input'] == 'occupancy':
            out_raw_feat = torch.ones((len(out_raw_data), 1))
        elif self.config['TRAIN']['D_TRAIN']['D_input'] == 'radial':
            out_raw_feat = torch.norm(out_raw_data-torch.tensor([0,127.5,0]), p=2, dim=1).reshape(-1,1)
        elif self.config['TRAIN']['D_TRAIN']['D_input'] == 'radial_height':
            out_raw_feat = torch.norm(out_raw_data-torch.tensor([0,127.5,0]), p=2, dim=1).reshape(-1,1)
            z = out_raw_data[:,2].reshape(-1,1)
            out_raw_feat = torch.cat((out_raw_feat, z), dim=1)

        out_occupancy = torch.from_numpy(occupancy_sparse).float()

        return idx, \
               {'coords': out_coords}, \
               {'sdf': out_sdf, 'normals': out_normals, 'label': out_label, 'invalid': out_invalid, 'label_points': out_labels_point, 'occ': out_occ},  \
               {'raw': out_raw_data, 'raw_feat': out_raw_feat, 'occupancy': out_occupancy}

    def getitem_valid(self, idx):
        # input, label, invalid, mask
        raw_data = unpack(np.fromfile(self.files['input'][idx], dtype=np.uint8))
        raw_data = raw_data.reshape(VOXEL_SCALE)
        raw_pos = raw_data.nonzero()
        raw_data_sparse = np.transpose(raw_pos)

        label = np.fromfile(self.files['label'][idx], dtype=np.uint16)
        label = self.comletion_remap_lut[label]
        label = label.reshape(VOXEL_SCALE)

        invalid = unpack(np.fromfile(self.files['invalid'][idx], dtype=np.uint8))
        invalid = invalid.reshape(VOXEL_SCALE)

        mask = get_eval_mask(label, invalid)

        out_raw_data = torch.from_numpy(raw_data_sparse).float()
        if self.config['TRAIN']['D_TRAIN']['D_input'] == 'occupancy':
            out_raw_feat = torch.ones((len(out_raw_data), 1))
        elif self.config['TRAIN']['D_TRAIN']['D_input'] == 'radial':
            out_raw_feat = torch.norm(out_raw_data-torch.tensor([0,127.5,0]), p=2, dim=1).reshape(-1,1)
        elif self.config['TRAIN']['D_TRAIN']['D_input'] == 'radial_height':
            out_raw_feat = torch.norm(out_raw_data-torch.tensor([0,127.5,0]), p=2, dim=1).reshape(-1,1)
            z = out_raw_data[:,2].reshape(-1,1)
            out_raw_feat = torch.cat((out_raw_feat, z), dim=1)

        index_info = self.filenames[idx][0]+'_'+self.filenames[idx][1]
        return index_info, \
               {'raw': out_raw_data, 'raw_feat': out_raw_feat}, \
               {'label': label, 'mask': mask}

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.getitem_train(idx)
        elif self.split == 'valid':
            return self.getitem_valid(idx)


def DG_DataMerge_train(batch):
    out_coords = []
    out_sdf = []
    out_normals = []
    out_labels = []
    out_invalid = []
    out_label_points = []
    out_occ = []
    out_indices = []
    out_raw_data = []
    out_feature = []
    out_occupancy = []

    for num, example in enumerate(batch):
        idx, points, gt, raw_occupancy = example

        out_indices.append(idx)

        out_coords.append(points['coords'])

        out_sdf.append(gt['sdf'])
        out_normals.append(gt['normals'])
        out_labels.append(gt['label'])
        out_invalid.append(gt['invalid'])
        out_label_points.append(gt['label_points'])
        out_occ.append(gt['occ'])

        out_raw_data.append(raw_occupancy['raw'])
        out_feature.append(raw_occupancy['raw_feat'])
        out_occupancy.append(raw_occupancy['occupancy'])

    out_coords  = torch.stack(out_coords)

    out_sdf     = torch.stack(out_sdf)
    out_normals = torch.stack(out_normals)
    out_labels = torch.stack(out_labels)
    out_invalid = torch.stack(out_invalid)
    out_label_points = torch.stack(out_label_points)
    out_occ = torch.stack(out_occ)
    
    indices_out = out_indices
    points_out = {'coords': out_coords}
    gt_output = {'sdf': out_sdf, 'normals': out_normals, 'label': out_labels, 'invalid': out_invalid, 'out_label_points': out_label_points, 'occ': out_occ}

    raw_data_out = ME.utils.batched_coordinates(out_raw_data)
    feature_out = torch.cat(out_feature)
    occupancy_out = ME.utils.batched_coordinates(out_occupancy)

    return indices_out, points_out, gt_output, raw_data_out, feature_out, occupancy_out


def DG_DataMerge_valid(batch):
    out_labels = []
    out_masks = []
    out_indices = []
    out_raw_data = []
    out_feature = []
    out_occupancy = []

    for num, example in enumerate(batch):
        idx, raw_occupancy, eval_info = example

        out_indices.append(idx)

        out_labels.append(eval_info['label'])
        out_masks.append(eval_info['mask'])

        out_raw_data.append(raw_occupancy['raw'])
        out_feature.append(raw_occupancy['raw_feat'])

        out_occupancy.append(torch.zeros([1,3]))  # no use

    indices_out = out_indices
    out_raw = np.stack(out_raw_data, axis=0)
    out_labels = np.stack(out_labels, axis=0)
    out_masks = np.stack(out_masks, axis=0)
    eval_out = {'raw': out_raw,'label': out_labels, 'mask': out_masks}

    raw_data_out = ME.utils.batched_coordinates(out_raw_data)
    feature_out = torch.cat(out_feature)

    occupancy_out = ME.utils.batched_coordinates(out_occupancy)

    return indices_out, eval_out, raw_data_out, feature_out, occupancy_out

