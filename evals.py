
import os
import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
import plyfile
import skimage.measure
import yaml
import time
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME

import dataio

from IPython import embed


config_file = os.path.join('semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))

inv_map = kitti_config['learning_map_inv']
maxkey = max(inv_map.keys())
inv_map_lut = np.zeros((maxkey + 100), dtype=np.int32)
inv_map_lut[list(inv_map.keys())] = list(inv_map.values())

color_map = kitti_config['color_map']
maxkey = max(color_map.keys())
color_map_lut = np.zeros((maxkey + 100, 3), dtype=np.int32)
color_map_lut[list(color_map.keys())] = list(color_map.values())


class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    # print('[IOU EVAL] IGNORE: ', self.ignore)
    # print('[IOU EVAL] INCLUDE: ', self.include)

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes,
                                 self.n_classes),
                                dtype=np.int64)

  def addBatch(self, x, y):  # x=preds, y=targets
    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    # check
    assert(x_row.shape == x_row.shape)

    # create indexes
    idxs = tuple(np.stack((x_row, y_row), axis=0))

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.conf_matrix, idxs, 1)

  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.copy()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns 'iou mean', 'iou per class' ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns 'acc mean'
    
  def get_confusion(self):
    return self.conf_matrix.copy()


def eval_cd(pred, gt, mask):
    '''pred gt mask are all of same size'''
    pred[mask == False] = 0
    gt[mask == False] = 0

    pred_xyz = np.transpose(pred.nonzero())
    gt_xyz = np.transpose(gt.nonzero())

    if pred_xyz.shape[0] == 0 or gt_xyz.shape[0] == 0:
        if pred_xyz.shape[0] == 0 and gt_xyz.shape[0] == 0:
            return 0
        print('chamfer distance infinite!')
        return 100000 # infinite

    cd1 = 0
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(gt_xyz)
    dist, indexes = neigh.kneighbors(pred_xyz, return_distance=True)
    cd1 = dist.mean()

    cd2 = 0
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(pred_xyz)
    dist, indexes = neigh.kneighbors(gt_xyz, return_distance=True)
    cd2 = dist.mean()

    return (cd1 + cd2) * 0.2


def get_discrete_sdf(config, decoder, shape_embedding, N=256, max_batch=64 ** 3):
    '''get discrete sdf from decoder'''
    N = int(N)
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / N

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    samples[:, :3] += 0.5
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples = samples.reshape(N,N,N,4)[:,:,:(N//8),:].reshape(-1,4)

    samples = samples.unsqueeze(0).cuda()

    # trilinear #
    shape_embedding = shape_embedding.unsqueeze(0).transpose(2,4)
    scaled_coords = samples.clone().detach()
    scaled_coords[:,:,2] = ((scaled_coords[:,:,2] + 1.) / 0.25 - 0.5) * 2.  # coords z located in [-1., -0.75], scale to [-1,1]
    if config['TRAIN']['shape_sample_strategy'] == 'trilinear':
        shapes = F.grid_sample(shape_embedding, scaled_coords[:,:,:3].unsqueeze(2).unsqueeze(3), \
                    mode='bilinear', padding_mode='border', align_corners=False)
    else:
        shapes = F.grid_sample(shape_embedding, scaled_coords[:,:,:3].unsqueeze(2).unsqueeze(3), \
                    mode='nearest', padding_mode='border', align_corners=False)
    shapes = shapes.squeeze(-1).squeeze(-1).transpose(1,2).cuda()   # batch_size * point_num * shape_embedding_size


    num_samples = N*N*(N//8)

    samples.requires_grad = False
    head = 0
    while head < num_samples:
        shapes_input = shapes[:, head : min(head + max_batch, num_samples), :].cuda()
        coords_input = samples[:, head : min(head + max_batch, num_samples), :3].cuda()

        samples[0, head : min(head + max_batch, num_samples), 3] = (
            decoder(shapes_input, coords_input)['model_out']
            .squeeze().detach().cpu()
        )

        head += max_batch
    
    sdf_values = samples[:, :, 3].cpu().numpy().reshape(N,N,N//8)

    return sdf_values


def get_discrete_sdf_label_dec(config, G_siren, G_label, shape_embedding, N=256, max_batch=64 ** 3):
    '''get discrete sdf from decoder'''
    N = int(N)
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / N

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    samples[:, :3] += 0.5
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples = samples.reshape(N,N,N,3)[:,:,:(N//8),:].reshape(-1,3)

    samples = samples.unsqueeze(0).cuda()

    # trilinear #
    shape_embedding = shape_embedding.unsqueeze(0).transpose(2,4)
    scaled_coords = samples.clone().detach()
    scaled_coords[:,:,2] = ((scaled_coords[:,:,2] + 1.) / 0.25 - 0.5) * 2.  # coords z located in [-1., -0.75], scale to [-1,1]
    if config['TRAIN']['shape_sample_strategy'] == 'trilinear':
        shapes = F.grid_sample(shape_embedding, scaled_coords[:,:,:3].unsqueeze(2).unsqueeze(3), \
                    mode='bilinear', padding_mode='border', align_corners=False)
    else:
        shapes = F.grid_sample(shape_embedding, scaled_coords[:,:,:3].unsqueeze(2).unsqueeze(3), \
                    mode='nearest', padding_mode='border', align_corners=False)
    shapes = shapes.squeeze(-1).squeeze(-1).transpose(1,2).cuda()   # batch_size * point_num * shape_embedding_size

    result_sdf = torch.zeros(N*N*(N//8), 1)
    result_label = torch.zeros(N*N*(N//8), 20)

    num_samples = N*N*(N//8)
    samples.requires_grad = False
    head = 0
    while head < num_samples:
        shapes_input = shapes[:, head : min(head + max_batch, num_samples), :].cuda()
        coords_input = samples[:, head : min(head + max_batch, num_samples), :3].cuda()

        sdf_out = G_siren(shapes_input, coords_input)['model_out']
        label_out = G_label(shapes_input, coords_input)['model_out']

        result_sdf[head : min(head + max_batch, num_samples), 0] = (sdf_out.squeeze().detach().cpu())
        result_label[head : min(head + max_batch, num_samples), :] = (label_out.squeeze().detach().cpu())

        head += max_batch
    
    result_sdf = result_sdf.cpu().numpy().reshape(N,N,N//8)
    result_label = result_label.cpu().numpy().reshape(N,N,N//8,20)

    return result_sdf, result_label


def scene_save_sc(model, shape_embedding, raw, label, mask, config, model_dir, index):
    # ratio = config['EVAL']['mesh']['ratio']
    ratio = 1.0

    if config['EVAL']['save_predict_point'] or config['EVAL']['mesh']['create_mesh']:
        save_dir = os.path.join(model_dir, str(index))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    sdf_values = get_discrete_sdf(config, model, shape_embedding, N=256*ratio)

    iou_out = np.zeros_like(config['EVAL']['eval_threshold'])
    cd_out = np.zeros_like(config['EVAL']['eval_threshold'])

    if config['EVAL']['mesh']['create_mesh']:
        dest_ply_path = os.path.join(save_dir, str(index)+'.ply')

        mesh_level = config['EVAL']['mesh']['mesh_level']

        voxel_size = 1.0 / ratio
        voxel_origin = [0, 0, 0]

        sdf_values = abs(sdf_values)    ###

        convert_sdf_samples_to_ply(
            sdf_values,
            voxel_origin,
            voxel_size,
            dest_ply_path,
            None,
            mesh_level,
        )

    return iou_out, cd_out


def scene_save_ssc_a(model, shape_embedding, class_out, raw, label, mask, config, model_dir, index):
    save_dir = os.path.join(model_dir, str(index))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    SCALE = [256,256,32]

    NUM_CLASS_COMPLET = 20
    complet_evaluator = iouEval(NUM_CLASS_COMPLET, [])

    label_iou = label[mask]

    class_out_points = np.transpose((class_out*mask).nonzero())
    k_neigh = NearestNeighbors(n_neighbors=1)
    k_neigh.fit(class_out_points)

    sdf_values = get_discrete_sdf(config, model, shape_embedding)

    zero_array = np.zeros(SCALE)
    one_array = np.ones(SCALE)
    iou_out = []
    for threshold in config['EVAL']['eval_threshold']:
        pred_voxels = np.where(abs(sdf_values) < threshold, one_array, zero_array)
        pred_iou = pred_voxels[mask]

        complet_evaluator.reset()
        complet_evaluator.addBatch(pred_iou.astype(int), label_iou.astype(int))
        conf = complet_evaluator.get_confusion()
        iou_cmplt = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0]) * 100

        iou_out.append(iou_cmplt)

        if config['EVAL']['save_predict_point']:
            pos = (pred_voxels * mask).nonzero()
            pred_points = np.transpose(pos)
            indexes = k_neigh.kneighbors(pred_points, return_distance=False).squeeze()
            co_pred = class_out[tuple(np.transpose(class_out_points[indexes]))]
            classes = inv_map_lut[co_pred]
            colors = color_map_lut[classes]
            points = np.concatenate((pred_points, colors), axis=1)
            occupancy_f = os.path.join(save_dir, 'pred_'+str(threshold)+'.txt')
            np.savetxt(occupancy_f, points)

    # raw
    occupancy_f = os.path.join(save_dir, 'raw.xyz')
    np.savetxt(occupancy_f, raw)
    # label
    pos = (label * mask).nonzero()
    label = label.astype(np.int)
    classes = inv_map_lut[label[pos]]
    colors = color_map_lut[classes]
    points = np.concatenate((np.transpose(pos), colors), axis=1)
    occupancy_f = os.path.join(save_dir, 'gt.txt')
    np.savetxt(occupancy_f, points)
    # explicit ssc
    pos = (class_out * mask).nonzero()
    class_out = class_out.astype(np.int)
    classes = inv_map_lut[class_out[pos]]
    colors = color_map_lut[classes]
    points = np.concatenate((np.transpose(pos), colors), axis=1)
    occupancy_f = os.path.join(save_dir, 'ssc.txt')
    np.savetxt(occupancy_f, points)

    if config['EVAL']['mesh']['create_mesh']:

        voxel_origin = [0, 0, 0]
        mesh_level = config['EVAL']['mesh']['mesh_level']

        for ratio in config['EVAL']['mesh']['ratio']:
            sdf_values = get_discrete_sdf(config, model, shape_embedding, N=256*ratio)

            voxel_size = 1.0 / ratio

            dest_ply_path = os.path.join(save_dir, str(index)+'_'+str(ratio)+'.ply')

            convert_sdf_samples_to_ply(
                sdf_values,
                voxel_origin,
                voxel_size,
                dest_ply_path,
                class_out * mask,
                mesh_level,
            )

    return np.array(iou_out)


def scene_save_ssc_b(G_siren, G_label, shape_embedding, raw, label, mask, config, model_dir, index):
    save_dir = os.path.join(model_dir, str(index))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    SCALE = [256,256,32]

    NUM_CLASS_COMPLET = 20
    complet_evaluator = iouEval(NUM_CLASS_COMPLET, [])

    label_iou = label[mask]

    sdf_values, label_values = get_discrete_sdf_label_dec(config, G_siren, G_label, shape_embedding)

    class_out = label_values.argmax(-1)
    class_out_points = np.transpose((class_out*mask).nonzero())
    k_neigh = NearestNeighbors(n_neighbors=1)
    k_neigh.fit(class_out_points)

    zero_array = np.zeros(SCALE)
    one_array = np.ones(SCALE)
    iou_out = []
    for threshold in config['EVAL']['eval_threshold']:
        pred_voxels = np.where(abs(sdf_values) < threshold, one_array, zero_array)
        pred_iou = pred_voxels[mask]

        complet_evaluator.reset()
        complet_evaluator.addBatch(pred_iou.astype(int), label_iou.astype(int))
        conf = complet_evaluator.get_confusion()
        iou_cmplt = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0]) * 100

        iou_out.append(iou_cmplt)

        if config['EVAL']['save_predict_point']:
            pos = (pred_voxels * mask).nonzero()
            pred_points = np.transpose(pos)
            indexes = k_neigh.kneighbors(pred_points, return_distance=False).squeeze()
            co_pred = class_out[tuple(np.transpose(class_out_points[indexes]))]
            classes = inv_map_lut[co_pred]
            colors = color_map_lut[classes]
            points = np.concatenate((pred_points, colors), axis=1)
            occupancy_f = os.path.join(save_dir, 'pred_'+str(threshold)+'.txt')
            np.savetxt(occupancy_f, points)

    # raw
    occupancy_f = os.path.join(save_dir, 'raw.xyz')
    np.savetxt(occupancy_f, raw)
    # label
    pos = (label * mask).nonzero()
    label = label.astype(np.int)
    classes = inv_map_lut[label[pos]]
    colors = color_map_lut[classes]
    points = np.concatenate((np.transpose(pos), colors), axis=1)
    occupancy_f = os.path.join(save_dir, 'gt.txt')
    np.savetxt(occupancy_f, points)
    # explicit ssc
    pos = (class_out * mask).nonzero()
    class_out = class_out.astype(np.int)
    classes = inv_map_lut[class_out[pos]]
    colors = color_map_lut[classes]
    points = np.concatenate((np.transpose(pos), colors), axis=1)
    occupancy_f = os.path.join(save_dir, 'ssc.txt')
    np.savetxt(occupancy_f, points)

    if config['EVAL']['mesh']['create_mesh']:

        voxel_origin = [0, 0, 0]
        mesh_level = config['EVAL']['mesh']['mesh_level']

        for ratio in config['EVAL']['mesh']['ratio']:
            sdf_values, label_values = get_discrete_sdf_label_dec(config, G_siren, G_label, shape_embedding, N=256*ratio)

            sdf_values = abs(sdf_values) ###

            voxel_size = 1.0 / ratio

            dest_ply_path = os.path.join(save_dir, str(index)+'_'+str(ratio)+'.ply')

            convert_sdf_samples_to_ply(
                sdf_values,
                voxel_origin,
                voxel_size,
                dest_ply_path,
                class_out * mask,
                mesh_level,
            )

    return np.array(iou_out)


def convert_sdf_samples_to_ply(
    sdf_values,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    label,
    level,
    offset=None,
    scale=None,
    ):
    '''
    Convert sdf samples to .ply with semantic infomation

    :param sdf_values: a numpy array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    '''
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            sdf_values, level=level, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    ##################
    if label is not None:
        label_points = np.transpose(label.nonzero())
        k_neigh = NearestNeighbors(n_neighbors=1)
        k_neigh.fit(label_points)
        indexes = k_neigh.kneighbors(mesh_points, return_distance=False).squeeze()

        co_label = label[tuple(np.transpose(label_points[indexes]))]
        classes = inv_map_lut[co_label]
        colors = color_map_lut[classes]
    ##################

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    if label is not None:
        verts_tuple = np.zeros((num_verts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(np.concatenate((mesh_points[i, :], colors[i, :]), axis=-1))
    else:
        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[('vertex_indices', 'i4', (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, 'vertex')
    el_faces = plyfile.PlyElement.describe(faces_tuple, 'face')

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug('saving mesh to %s' % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        'converting to ply format and writing to file took {} s'.format(
            time.time() - start_time
        )
    )




