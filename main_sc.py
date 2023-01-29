
import os
import sys
import configargparse
import yaml
import shutil
import time

from tqdm.autonotebook import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils import data

import MinkowskiEngine as ME

from sklearn.neighbors import NearestNeighbors

import dataio, modules, loss, evals
from min_norm_solvers import MinNormSolver, gradient_normalizers

from IPython import embed


config_file = os.path.join('semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))

class_strings = kitti_config["labels"]
class_inv_remap = kitti_config["learning_map_inv"]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.val > self.max:
            self.max = self.val


def eval_single(D_model, G_model, valid_iter, total_steps, iou_score, writer, config):
    x_size = int(256 / config['TRAIN']['chunk_size'])
    y_size = int(256 / config['TRAIN']['chunk_size'])
    z_size = int(32 / config['TRAIN']['chunk_size'])

    if dist.get_rank() == 0:    
        print('================================')
        print(config['experiment_name'])

    with torch.no_grad():
        D_model.eval()
        G_model.eval()

        SCALE = [256,256,32]
        zero_array = np.zeros(SCALE)
        one_array = np.ones(SCALE)

        NUM_CLASS_COMPLET = 20
        complet_evaluator = []
        for i in range(len(config['TRAIN']['eval_threshold'])):
            complet_evaluator.append(evals.iouEval(NUM_CLASS_COMPLET, [])) 

        indices, eval_info, raw_data, in_feat, occupancy = valid_iter.next()

        sparse_input = ME.SparseTensor(
            features=in_feat,
            coordinates=raw_data,
            device='cuda',
        )

        cm = sparse_input.coordinate_manager
        target_key, _ = cm.insert_and_map(
            occupancy.cuda(),
            string_id='target',
        )

        out_cls, targets, sout, shape_out = D_model(sparse_input, target_key)
        
        shape_out = shape_out.dense( \
            shape=torch.Size([config['DATA_IO']['valid_batch_size'],config['TRAIN']['shape_embedding_size'],x_size,y_size,z_size]))[0]
        if config['TRAIN']['shape_normalize'] == True:
            shape_out = F.normalize(shape_out, p=2, dim=1)

        sdf_values = evals.get_discrete_sdf(config, G_model, shape_out[0])

        label = eval_info['label'][0]
        mask = eval_info['mask'][0]
        label_iou = label[mask]

        for i in range(len(config['TRAIN']['eval_threshold'])):
            threshold = config['TRAIN']['eval_threshold'][i]
            pred_voxels = np.where(abs(sdf_values) < threshold, one_array, zero_array)
            pred_iou = pred_voxels[mask]

            complet_evaluator[i].addBatch(pred_iou.astype(int), label_iou.astype(int))

        iou_list = []
        for i in range(len(complet_evaluator)):
            conf = complet_evaluator[i].get_confusion()
            iou_cmplt = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0]) * 100

            iou_list.append(iou_cmplt)

        if dist.get_rank() == 0:    
            writer.add_scalar('iou', iou_list[1], total_steps)
            iou_score.update(val=iou_list[1])
            print('================================')
            print(indices)
            print('iou: ', iou_list, iou_score.avg)
            print('best iou: ', iou_score.max)
            print('================================')


def train_single_epoch(D_model, G_model, model_params, train_dataloader, valid_dataloader, 
    d_optim, g_optim, loss_fn, epoch, total_steps, iou_score, train_losses, writer, pbar, config):

    steps_til_eval = config['TRAIN']['steps_til_eval']
    steps_til_summary = config['TRAIN']['steps_til_summary']
    clip_grad = config['TRAIN']['clip_grad']
    x_size = int(256 / config['TRAIN']['chunk_size'])
    y_size = int(256 / config['TRAIN']['chunk_size'])
    z_size = int(32 / config['TRAIN']['chunk_size'])

    valid_iter = iter(valid_dataloader)

    for step, (indices, points, gt, raw_data, in_feat, occupancy) in enumerate(train_dataloader):
        time1 = time.time()

        d_optim.zero_grad()
        g_optim.zero_grad()

        D_model.train()
        G_model.train()

        ####### D_model train #######
        sparse_input = ME.SparseTensor(
            features=in_feat,
            coordinates=raw_data,
            device='cuda',
        )

        cm = sparse_input.coordinate_manager
        target_key, _ = cm.insert_and_map(
            occupancy.cuda(),
            string_id="target",
        )

        out_cls, targets, sout, shape_out = D_model(sparse_input, target_key)

        shape_out = shape_out.dense( \
            shape=torch.Size([config['DATA_IO']['train_batch_size'],config['TRAIN']['shape_embedding_size'],x_size,y_size,z_size]))[0]
        if config['TRAIN']['shape_normalize'] == True:
            shape_out = F.normalize(shape_out, p=2, dim=1)

        ####### G_model train #######
        coords = points['coords'].cuda()

        # trilinear #
        shape_out = shape_out.transpose(2,4) # transpose axis x and z
        scaled_coords = coords.clone().detach()
        scaled_coords[:,:,2] = ((scaled_coords[:,:,2] + 1.) / 0.25 - 0.5) * 2.  # coords z located in [-1., -0.75], scale to [-1,1]
        if config['TRAIN']['shape_sample_strategy'] == 'trilinear':
            shapes = F.grid_sample(shape_out, scaled_coords[:,:,:3].unsqueeze(2).unsqueeze(3), \
                        mode='bilinear', padding_mode='border', align_corners=False)
        else:
            shapes = F.grid_sample(shape_out, scaled_coords[:,:,:3].unsqueeze(2).unsqueeze(3), \
                        mode='nearest', padding_mode='border', align_corners=False)
        shapes = shapes.squeeze(-1).squeeze(-1).transpose(1,2).cuda()   # batch_size * point_num * shape_embedding_size

        gt = {key: value.cuda() for key, value in gt.items()}

        g_model_output = G_model(shapes, coords)

        loss_data = loss_fn.all_loss(out_cls, targets, g_model_output, gt)
        loss = loss_data['cmplt_loss'] + \
               loss_data['sdf'] + \
               loss_data['inter'] + \
               loss_data['normal_constraint'] + \
               loss_data['grad_constraint']

        train_losses.append(loss.item())

        loss.backward()
        if clip_grad:
            if isinstance(clip_grad, bool):
                torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.)
            else:
                torch.nn.utils.clip_grad_norm_(model_params, max_norm=clip_grad)
        d_optim.step()
        g_optim.step()

        if dist.get_rank() == 0:
            writer.add_scalar('training_loss', loss.item(), total_steps)
            for k, v in loss_data.items():
                writer.add_scalar(k, loss_data[k].item(), total_steps)

            pbar.update(1)
            if not total_steps % steps_til_summary:
                tqdm.write('Epoch %d, Total loss %0.6f, iteration time %0.6f' % (epoch, loss.item(), time.time() - time1))

        total_steps += 1

        if not total_steps % steps_til_eval and total_steps:
            eval_single(D_model, G_model, valid_iter, total_steps, iou_score, writer, config)

    return total_steps, train_losses


def train_pipeline(D_model, G_model, train_dataloader, valid_dataloader, train_sampler, valid_sampler, loss_fn, model_dir, config):
    epochs_til_ckpt = config['TRAIN']['epochs_til_ckpt']

    if dist.get_rank() == 0:
        summaries_dir = os.path.join(model_dir, 'summaries')
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        writer = SummaryWriter(summaries_dir)
    else:
        writer = 0

    epochs = config['TRAIN']['num_epochs']
    epoch = 0
    lr = config['TRAIN']['lr']

    model_params = []
    model_params += D_model.parameters()
    model_params += G_model.parameters()

    d_optim = torch.optim.Adam(lr=lr, params=D_model.parameters())
    g_optim = torch.optim.Adam(lr=lr, params=G_model.parameters())

    if config['TRAIN']['lr_scheduler']:
        d_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, step_size=10, gamma=0.9, last_epoch=-1)
        g_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=10, gamma=0.9, last_epoch=-1)

    if config['TRAIN']['resume']:
        checkpoint = torch.load(config['TRAIN']['resume_path'], 'cuda')

        d_optim.load_state_dict(checkpoint["d_optim"])
        g_optim.load_state_dict(checkpoint["g_optim"])

        d_scheduler.load_state_dict(checkpoint["d_scheduler"])
        g_scheduler.load_state_dict(checkpoint["g_scheduler"])
        
        total_steps = checkpoint['total_steps']

        # epoch = checkpoint['epoch']
    else:
        total_steps = 0
        # epoch = 0

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        if dist.get_rank() == 0:
            pbar.update(len(train_dataloader) * epoch)

        train_losses = []

        iou_score=AverageMeter()

        epoch_rest = epochs - epoch
        for i in range(epoch_rest):
            if not epoch % epochs_til_ckpt and epoch and dist.get_rank() == 0:
                torch.save(
                    {
                        'D_model': D_model.module.state_dict(),
                        'G_model': G_model.module.state_dict(),
                        'd_optim': d_optim.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_scheduler': d_scheduler.state_dict(),
                        'g_scheduler': g_scheduler.state_dict(),
                        'total_steps': total_steps,
                        # 'epoch': epoch,
                    },
                    os.path.join(checkpoints_dir, 'weights_epoch_%04d_steps_%d.pth' % (epoch, total_steps))
                )
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d_steps_%d.txt' % (epoch, total_steps)),
                        np.array(train_losses))

            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

            total_steps, train_losses = train_single_epoch(D_model, G_model, model_params, \
                                            train_dataloader, valid_dataloader, d_optim, g_optim, loss_fn, \
                                            epoch, total_steps, iou_score, train_losses, writer, pbar, config)

            if config['TRAIN']['lr_scheduler']:
                d_scheduler.step()
                g_scheduler.step()

            epoch += 1

        if dist.get_rank() == 0:
            torch.save(
                {
                    'D_model': D_model.module.state_dict(),
                    'G_model': G_model.module.state_dict(),
                    'd_optim': d_optim.state_dict(),
                    'g_optim': g_optim.state_dict(),
                    'd_scheduler': d_scheduler.state_dict(),
                    'g_scheduler': g_scheduler.state_dict(),
                    'total_steps': total_steps,
                    # 'epoch': epoch,
                },
                os.path.join(checkpoints_dir, 'weights_final.pth')
            )
            np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                    np.array(train_losses))


def train(opt, config, expr_path):
    # dataset and dataloader
    train_dataset = dataio.DG_Dataset(config, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=dataio.DG_DataMerge_train,
        batch_size=config['DATA_IO']['train_batch_size'],
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=config['DATA_IO']['num_workers'],
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    valid_dataset = dataio.DG_Dataset(config, split='valid')
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        collate_fn=dataio.DG_DataMerge_valid,
        batch_size=config['DATA_IO']['valid_batch_size'],
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=config['DATA_IO']['num_workers'],
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    # model
    D_model = modules.D_Net(config)
    # G_model = modules.G_siren(config)
    if config['TRAIN']['encode_xyz'] == True:
        xyz_dim = 3 * (config['TRAIN']['inc_input'] + config['TRAIN']['encode_levels'] * 2)
    else:
        xyz_dim = 3
    G_model = modules.SingleBVPNet(out_features=1,
                           type=config['TRAIN']['G_TRAIN']['nonlinearity'],
                           in_features=(xyz_dim+config['TRAIN']['shape_embedding_size']), 
                           hidden_features=config['TRAIN']['G_TRAIN']['hidden_features'],
                           num_hidden_layers=config['TRAIN']['G_TRAIN']['num_hidden_layers'],
                           config=config)

    D_model = D_model.cuda()
    G_model = G_model.cuda()

    D_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(D_model)
    G_model = nn.SyncBatchNorm.convert_sync_batchnorm(G_model)

    D_model = torch.nn.parallel.DistributedDataParallel(D_model, device_ids=[opt.local_rank])
    G_model = torch.nn.parallel.DistributedDataParallel(G_model, device_ids=[opt.local_rank])

    if config['TRAIN']['resume']:
        checkpoint = torch.load(config['TRAIN']['resume_path'], 'cuda')

        D_model.module.load_state_dict(checkpoint["D_model"])
        G_model.module.load_state_dict(checkpoint["G_model"])

    loss_fn = loss.Loss_sc(train_dataset.compl_labelweights, config['TRAIN']['loss_weights'])

    # train
    train_pipeline(D_model=D_model, G_model=G_model, \
                train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, \
                train_sampler=train_sampler, valid_sampler=valid_sampler, \
                loss_fn=loss_fn, model_dir=expr_path, config=config)


def valid_pipeline(D_model, G_model, valid_dataloader, model_dir, config):

    x_size = int(256 / config['TRAIN']['chunk_size'])
    y_size = int(256 / config['TRAIN']['chunk_size'])
    z_size = int(32 / config['TRAIN']['chunk_size'])

    print(config['experiment_name'])
    results = []
    results.append(config['experiment_name'])

    all_cd = np.zeros_like(config['EVAL']['eval_threshold'])

    with torch.no_grad():
        D_model.eval()
        G_model.eval()

        SCALE = [256,256,32]
        zero_array = np.zeros(SCALE)
        one_array = np.ones(SCALE)

        NUM_CLASS_COMPLET = 20
        complet_evaluator = []
        for i in range(len(config['EVAL']['eval_threshold'])):
            complet_evaluator.append(evals.iouEval(NUM_CLASS_COMPLET, [])) 

        with tqdm(total=len(valid_dataloader)) as pbar:
            for step, (indices, eval_info, raw_data, in_feat, occupancy) in enumerate(valid_dataloader):

                sparse_input = ME.SparseTensor(
                    features=in_feat,
                    coordinates=raw_data,
                    device='cuda',
                )

                cm = sparse_input.coordinate_manager
                target_key, _ = cm.insert_and_map(
                    occupancy.cuda(),
                    string_id="target",
                )

                out_cls, targets, sout, shape_out = D_model(sparse_input, target_key)

                shape_out = shape_out.dense( \
                    shape=torch.Size([config['DATA_IO']['valid_batch_size'],config['TRAIN']['shape_embedding_size'],x_size,y_size,z_size]))[0]
                if config['TRAIN']['shape_normalize'] == True:
                    shape_out = F.normalize(shape_out, p=2, dim=1)

                sdf_values = evals.get_discrete_sdf(config, G_model, shape_out[0])

                label = eval_info['label'][0]
                mask = eval_info['mask'][0]
                label_iou = label[mask]

                single_cd = np.zeros_like(config['EVAL']['eval_threshold'])
                for i in range(len(config['EVAL']['eval_threshold'])):
                    threshold = config['EVAL']['eval_threshold'][i]
                    pred_voxels = np.where(abs(sdf_values) < threshold, one_array, zero_array)
                    pred_iou = pred_voxels[mask]

                    complet_evaluator[i].addBatch(pred_iou.astype(int), label_iou.astype(int))

                    if config['EVAL']['eval_cd']:
                        single_cd[i] = evals.eval_cd(pred_voxels, label, mask)
                    else:
                        single_cd[i] = 0

                all_cd += single_cd

                pbar.update(1)

                if config['GENERAL']['debug'] and step > -1:
                    break

        iou_list = []
        for i in range(len(complet_evaluator)):
            conf = complet_evaluator[i].get_confusion()
            iou_cmplt = (np.sum(conf[1:, 1:])) / (np.sum(conf) - conf[0, 0]) * 100

            iou_list.append(iou_cmplt)
        print('iou: ', iou_list)
                

    results.append('========================')
    results.append('\nIoU:  '+str(iou_list))
    results.append('\nCD:   '+str(all_cd / len(valid_dataloader)))
    results.append('\nThreshold: '+str(config['EVAL']['eval_threshold']))
    np.savetxt(os.path.join(model_dir, 'pred_iou.txt'), results, fmt='%s')


def valid(opt, config, expr_path):
    valid_dataset = dataio.DG_Dataset(config, split='valid')
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset,
        collate_fn=dataio.DG_DataMerge_valid,
        batch_size=config['DATA_IO']['valid_batch_size'],
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=True,
        num_workers=config['DATA_IO']['num_workers'],
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    # model
    D_model = modules.D_Net(config)
    # G_model = modules.G_siren(config)
    if config['TRAIN']['encode_xyz'] == True:
        xyz_dim = 3 * (config['TRAIN']['inc_input'] + config['TRAIN']['encode_levels'] * 2)
    else:
        xyz_dim = 3
    G_model = modules.SingleBVPNet(out_features=1,
                           type=config['TRAIN']['G_TRAIN']['nonlinearity'],
                           in_features=(xyz_dim+config['TRAIN']['shape_embedding_size']), 
                           hidden_features=config['TRAIN']['G_TRAIN']['hidden_features'],
                           num_hidden_layers=config['TRAIN']['G_TRAIN']['num_hidden_layers'],
                           config=config)

    D_model = D_model.cuda()
    G_model = G_model.cuda()

    D_model = torch.nn.parallel.DistributedDataParallel(D_model, device_ids=[opt.local_rank])
    G_model = torch.nn.parallel.DistributedDataParallel(G_model, device_ids=[opt.local_rank])

    checkpoint = torch.load(config['EVAL']['checkpoint_path'], 'cuda')

    D_model.module.load_state_dict(checkpoint['D_model'])
    G_model.module.load_state_dict(checkpoint['G_model'])

    # config saved
    with open(os.path.join(expr_path, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # valid
    valid_pipeline(D_model=D_model, G_model=G_model, \
                valid_dataloader=valid_dataloader, \
                model_dir=expr_path, config=config)


def visualize_pipeline(D_model, G_model, dataloader, model_dir, config):
    
    x_size = int(256 / config['TRAIN']['chunk_size'])
    y_size = int(256 / config['TRAIN']['chunk_size'])
    z_size = int(32 / config['TRAIN']['chunk_size'])

    print(config['experiment_name'])

    results = []
    all_iou = np.zeros_like(config['EVAL']['eval_threshold'])

    results.append(config['experiment_name'])
    results.append('========================')

    with torch.no_grad():
        D_model.eval()
        G_model.eval()

        with tqdm(total=len(dataloader)) as pbar:
            for step, (indices, eval_info, raw_data, in_feat, occupancy) in enumerate(dataloader):

                sparse_input = ME.SparseTensor(
                    features=in_feat,
                    coordinates=raw_data,
                    device='cuda',
                )

                cm = sparse_input.coordinate_manager
                target_key, _ = cm.insert_and_map(
                    occupancy.cuda(),
                    string_id="target",
                )

                out_cls, targets, sout, shape_out = D_model(sparse_input, target_key)

                shape_out = shape_out.dense( \
                    shape=torch.Size([config['DATA_IO']['valid_batch_size'],config['TRAIN']['shape_embedding_size'],x_size,y_size,z_size]))[0]
                if config['TRAIN']['shape_normalize'] == True:
                    shape_out = F.normalize(shape_out, p=2, dim=1)

                raw = eval_info['raw'][0]
                label = eval_info['label'][0]
                mask = eval_info['mask'][0]

                iou, cd = evals.scene_save_sc(G_model, shape_out[0], raw, label, mask, config, model_dir, indices[0])

                results.append('>>> '+str(indices[0])+' '+str(iou))

                all_iou += iou

                print(indices)
                print('IoU:', iou)
                print('CD:', cd)

                if config['GENERAL']['debug']: break    
        
            print('================================')

        results.append('========================')
        results.append('\nAverage IoU:  '+str(all_iou / len(dataloader)))
        results.append('\nThreshold: '+str(config['EVAL']['eval_threshold']))
        np.savetxt(os.path.join(model_dir, 'pred_iou.txt'), results, fmt='%s')


def visualize(opt, config, expr_path):
    dataset = dataio.DG_Dataset(config, split='valid')
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=dataio.DG_DataMerge_valid,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
        num_workers=config['DATA_IO']['num_workers'],
        worker_init_fn=lambda x: np.random.seed(x + int(time.time()))
    )

    # model
    D_model = modules.D_Net(config)
    # G_model = modules.G_siren(config)
    if config['TRAIN']['encode_xyz'] == True:
        xyz_dim = 3 * (config['TRAIN']['inc_input'] + config['TRAIN']['encode_levels'] * 2)
    else:
        xyz_dim = 3
    G_model = modules.SingleBVPNet(out_features=1,
                           type=config['TRAIN']['G_TRAIN']['nonlinearity'],
                           in_features=(xyz_dim+config['TRAIN']['shape_embedding_size']), 
                           hidden_features=config['TRAIN']['G_TRAIN']['hidden_features'],
                           num_hidden_layers=config['TRAIN']['G_TRAIN']['num_hidden_layers'],
                           config=config)

    D_model = D_model.cuda()
    G_model = G_model.cuda()

    D_model = torch.nn.parallel.DistributedDataParallel(D_model, device_ids=[opt.local_rank])
    G_model = torch.nn.parallel.DistributedDataParallel(G_model, device_ids=[opt.local_rank])

    checkpoint = torch.load(config['EVAL']['checkpoint_path'], 'cuda')

    D_model.module.load_state_dict(checkpoint['D_model'])
    G_model.module.load_state_dict(checkpoint['G_model'])

    # config saved
    with open(os.path.join(expr_path, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # valid
    visualize_pipeline(D_model=D_model, G_model=G_model, \
                dataloader=dataloader, \
                model_dir=expr_path, config=config)


def parse_args():
    # argument
    p = configargparse.ArgumentParser()
    p.add_argument('--task', type=str, help='train, valid, visualize')

    p.add_argument('--config', type=str, default='opt.yaml', help='path to config file')
    p.add_argument('--experiment_name', type=str, required=True, help='name of experiment')

    p.add_argument('--local_rank', type=int)
    opt = p.parse_args()

    return opt


def main():
    opt = parse_args()

    config = yaml.safe_load(open(opt.config, 'r'))
    config['experiment_name']=opt.experiment_name
    config['GENERAL']['task']=opt.task

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(opt.local_rank)

    # expr path
    if config['GENERAL']['task'] == 'train':
        log_root = config['GENERAL']['logging_root']
    elif config['GENERAL']['task'] == 'valid' or config['GENERAL']['task'] == 'visualize':
        log_root = config['GENERAL']['eval_logging_root']
    expr_path = os.path.join(log_root, opt.experiment_name)

    if dist.get_rank() == 0:
        if os.path.exists(expr_path):
            if not (config['TRAIN']['resume'] and (config['GENERAL']['task'] == 'train')):
                overwrite = input('The model directory %s exists. Overwrite? (y/n)'%expr_path)
                if overwrite == 'y':
                    shutil.rmtree(expr_path)
                    os.makedirs(expr_path)
                else:
                    raise RuntimeError('The model directory %s already exists.'%expr_path)
        else:
            os.makedirs(expr_path)

        # config saved
        config_path = os.path.join(expr_path, 'config.yaml')
        if os.path.exists(config_path):
            os.remove(config_path)
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f)
        shutil.copy('dataio.py', os.path.join(expr_path, 'dataio.py'))
        shutil.copy('modules.py', os.path.join(expr_path, 'modules.py'))
        shutil.copy('loss.py', os.path.join(expr_path, 'loss.py'))
        shutil.copy('evals.py', os.path.join(expr_path, 'evals.py'))
        shutil.copy('main_sc.py', os.path.join(expr_path, 'main_sc.py'))

    if config['GENERAL']['task'] == 'train':
        train(opt, config, expr_path)
    elif config['GENERAL']['task'] == 'valid':
        valid(opt, config, expr_path)
    elif config['GENERAL']['task'] == 'visualize':
        visualize(opt, config, expr_path)


if __name__ == '__main__':
    main()

