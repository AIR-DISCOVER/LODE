
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import diff_operators
import MinkowskiEngine as ME

from IPython import embed

class Loss_sc:
    def __init__(self, complt_w, loss_weights):
        self.complt_w = torch.Tensor(complt_w).cuda()
        self.loss_weights = loss_weights

    def cmplt_loss(self, out_cls, targets):
        
        cmplt_crit = nn.BCEWithLogitsLoss()

        num_layers, cmplt_loss = len(out_cls), torch.tensor(0.).cuda()
        for out_cl, target in zip(out_cls, targets):
            curr_loss = cmplt_crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).cuda())
            cmplt_loss += curr_loss / num_layers

        return cmplt_loss * self.loss_weights[4]

    def sdf_loss(self, model_output, gt):
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        shape_coords = model_output['model_in']
        pred_sdf = model_output['model_out']

        gradient = diff_operators.gradient(pred_sdf, shape_coords)[..., -3:]

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
        normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                        torch.zeros_like(gradient[..., :1]))
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

        return {'sdf': torch.abs(sdf_constraint).mean() * self.loss_weights[0],
                'inter': inter_constraint.mean() * self.loss_weights[1],
                'normal_constraint': normal_constraint.mean() * self.loss_weights[2],
                'grad_constraint': grad_constraint.mean() * self.loss_weights[3]}

    def all_loss(self, out_cls, targets, g_model_output, gt):

        ret_cmplt_loss = self.cmplt_loss(out_cls, targets)
        ret_sdf_loss = self.sdf_loss(g_model_output, gt)

        return {
                'cmplt_loss': ret_cmplt_loss,
                'sdf': ret_sdf_loss['sdf'],
                'inter': ret_sdf_loss['inter'],
                'normal_constraint': ret_sdf_loss['normal_constraint'],
                'grad_constraint': ret_sdf_loss['grad_constraint'],
                }


class Loss_ssc_a:
    def __init__(self, complt_w, loss_weights):
        self.complt_w = torch.Tensor(complt_w).cuda()
        self.loss_weights = loss_weights

    def seg_loss(self, class_out0, gt):
        label = gt['label']
        invalid = gt['invalid']

        masks = torch.ones_like(label, dtype=torch.bool)
        masks[:,:,:,:] = False
        masks[invalid == 1] = True
        label[masks] = 255

        seg_label = label[tuple(np.transpose(class_out0.C.cpu().numpy()))].cuda()
        seg_loss = F.cross_entropy(class_out0.F, seg_label, weight=self.complt_w, ignore_index=255)

        return seg_loss * self.loss_weights[5]

    def moo_seg_loss(self, class_out0, class_out0_F, gt):
        label = gt['label']
        invalid = gt['invalid']

        masks = torch.ones_like(label, dtype=torch.bool)
        masks[:,:,:,:] = False
        masks[invalid == 1] = True
        label[masks] = 255

        seg_label = label[tuple(np.transpose(class_out0.C.cpu().numpy()))].cuda()
        seg_loss = F.cross_entropy(class_out0_F, seg_label, weight=self.complt_w, ignore_index=255)

        return seg_loss * self.loss_weights[5]

    def ssc_loss(self, class_out1, gt):
        label = gt['label']
        invalid = gt['invalid']

        masks = torch.ones_like(label, dtype=torch.bool)
        masks[:,:,:,:] = False
        masks[invalid == 1] = True
        label[masks] = 255

        ssc_loss = F.cross_entropy(class_out1, label, weight=self.complt_w, ignore_index=255)

        return ssc_loss * self.loss_weights[6]

    def cmplt_loss(self, out_cls, targets):
        
        cmplt_crit = nn.BCEWithLogitsLoss()

        num_layers, cmplt_loss = len(out_cls), torch.tensor(0.).cuda()
        for out_cl, target in zip(out_cls, targets):
            curr_loss = cmplt_crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).cuda())
            cmplt_loss += curr_loss / num_layers

        return cmplt_loss * self.loss_weights[4]

    def sdf_loss(self, model_output, gt):
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        shape_coords = model_output['model_in']
        pred_sdf = model_output['model_out']

        gradient = diff_operators.gradient(pred_sdf, shape_coords)[..., -3:]

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
        normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                        torch.zeros_like(gradient[..., :1]))
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

        return {'sdf': torch.abs(sdf_constraint).mean() * self.loss_weights[0],
                'inter': inter_constraint.mean() * self.loss_weights[1],
                'normal_constraint': normal_constraint.mean() * self.loss_weights[2],
                'grad_constraint': grad_constraint.mean() * self.loss_weights[3]}

    def shape_siren_loss(self, out_cls, targets, g_model_output, gt):
        ret_cmplt_loss = self.cmplt_loss(out_cls, targets)

        ret_sdf_loss = self.sdf_loss(g_model_output, gt)

        final_loss = ret_cmplt_loss + ret_sdf_loss['sdf'] + ret_sdf_loss['inter'] + \
                    ret_sdf_loss['normal_constraint'] + ret_sdf_loss['grad_constraint']

        return final_loss

    def all_loss(self, class_out0, class_out1, out_cls, targets, g_model_output, gt):

        ret_seg_loss = self.seg_loss(class_out0, gt)

        ret_ssc_loss = self.ssc_loss(class_out1, gt)

        ret_cmplt_loss = self.cmplt_loss(out_cls, targets)
        ret_sdf_loss = self.sdf_loss(g_model_output, gt)

        return {
                'seg_loss': ret_seg_loss,
                'ssc_loss': ret_ssc_loss,
                'cmplt_loss': ret_cmplt_loss,
                'sdf': ret_sdf_loss['sdf'],
                'inter': ret_sdf_loss['inter'],
                'normal_constraint': ret_sdf_loss['normal_constraint'],
                'grad_constraint': ret_sdf_loss['grad_constraint'],
                }


class Loss_ssc_b:
    def __init__(self, complt_w, loss_weights):
        self.complt_w = torch.Tensor(complt_w).cuda()
        self.loss_weights = loss_weights

    def cmplt_loss(self, out_cls, targets):
        
        cmplt_crit = nn.BCEWithLogitsLoss()

        num_layers, cmplt_loss = len(out_cls), torch.tensor(0.).cuda()
        for out_cl, target in zip(out_cls, targets):
            curr_loss = cmplt_crit(out_cl.F.squeeze(), target.type(out_cl.F.dtype).cuda())
            cmplt_loss += curr_loss / num_layers

        return cmplt_loss * self.loss_weights[4]

    def sdf_loss(self, model_in, sdf_out, gt):
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']

        shape_coords = model_in
        pred_sdf = sdf_out

        gradient = diff_operators.gradient(pred_sdf, shape_coords)[..., -3:]

        # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
        sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
        inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
        normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                        torch.zeros_like(gradient[..., :1]))
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

        return {'sdf': torch.abs(sdf_constraint).mean() * self.loss_weights[0],
                'inter': inter_constraint.mean() * self.loss_weights[1],
                'normal_constraint': normal_constraint.mean() * self.loss_weights[2],
                'grad_constraint': grad_constraint.mean() * self.loss_weights[3]}

    def label_loss(self, label_out, gt):
        gt_label = gt['out_label_points']

        pred_label = label_out.transpose(1,2)

        label_loss = F.cross_entropy(pred_label, gt_label, weight=self.complt_w, ignore_index=255)

        return label_loss * self.loss_weights[5]

    def all_loss(self, out_cls, targets, g_model_output, gt):

        ret_cmplt_loss = self.cmplt_loss(out_cls, targets)

        ret_sdf_loss = self.sdf_loss(g_model_output['model_in'], g_model_output['sdf_out'], gt)

        ret_label_loss = self.label_loss(g_model_output['label_out'], gt)

        return {
                'cmplt_loss': ret_cmplt_loss,
                'sdf': ret_sdf_loss['sdf'],
                'inter': ret_sdf_loss['inter'],
                'normal_constraint': ret_sdf_loss['normal_constraint'],
                'grad_constraint': ret_sdf_loss['grad_constraint'],
                'label_loss': ret_label_loss,
                }

