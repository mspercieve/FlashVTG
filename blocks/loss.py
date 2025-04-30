# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import LOSSES, Parameter, build_loss

import torch.nn as nn
import torch.nn.functional as F
from .utils import weight_reduce_loss

def quality_focal_loss(
    pred,       # (B, N)
    label,      # (B, N), 0 or 1
    score,      # (B, N), quality (IoU)
    weight=None,
    beta=2.0,
    reduction='mean',
    avg_factor=None
):
    pred_sigmoid = pred.sigmoid()
    zerolabel = pred.new_zeros(pred.shape)

    # 기본 loss: BCE * pt^beta
    loss = F.binary_cross_entropy_with_logits(pred, zerolabel, reduction='none') * pred_sigmoid.pow(beta)
    # positive index (label == 1)
    pos_mask = label > 0
    if pos_mask.sum() > 0:
        # overwrite positive loss
        pos_pred = pred[pos_mask]
        pos_score = score[pos_mask]
        pos_pred_sigmoid = pred_sigmoid[pos_mask]

        pt = torch.abs(pos_score - pos_pred_sigmoid)  # closer → lower loss
        loss[pos_mask] = F.binary_cross_entropy_with_logits(pos_pred, pos_score, reduction='none') * pt.pow(beta)

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def distribution_focal_loss(
        pred,          # [B, N, C]
        label,         # [B, N] (float, continuous target)
        weight=None,
        reduction='mean',
        avg_factor=None):
    
    B, N, C = pred.shape

    disl = label.long()           # [B, N]
    disr = disl + 1               # [B, N]

    wl = disr.float() - label     # [B, N]
    wr = label - disl.float()     # [B, N]

    # 3. Flatten
    pred_flat = pred.view(-1, C)      # [B*N, C]
    disl_flat = disl.view(-1)         # [B*N]
    disr_flat = disr.view(-1)         # [B*N]
    wl_flat = wl.view(-1)             # [B*N]
    wr_flat = wr.view(-1)             # [B*N]

    # 4. Cross-entropy loss for left and right bins
    loss = F.cross_entropy(pred_flat, disl_flat, reduction='none') * wl_flat + \
           F.cross_entropy(pred_flat, disr_flat, reduction='none') * wr_flat

    loss = loss.view(B, N)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@LOSSES.register()
class QualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=2.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(QualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                score,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * quality_focal_loss(
                pred,
                target,
                score,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls


@LOSSES.register()
class DistributionFocalLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss_cls


@LOSSES.register()
class SampledNCELoss(nn.Module):

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=False,
                 direction=('row', 'col'),
                 loss_weight=1.0):
        super(SampledNCELoss, self).__init__()

        scale = torch.Tensor([math.log(1 / temperature)])

        if learnable:
            self.scale = Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        self.temperature = temperature
        self.max_scale = max_scale
        self.learnable = learnable
        self.direction = (direction, ) if isinstance(direction, str) else direction
        self.loss_weight = loss_weight

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, direction={}, loss_weight={}'
                .format(self.temperature, self.max_scale, self.learnable, self.direction,
                        self.loss_weight))

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(video_emb.size(0), device=video_emb.device)

        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale)
        i_sim = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale
        i_sim = i_sim + torch.where(loss_msk > 0, .0, float('-inf'))

        loss = 0

        if 'row' in self.direction:
            i_met = F.log_softmax(i_sim, dim=1)[batch_inds, pos_clip]
            loss = loss - i_met.sum() / i_met.size(0)

        if 'col' in self.direction:
            j_sim = i_sim.t()
            j_met = F.log_softmax(j_sim, dim=1)[pos_clip, batch_inds]
            loss = loss - j_met.sum() / j_met.size(0)

        loss = loss * self.loss_weight
        return loss


@LOSSES.register()
class BundleLoss(nn.Module):

    def __init__(self,
                 sample_radius=1.5,
                 loss_cls=None,
                 loss_reg=None,
                 loss_sal=None,
                 loss_qfl=None,
                 loss_dfl=None,
                 ):
        super(BundleLoss, self).__init__()

        self._loss_cls = build_loss(loss_cls)
        self._loss_reg = build_loss(loss_reg)
        self._loss_sal = build_loss(loss_sal)
        self._loss_qfl = build_loss(loss_qfl)
        self._loss_dfl = build_loss(loss_dfl)
        self.sample_radius = sample_radius

    def get_target_single(self, point, gt_bnd, gt_cls):
        num_pts, num_gts = point.size(0), gt_bnd.size(0)

        lens = gt_bnd[:, 1] - gt_bnd[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        gt_seg = gt_bnd[None].expand(num_pts, num_gts, 2)
        s = point[:, 0, None] - gt_seg[:, :, 0] # [num_pts, num_gts] 
        e = gt_seg[:, :, 1] - point[:, 0, None] # [num_pts, num_gts] 
        r_tgt = torch.stack((s, e), dim=-1) # [num_pts, num_gts, 2]

        if self.sample_radius > 0:
            center = (gt_seg[:, :, 0] + gt_seg[:, :, 1]) / 2
            t_mins = center - point[:, 3, None] * self.sample_radius
            t_maxs = center + point[:, 3, None] * self.sample_radius
            dist_s = point[:, 0, None] - torch.maximum(t_mins, gt_seg[:, :, 0])
            dist_e = torch.minimum(t_maxs, gt_seg[:, :, 1]) - point[:, 0, None]
            center = torch.stack((dist_s, dist_e), dim=-1) # [num_pts, num_gts, 2]
            cls_msk = center.min(-1)[0] >= 0 #[num_pts, num_gts] 
        else:
            cls_msk = r_tgt.min(-1)[0] >= 0

        reg_dist = r_tgt.max(-1)[0]
        reg_msk = torch.logical_and((reg_dist >= point[:, 1, None]),
                                    (reg_dist <= point[:, 2, None]))

        lens.masked_fill_(cls_msk == 0, float('inf'))
        lens.masked_fill_(reg_msk == 0, float('inf')) 
        min_len, min_len_inds = lens.min(dim=1)

        min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)),
                                         (lens < float('inf'))).to(r_tgt.dtype)

        label = F.one_hot(gt_cls[:, 0], 2).to(r_tgt.dtype)
        c_tgt = torch.matmul(min_len_mask, label).clamp(min=0.0, max=1.0)[:, 1]
        r_tgt = r_tgt[range(num_pts), min_len_inds] / point[:, 3, None]
        return c_tgt, r_tgt

    def get_target(self, data):
        cls_tgt, reg_tgt = [], []

        for i in range(data['boundary'].size(0)):
            gt_bnd = data['boundary'][i] * data['fps'][i]
            gt_cls = gt_bnd.new_ones(gt_bnd.size(0), 1).long()

            c_tgt, r_tgt = self.get_target_single(data['point'], gt_bnd, gt_cls)

            cls_tgt.append(c_tgt)
            reg_tgt.append(r_tgt)

        cls_tgt = torch.stack(cls_tgt)
        reg_tgt = torch.stack(reg_tgt)

        return cls_tgt, reg_tgt

    def get_iou(self, point, reg_pred, reg_tgt):
        """
        Args:
            point:    [N, 4], 마지막 dim은 (center, min_reg, max_reg, stride)
            reg_tgt:  [B, N, 2], regression target (offsets to GT boundaries)

        Returns:
            iou_targets: [B, N]
        """
        # [N] → [1, N] → broadcast to [B, N]
        center = point[:, 0][None, :]         # [1, N]
        stride = point[:, 3][None, :]         # [1, N]

        # predicted box
        pred_start = center - reg_pred[:, :, 0] * stride
        pred_end   = center + reg_pred[:, :, 1] * stride

        # GT box (reconstructed same way)
        gt_start = center - reg_tgt[:, :, 0] * stride
        gt_end   = center + reg_tgt[:, :, 1] * stride

        # intersection and union
        inter_left  = torch.max(pred_start, gt_start)
        inter_right = torch.min(pred_end, gt_end)
        inter = (inter_right - inter_left).clamp(min=0)

        union_left  = torch.min(pred_start, gt_start)
        union_right = torch.max(pred_end, gt_end)
        union = (union_right - union_left).clamp(min=1e-6)

        iou = inter / union  # [B, N]
        return iou

    def loss_qfl(self, data, output, cls_tgt, reg_tgt):
        src = data['out_class'].squeeze(-1)
        reg_pred = data['out_coord']
        msk = torch.cat(data['pymid_msk'], dim=1)
        score = self.get_iou(data['point'], reg_pred, reg_tgt)
        loss_qfl = self._loss_qfl(src, cls_tgt, score =score, weight=msk, avg_factor=msk.sum())

        output['loss_qfl'] = loss_qfl
        return output
    
    def loss_dfl(self, data, output, reg_tgt):
        src = data['out_coord']
        msk = torch.cat(data['pymid_msk'], dim=1)

        #dfl for start index
        src_s = src[:, :, 0]
        reg_tgt_s = reg_tgt[:, :, 0]
        loss_dfl_s = self._loss_dfl(src_s, reg_tgt_s, weight=msk, avg_factor=msk.sum())

        #dfl for end index
        src_e = src[:, :, 1]
        reg_tgt_e = reg_tgt[:, :, 1]
        loss_dfl_e = self._loss_dfl(src_e, reg_tgt_e, weight=msk, avg_factor=msk.sum())

        loss_dfl = loss_dfl_s + loss_dfl_e
        output['loss_dfl'] = loss_dfl
        return loss_dfl


    def loss_cls(self, data, output, cls_tgt):
        src = data['out_class'].squeeze(-1)
        msk = torch.cat(data['pymid_msk'], dim=1)
        loss_cls = self._loss_cls(src, cls_tgt, weight=msk, avg_factor=msk.sum())

        output['loss_cls'] = loss_cls
        return output

    def loss_reg(self, data, output, cls_tgt, reg_tgt):
        src = data['out_coord']
        msk = cls_tgt.unsqueeze(2).repeat(1, 1, 2).bool()

        loss_reg = self._loss_reg(src, reg_tgt, weight=msk, avg_factor=msk.sum())

        output['loss_reg'] = loss_reg
        return output

    def loss_sal(self, data, output):
        video_emb = data['video_emb']
        query_emb = data['query_emb']
        video_msk = data['video_msk']

        saliency = data['saliency']
        pos_clip = data['pos_clip'][:, 0]

        output['loss_sal'] = self._loss_sal(video_emb, query_emb, video_msk, saliency,
                                            pos_clip)
        return output

    def forward(self, data, output):
        if self._loss_reg is not None:
            cls_tgt, reg_tgt = self.get_target(data)
            output = self.loss_reg(data, output, cls_tgt, reg_tgt)
        else:
            cls_tgt = data['saliency']

        if self._loss_cls is not None:
            output = self.loss_cls(data, output, cls_tgt)

        if self._loss_sal is not None:
            output = self.loss_sal(data, output)

        if self._loss_qfl is not None:
            cls_tgt, reg_tgt = self.get_target(data)
            output = self.loss_qfl(data, output, cls_tgt, reg_tgt)
        
        if self._loss_dfl is not None:
            output = self.loss_dfl(data, output, reg_tgt)

        return output
