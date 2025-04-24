# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/SSD1/minseok/MR_HD/FlashVTG/')
from blocks.utils import weight_reduce_loss

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

    disl = disl.clamp(0, C - 1)
    disr = disr.clamp(0, C - 1)

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

    def forward(self, sim_score, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(sim_score.size(0), device=sim_score.device)
        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale).to(sim_score.device)
        i_sim = sim_score * scale
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


class MarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, saliency_scores, pos_indices, neg_indices):
        batch_size = saliency_scores.size(0)
        num_pairs = pos_indices.size(1)

        batch_indices = torch.arange(batch_size, device=saliency_scores.device)
        pos_scores = torch.stack([
            saliency_scores[batch_indices, pos_indices[:, i]] for i in range(num_pairs)
        ], dim=1)
        neg_scores = torch.stack([
            saliency_scores[batch_indices, neg_indices[:, i]] for i in range(num_pairs)
        ], dim=1)

        margin_loss = torch.clamp(self.margin + neg_scores - pos_scores, min=0)
        return margin_loss.sum() / (batch_size * num_pairs)


class RankingContrastiveLoss(nn.Module):
    def __init__(self, tau=0.5, ranks=12):
        super().__init__()
        self.tau = tau
        self.ranks = ranks

    def forward(self, scores, contrast_labels, mask):
        loss_total = 0.
        for rand_idx in range(1, self.ranks):
            drop_mask = ~(contrast_labels > 100)
            pos_mask = contrast_labels >= rand_idx
            if pos_mask.sum() == 0:
                continue
            batch_drop_mask = pos_mask.sum(1) > 0

            dropped_scores = scores * drop_mask / self.tau + ~drop_mask * -1e+3
            logits = dropped_scores - dropped_scores.max(dim=1, keepdim=True)[0]
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

            mean_log_prob_pos = (pos_mask * log_prob * mask).sum(1) / (pos_mask.sum(1) + 1e-6)
            loss = - mean_log_prob_pos * batch_drop_mask
            loss_total += loss.mean()

        return loss_total / self.ranks


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, logits, labels):
        logits = logits.reshape(-1)
        labels = labels.reshape(-1)
        return self.criterion(logits, labels)


class SetCriterion(nn.Module):
    """ This class computes the loss."""

    def __init__(self, weight_dict, eos_coef, losses, saliency_margin=1, args=None):
        """ Create the criterion."""
        super().__init__()
        self.args=args
        self.weight_dict = weight_dict
        self.losses = losses
        self.saliency_margin = saliency_margin
        self.device = args.device

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1

        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.args.device)
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none').to(self.args.device)
        self.bce_criterion = nn.BCELoss(reduction='none')
        self.SampledNCELoss = SampledNCELoss().to(self.args.device)

    def norm(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x
    
    def get_target_single(self, point, gt_bnd, gt_cls):
        num_pts, num_gts = point.size(0), gt_bnd.size(0)

        lens = gt_bnd[:, 1] - gt_bnd[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        gt_seg = gt_bnd[None].expand(num_pts, num_gts, 2)
        s = point[:, 0, None] - gt_seg[:, :, 0] # [num_pts, num_gts] 
        e = gt_seg[:, :, 1] - point[:, 0, None] # [num_pts, num_gts] 
        r_tgt = torch.stack((s, e), dim=-1) # [num_pts, num_gts, 2]

        if self.args.sample_radius > 0:
            center = (gt_seg[:, :, 0] + gt_seg[:, :, 1]) / 2
            t_mins = center - point[:, 3, None] * self.args.sample_radius
            t_maxs = center + point[:, 3, None] * self.args.sample_radius
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
            if self.args.use_dfl:
                num_bins = self.args.num_bins
                bin_size = self.args.sample_radius / (num_bins - 1)
                r_tgt = r_tgt.clamp(min=0.0, max=self.args.sample_radius - 1e-8)
                r_tgt = r_tgt / bin_size
                r_tgt = torch.where(r_tgt >= num_bins - 1, r_tgt - 1e-3, r_tgt)

            cls_tgt.append(c_tgt)
            reg_tgt.append(r_tgt)

        cls_tgt = torch.stack(cls_tgt)
        reg_tgt = torch.stack(reg_tgt)

        return cls_tgt, reg_tgt

    def get_reg_from_cls(self, logits, num_bins):
        """
        logits: [B, N, num_bins]
        returns: [B, N], expected value per token
        """
        probs = F.softmax(logits, dim=-1)
        bins = torch.arange(num_bins, dtype=probs.dtype, device=probs.device).view(1, 1, -1)  # shape [1, 1, C]
        expected = (probs * bins).sum(-1)
        return expected

    def get_iou(self, point, reg_pred, reg_tgt):
        """
        Args:
            point:    [N, 4], 마지막 dim은 (center, min_reg, max_reg, stride)
            reg_pred: [B, N, 2*num_bins] if DFL, [B, N, 2] if regression
            reg_tgt:  [B, N, 2]
        Returns:
            iou: [B, N]
        """
        center = point[:, 0][None, :]  # [1, N]
        stride = point[:, 3][None, :]  # [1, N]

        if self.args.use_dfl:
            num_bins = self.args.num_bins
            start_logits = reg_pred[:, :, :num_bins]
            end_logits   = reg_pred[:, :, num_bins:]
            start = self.get_reg_from_cls(start_logits, num_bins)
            end   = self.get_reg_from_cls(end_logits,   num_bins)
        else:
            start = reg_pred[:, :, 0]
            end   = reg_pred[:, :, 1]

        # Predicted boundary
        pred_start = center - start * stride
        pred_end   = center + end * stride

        # GT boundary
        gt_start = center - reg_tgt[:, :, 0] * stride
        gt_end   = center + reg_tgt[:, :, 1] * stride

        # IoU calculation
        inter_left  = torch.max(pred_start, gt_start)
        inter_right = torch.min(pred_end, gt_end)
        inter = (inter_right - inter_left).clamp(min=0)

        union_left  = torch.min(pred_start, gt_start)
        union_right = torch.max(pred_end, gt_end)
        union = (union_right - union_left).clamp(min=1e-6)

        iou = inter / union
        return iou



    def loss_phrase(self, outputs, targets, r=0.3, log=True):
        self.r = r
        attw = outputs["sqan_att"] # [B,num_att,N]
        NA = attw.size(1)

        attw_T = torch.transpose(attw, 1, 2).contiguous()

        I = torch.eye(NA).unsqueeze(0).type_as(attw) * self.r
        P = torch.norm(torch.bmm(attw, attw_T) - I, p="fro", dim=[1,2], keepdim=True)
        da_loss = (P**2).mean()

        return {"loss_phrase": da_loss}

    def loss_labels(self, outputs, targets, log=True):
        sal_score = targets["saliency_all_labels"]
        conf = outputs["out_class"][:, :sal_score.shape[1], 0]

        norm_sal_score = self.norm(sal_score)
        norm_conf = self.norm(conf)
        losses = F.mse_loss(norm_sal_score, norm_conf)
        return {"loss_label": losses}

    def loss_saliency(self, outputs, targets, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        
        if outputs["saliency_scores_neg"] is not None:
            vid_token_mask = outputs["video_msk"]
            real_neg_mask = outputs["real_neg_mask"]
            saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
            loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            margin_loss = MarginRankingLoss(margin=self.saliency_margin)
            ranking_loss = RankingContrastiveLoss(tau=0.5, ranks=12)
            bce_loss = BinaryCrossEntropyLoss()

            # for saliency scores
            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]
            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1. - realneg_vid_token_mask) * -1e+3

            rank_loss_saliency = ranking_loss(realneg_saliency_scores, realneg_saliency_contrast_label, realneg_vid_token_mask)

            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)

            margin_loss_saliency = margin_loss(saliency_scores, pos_indices, neg_indices)

            loss_saliency = margin_loss_saliency + loss_neg_pair + rank_loss_saliency
            # for t2vattnvalues

        if outputs["t2vattnvalues_neg"] is not None:
            saliency_scores_neg = outputs["t2vattnvalues_neg"].clone()  # (N, L)
            loss_neg_pair_attn = (- torch.log(1. - saliency_scores_neg) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()    
        saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        # real neg
        realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
        realneg_saliency_contrast_label = torch.cat(
            [saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
        realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
        realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (
                    1. - realneg_vid_token_mask) * -1e+3

        rank_loss_t2v = ranking_loss(realneg_saliency_scores, realneg_saliency_contrast_label, realneg_vid_token_mask)
        margin_loss_t2v = margin_loss(saliency_scores, pos_indices, neg_indices)
        # bce for t2v (MR loss)
        saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
        logits = saliency_scores.reshape(-1)
        labels_x = saliency_binary_label.reshape(-1)
        bce_loss_t2v = bce_loss(logits, labels_x)

        loss_saliency_t2v = margin_loss_t2v + loss_neg_pair_attn + rank_loss_t2v + bce_loss_t2v
        loss_saliency = loss_saliency + (loss_saliency_t2v * self.args.lw_wattn)

        return {"loss_saliency": loss_saliency}

    def loss_sal(self, outputs, targets, log=True):

        video_emb = outputs['video_emb']
        query_emb = outputs['query_emb']
        video_msk = outputs['video_msk']
        saliency = targets["saliency_all_labels"]
        pos_clip = targets["saliency_pos_labels"][:, 0]

        sampled_nce = SampledNCELoss()
        loss_sal = sampled_nce(video_emb, query_emb, video_msk, saliency, pos_clip)
        return {"loss_sal": loss_sal}

    def loss_reg(self, outputs, targets, log=True):
        cls_tgt, reg_tgt = self.get_target(outputs)
        pred = outputs["out_coord"]
        msk = cls_tgt.unsqueeze(2).repeat(1, 1, 2).bool()
        target = reg_tgt

        if self.args.use_dfl == False:
            loss_reg = F.l1_loss(pred[msk], target[msk], reduction='mean')
        else:
            dfl_loss_fn = DistributionFocalLoss()
            num_bins = self.args.num_bins
            start_logits = pred[:, :, :num_bins]
            end_logits   = pred[:, :, num_bins:]
            start_label  = reg_tgt[:, :, 0]
            end_label    = reg_tgt[:, :, 1]
            msk = cls_tgt.bool()

            loss_start = dfl_loss_fn(start_logits, start_label, weight=msk, avg_factor=msk.sum())
            loss_end   = dfl_loss_fn(end_logits,   end_label,   weight=msk, avg_factor=msk.sum())
            loss_reg = (loss_start + loss_end) * 0.5

        return {"loss_reg": loss_reg}

    def loss_cls(self, outputs, targets, log=True):
        alpha=-1
        gamma=2.0
        cls_tgt, reg_tgt = self.get_target(outputs)
        pred = outputs["out_class"].squeeze(-1)
        msk = torch.cat(outputs["pymid_msk"], dim=1)
        target = cls_tgt

        p = pred.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t)**gamma)

        if alpha >= 0:
            alpha_t = alpha * target + (1 - alpha) * (1 - target)
            loss = alpha_t * loss

        loss = loss * msk
        loss = loss.sum() / msk.sum()
        return {"loss_cls": loss}

    def loss_qfl(self, outputs, targets, log=True):
        quality_focal_loss = QualityFocalLoss()
        cls_tgt, reg_tgt = self.get_target(outputs)
        score = self.get_iou(outputs["point"], outputs["out_coord"], reg_tgt)
        msk = torch.cat(outputs["pymid_msk"], dim=1)

        loss = quality_focal_loss(
            outputs["out_class"].squeeze(-1),
            cls_tgt,
            score,
            weight=msk,
            avg_factor=msk.sum()
        )
        
        return {"loss_qfl": loss}
    
    def loss_dfl(self, outputs, targets, log=True):
        dfl_loss_fn = DistributionFocalLoss()
        cls_tgt, reg_tgt = self.get_target(outputs)
        msk = torch.cat(outputs["pymid_msk"], dim=1)
        
        # out_coord: [B, N, 2*num_bins]
        num_bins = self.args.num_bins
        start_logits = outputs["out_coord"][:, :, :num_bins]
        end_logits   = outputs["out_coord"][:, :, num_bins:]
        start_label  = reg_tgt[:, :, 0]
        end_label    = reg_tgt[:, :, 1]

        loss_start = dfl_loss_fn(start_logits, start_label, weight=msk, avg_factor=msk.sum())
        loss_end   = dfl_loss_fn(end_logits,   end_label,   weight=msk, avg_factor=msk.sum())

        return {"loss_dfl": (loss_start + loss_end) * 0.5}


    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "sal": self.loss_sal,
            "phrase": self.loss_phrase,
            "cls": self.loss_cls,
            "reg": self.loss_reg,
            "qfl": self.loss_qfl,

        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, **kwargs)

    def update_weight_dict(self, tgt_epoch, curr_epoch, loss, weight):
        if curr_epoch == tgt_epoch:
            self.weight_dict.update({loss: weight})

    def extract_relevant_windows(self, data_list):
        all_windows = [instance['relevant_windows'] for instance in data_list]
        max_len = max(len(windows) for windows in all_windows)

        padded_windows = []
        for windows in all_windows:
            new_windows = windows.copy()  
            while len(new_windows) < max_len:
                new_windows.append([float('inf'), float('inf')])
            padded_windows.append(new_windows)
        
        result_tensor = torch.tensor(padded_windows, dtype=torch.float32)
        
        return result_tensor

    def forward(self, batch, curr_epoch, outputs, targets):
        """ This performs the loss computation."""
        losses = {}
        outputs["boundary"] = self.extract_relevant_windows(batch[0]).to(self.device) if batch[0][0]['relevant_windows'] != None else None
        outputs["saliency"] = targets["saliency_all_labels"]
        outputs["pos_clip"] = targets["saliency_pos_labels"][:, 0].unsqueeze(1)
        outputs["label"] = batch[0]
        outputs["fps"] = targets["fps"]

        # Compute all the requested losses
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        
        #self.update_weight_dict(self.args.tgt_epoch, curr_epoch, "loss_sal", 0.0)
        #self.update_weight_dict(30, curr_epoch, "loss_cls", 0.0)
        #self.update_weight_dict(30, curr_epoch, "loss_qfl", self.args.lw_cls)
        #self.update_weight_dict(30, curr_epoch, "loss_label", 0.0)
        
        return losses

class Parameter(nn.Parameter):
    """
    An :obj:`nn.Parameter` class that supports multiple inputs initializes the
    parameters using a scaled normal distribution.
    """

    def __new__(cls, *args, requires_grad=True, **kwargs):
        if torch.is_tensor(args[0]):
            data = args[0]
        elif isinstance(args[0], float):
            data = torch.Tensor([args[0]])
        elif isinstance(args[0], (list, tuple)):
            data = torch.randn(args[0], **kwargs) / args[0][-1]**0.5
        else:
            data = torch.randn(args, **kwargs) / args[-1]**0.5

        return torch.Tensor._make_subclass(cls, data, requires_grad)

