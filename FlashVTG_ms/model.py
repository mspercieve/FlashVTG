# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
FlashVTG model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from FlashVTG_ms.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from FlashVTG_ms.position_encoding import build_position_encoding, PositionEmbeddingSine
import math
from nncore.nn import build_model as build_adapter
from blocks.generator import PointGenerator
from LGI import Phrase_Generate, PhraseWeight, Phrase_Context, CrossAttention, AttentivePooling, Aggregate_Module

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res

class ConfidenceScorer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers=1, num_mlp_layers=3):
        super(ConfidenceScorer, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for i in range(num_conv_layers):
            if i == 0:
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            else:
                self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            self.activations.append(nn.ReLU(inplace=True))
        
        self.fc = MLP(out_channels, out_channels // 2, 1, num_layers=num_mlp_layers)
    
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        
        for conv, activation in zip(self.convs, self.activations):
            x = conv(x)
            x = activation(x)
        
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.fc(x)
        
        return x

class FlashVTG_ms(nn.Module):
    """ FlashVTG. """

    def __init__(self, transformer, position_embed, txt_position_embed, n_input_proj, input_dropout, txt_dim, vid_dim, aud_dim=0, use_txt_pos=False,
                strides=(1, 2, 4, 8),
                buffer_size=2048,
                max_num_moment=50,
                merge_cls_sal=True,
                pyramid_cfg=None,
                pooling_cfg=None,
                coord_head_cfg=None,
                args=None):
        """ Initializes the model."""
        super().__init__()
        self.args=args
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.PositionEmbeddingSine = PositionEmbeddingSine(hidden_dim, normalize=True)
        
        # input projection
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_word_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # set up dummy token
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm)

        # build muti-scale pyramid
        self.pyramid = build_adapter(pyramid_cfg, hidden_dim, strides)

        self.pooling = build_adapter(pooling_cfg, hidden_dim)
        self.class_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.coef = nn.Parameter(torch.ones(len(strides)))
        if args.use_dfl:
            self.coord_head = build_adapter(coord_head_cfg, hidden_dim, args.num_bins * 2)
        else:
            self.coord_head = build_adapter(coord_head_cfg, hidden_dim, 2)
        self.generator = PointGenerator(strides, buffer_size)
        self.max_num_moment = max_num_moment
        self.merge_cls_sal = merge_cls_sal
        self.args = args
        self.num_phrase = args.num_phrase
        self.x = nn.Parameter(torch.tensor(0.5))

        # build phrase embedding
        self.phrase_generate = Phrase_Generate(args.num_phrase, hidden_dim, args.nheads, args.dropout, args.phrase_layers)
        self.phrase_weight = PhraseWeight(args.hidden_dim)
        self.phrase_context = Phrase_Context(hidden_dim, args.nheads, args.dropout, args.context_layers)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_norm_neg = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, args.nheads, args.dropout)
        self.attentive_pool = AttentivePooling(hidden_dim)
        self.agg = Aggregate_Module(hidden_dim, args.dropout)
        self.fuse_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, targets=None):
        if vid is not None:
            _count = [v.count('_') for v in vid]
            if self.args.dset_name == 'hl':
                _position_to_cut = [find_nth(v, '_', _count[i]-1) for i, v in enumerate(vid)]
                ori_vid = [v[:_position_to_cut[i]] for i, v in enumerate(vid)]
            else:
                ori_vid = [v for v in vid]

        # Project inputs to the same hidden dimension
        src_glob, src_word = torch.split(src_txt, [1, src_txt.size(1)-1], dim=1)
        src_vid = self.input_vid_proj(src_vid)
        B, T, C = src_vid.shape
        src_glob = self.input_txt_proj(src_glob)
        src_word = self.input_word_proj(src_word)
        src_txt = torch.cat([src_glob, src_word], dim=1)  # [B, N, C]
        # Add type embeddings
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        # Add position embeddings

        pos_vid = self.position_embed(torch.cat([src_vid, src_vid], dim=-1), src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)
        # separate sentence and word
        _, src_word_msk = torch.split(src_txt_mask, [1, src_txt_mask.size(1)-1], dim=1)
        pos_glob, pos_word = torch.split(pos_txt, [1, pos_txt.size(1)-1], dim=1)

        # Phrase Generate
        phrase_emb, phrase_att = self.phrase_generate(src_txt, src_txt_mask) # [B, N, C]
        # key phrase score
        phrase_score = self.phrase_weight(phrase_emb, src_glob) # [B, N]
        phrase_score = phrase_score.unsqueeze(-1).unsqueeze(-1)
        context_emb = self.phrase_context(phrase_emb, src_vid, src_vid_mask) # [B, N, T, C]
        context_emb = (context_emb * phrase_score).sum(1).view(B,T,C) # [B,T,C]


        # Dummy Generate
        txt_dummy = self.dummy_rep_token.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1)
        src_txt_dummy = torch.cat([txt_dummy, src_glob], dim=1)


        src_txt_mask_dummy = torch.tensor([[True] * (self.args.num_dummies + 1)]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)

        pos_dummy = self.dummy_rep_pos.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1)
        pos_txt_dummy = torch.cat([pos_dummy, pos_glob], dim=1)
        src_txt_dummy = src_txt_dummy.permute(1, 0, 2) # (L, batch_size, d)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2) # (L, batch_size, d)

        memory = self.txtproj_encoder(src_txt_dummy, src_key_padding_mask=~(src_txt_mask_dummy.bool()), pos=pos_txt_dummy)
        dummy_token = memory[:self.args.num_dummies].permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)

        src_txt_dummy = torch.cat([dummy_token, src_glob], dim=1)
        src_txt_mask_dummy = torch.tensor([[True] * (self.args.num_dummies + 1)]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)

        src = torch.cat([src_vid, src_txt_dummy], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask_dummy], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1)

        video_length = src_vid.shape[1]

        # global text update
        vid_fuse, video_msk, pos_embed, attn_weights = self.transformer(src, context_emb, ~mask, pos, video_length=video_length)
        video_emb = vid_fuse.permute(1, 0, 2)  # (L, batch_size, d) -> (batch_size, L, d)
        # video_emb = self.agg(glob_emb, context_emb)
        memory_global = video_emb.mean(1)

        proj1_result = self.saliency_proj1(video_emb)
        proj2_result = self.saliency_proj2(memory_global).unsqueeze(1)
        intermediate_result = proj1_result * proj2_result  # (bsz, L, d)
        saliency_scores = torch.sum(intermediate_result, dim=-1) / np.sqrt(self.hidden_dim)  # (bsz, L)
        video_msk = (~video_msk).int()
        pymid, pymid_msk = self.pyramid(
            video_emb, video_msk, return_mask=self.training == True
        )
        point = self.generator(pymid)

        with torch.autocast("cuda", enabled=False):
            video_emb = video_emb.float()
            query_emb = src_glob.float()
            #query_emb = self.pooling(src_txt.float(), src_txt_mask)
            
            out_class = [self.class_head(e.float()) for e in pymid]
            out_class = torch.cat(out_class, dim=1)
            if self.coord_head is not None:
                out_coord = [
                    self.coord_head(e.float()).exp() * self.coef[i]
                    for i, e in enumerate(pymid)
                ]
                out_coord = torch.cat(out_coord, dim=1)
            else:
                out_coord = None 

            bs, t = src_vid.shape[0], src_vid.shape[1]
            output = dict(_avg_factor=bs)
            output["saliency_scores"] = saliency_scores
            output["t2vattnvalues"] = (attn_weights[:,:,self.args.num_dummies:] * (src_txt_mask.unsqueeze(1).repeat(1, video_length, 1))).sum(2)
            output["t2vattnvalues"] = torch.clamp(output["t2vattnvalues"], 0, 1)
            output["sqan_att"] = phrase_att
            if self.training == True:

                output["point"] = point
                output["video_emb"] = video_emb
                output["query_emb"] = query_emb
                output["video_msk"] = video_msk
                output["pymid_msk"] = pymid_msk
                output["out_class"] = out_class
                output["out_coord"] = out_coord 
                '''
                boundarys = []
                out_class = out_class.sigmoid() # [bs, (1+1/2+1/4+1/8)L, 1]
                for idx, boundary in enumerate(out_coord):
                    boundary = boundary.clone()

                    boundary[:, 0] = boundary[:, 0] * -1
                    boundary = boundary * point[:, 3, None].repeat(1, 2)
                    boundary = boundary + point[:, 0, None].repeat(1, 2)
                    boundary = boundary / (1/self.args.clip_length)
                    boundary = torch.cat((boundary, out_class[idx]), dim=-1)  

                    _, inds = out_class[idx, :, 0].sort(descending=True)
                    boundary = boundary[inds[:]]
                    boundarys.append(boundary)

                boundarys = torch.stack(boundarys, dim=0)
                output["pred_spans"] = boundarys
                '''

            if self.training == False:
                assert bs == 1, "batch size larger than 1 is not supported for inference"
                out_class = out_class.sigmoid()

                output["_out"] = dict(label=targets.get("label", [None])[0])
                output["_out"]["video_msk"] = video_msk
                output["_out"]["saliency"] = saliency_scores[0]

                if  self.args.use_dfl==False and self.coord_head is not None:
                    boundary = out_coord[0]
                    boundary[:, 0] *= -1
                    boundary *= point[:, 3, None].repeat(1, 2)
                    boundary += point[:, 0, None].repeat(1, 2)  
                    boundary /= 1/self.args.clip_length
                    boundary = torch.cat((boundary, out_class[0]), dim=-1)  

                    _, inds = out_class[0, :, 0].sort(descending=True)
                    boundary = boundary[inds[: self.max_num_moment]]  

                    output["_out"]["boundary"] = boundary

                elif self.args.use_dfl and self.coord_head is not None:
                    boundary = out_coord[0]  # shape: (N, num_bins * 2)
                    num_bins = self.args.num_bins
                    bin_size = self.args.sample_radius / (num_bins - 1)

                    start_logits = boundary[:, :num_bins]     # (N, num_bins)
                    end_logits = boundary[:, num_bins:]       # (N, num_bins)

                    start_prob = F.softmax(start_logits, dim=-1)  # (N, num_bins)
                    end_prob = F.softmax(end_logits, dim=-1)      # (N, num_bins)

                    bin_centers = torch.linspace(0, self.args.sample_radius, steps=num_bins, device=boundary.device)  # (num_bins,)

                    start = torch.sum(start_prob * bin_centers[None, :], dim=-1)  # (N,)
                    end = torch.sum(end_prob * bin_centers[None, :], dim=-1)      # (N,)
                    boundary = torch.stack([start, end], dim=-1)  # (N, 2)
                    boundary[:, 0] *= -1
                    boundary = boundary * point[:, 3, None].repeat(1, 2)
                    boundary = boundary + point[:, 0, None].repeat(1, 2)
                    boundary = boundary / (1 / self.args.clip_length)

                    boundary = torch.cat((boundary, out_class[0]), dim=-1)
                    _, inds = out_class[0, :, 0].sort(descending=True)
                    boundary = boundary[inds[: self.max_num_moment]]
                    output["_out"]["boundary"] = boundary

        if self.training == True and self.args.use_neg:
            ### Neg Pairs ###
            neg_vid = ori_vid[1:] + ori_vid[:1] 
            real_neg_mask = torch.Tensor(element_wise_list_equal(ori_vid, neg_vid)).to(src_txt_dummy.device)
            real_neg_mask = real_neg_mask == False
            if real_neg_mask.sum() != 0:
                # phrase neg
                phrase_emb_neg = torch.cat([phrase_emb[1:], phrase_emb[0:1]], dim=0)
                src_vid_neg = src_vid[real_neg_mask]
                vid_mask_neg = src_vid_mask[real_neg_mask]
                phrase_emb_neg = phrase_emb_neg[real_neg_mask]

                src_glob_neg = torch.cat([src_glob[1:], src_glob[0:1]], dim=0)
                phrase_score_neg = self.phrase_weight(phrase_emb_neg, src_glob_neg[real_neg_mask]) # [B, N]
                phrase_score_neg = phrase_score_neg.unsqueeze(-1).unsqueeze(-1)

                context_emb_neg = self.phrase_context(phrase_emb_neg, src_vid_neg, vid_mask_neg) # [B, N, T, C]
                context_emb_neg = (context_emb_neg * phrase_score_neg).sum(1) # [B, T, C]
                # dummy neg
                src_txt_dummy_neg = torch.cat([src_txt_dummy[1:], src_txt_dummy[0:1]], dim=0)
                src_txt_mask_dummy_neg = torch.cat([src_txt_mask_dummy[1:], src_txt_mask_dummy[0:1]], dim=0)
                src_dummy_neg = torch.cat([src_vid, src_txt_dummy_neg], dim=1)
                mask_dummy_neg = torch.cat([src_vid_mask, src_txt_mask_dummy_neg], dim=1).bool()
                pos_neg = pos.clone() 

                mask_dummy_neg = mask_dummy_neg[real_neg_mask] 
                src_dummy_neg = src_dummy_neg[real_neg_mask] 
                pos_neg = pos_neg[real_neg_mask]
                src_txt_mask_dummy_neg = src_txt_mask_dummy_neg[real_neg_mask]
                
                memory_neg, video_msk, pos_embed, attn_weights_neg= self.transformer(src_dummy_neg, context_emb_neg, ~mask_dummy_neg, pos_neg, video_length=video_length)
                memory_neg = memory_neg.permute(1, 0, 2)  # (L, batch_size, d) -> (batch_size, L, d)
                vid_mem_neg = memory_neg
                #vid_mem_neg = self.agg(memory_neg, context_emb_neg)
                txt_glob_neg = torch.cat([src_glob[1:], src_glob[0:1]], dim=0)
                memory_global_neg = vid_mem_neg.mean(1).clone()
                proj1_result_neg = self.saliency_proj1(vid_mem_neg)
                proj2_result_neg = self.saliency_proj2(memory_global_neg)
                proj2_result_neg = proj2_result_neg.unsqueeze(1)
                intermediate_result_neg = proj1_result_neg * proj2_result_neg
                saliency_scores_neg = torch.sum(intermediate_result_neg, dim=-1) / np.sqrt(self.hidden_dim)
                output["saliency_scores_neg"] = saliency_scores_neg
                output["src_txt_mask_neg"] = src_txt_mask_dummy_neg

                output["t2vattnvalues_neg"] = (attn_weights_neg[:, :, self.args.num_dummies:] * (src_txt_mask_dummy_neg[:, self.args.num_dummies:].unsqueeze(1).repeat(1, video_length, 1))).sum(2)
                output["t2vattnvalues_neg"] = torch.clamp(output["t2vattnvalues_neg"], 0, 1) 
            else:
                output["saliency_scores_neg"] = None
                output["t2vattnvalues_neg"] = None
            output["real_neg_mask"] = real_neg_mask
            output["dummy_tokens"] = dummy_token
        else:
            output["saliency_scores_neg"] = None
            output["t2vattnvalues_neg"] = None
            output["real_neg_mask"] = None
            output["dummy_tokens"] = dummy_token

        return output

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


class SampledNCELoss(nn.Module):

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=False,
                 direction=('row', 'col')):
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

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, direction={}, loss_weight={}'
                .format(self.temperature, self.max_scale, self.learnable, self.direction,
                        self.loss_weight))

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(video_emb.size(0), device=video_emb.device)

        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale)
        i_sim = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale # (B, T)
        i_sim = i_sim + torch.where(loss_msk > 0, .0, float('-inf'))

        loss = 0

        if 'row' in self.direction:
            i_met = F.log_softmax(i_sim, dim=1)[batch_inds, pos_clip] 
            loss = loss - i_met.sum() / i_met.size(0)

        if 'col' in self.direction:
            j_sim = i_sim.t()
            j_met = F.log_softmax(j_sim, dim=1)[pos_clip, batch_inds]
            loss = loss - j_met.sum() / j_met.size(0)

        return loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model1(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = FlashVTG_ms(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        input_dropout=args.input_dropout,
        n_input_proj=args.n_input_proj,
        strides=args.cfg.model.strides,
        buffer_size=args.cfg.model.buffer_size,
        max_num_moment=args.cfg.model.max_num_moment,
        pyramid_cfg=args.cfg.model.pyramid_cfg,
        pooling_cfg=args.cfg.model.pooling_cfg,
        coord_head_cfg=args.cfg.model.coord_head_cfg,
        args=args
    )

    weight_dict = {"loss_label": args.label_loss_coef,
                    #"loss_label": 0,
                   "loss_saliency": args.lw_saliency,
                   'loss_reg': args.lw_reg,
                   "loss_cls": args.lw_cls,
                   "loss_sal": args.lw_sal,
                   "loss_phrase": args.lw_phrase,
                   "loss_qfl": 0,
                   }

    losses = ["saliency", 'labels', 'phrase', 'sal', 'reg', 'cls', 'qfl']
    #losses = ["labels", "phrase"]
    from FlashVTG_ms.loss import SetCriterion
    criterion = SetCriterion(
        weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, saliency_margin=args.saliency_margin, args=args
    )
    criterion.to(device)
    return model, criterion
