import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('utils/')
import net_utils
import math
from .position_encoding import PositionEmbeddingSine
from einops import rearrange, repeat
from natten.functional import natten2dqkrpb, natten2dav
from .transformer import TransformerEncoderLayer, TransformerEncoder

class Attention(nn.Module):
    def __init__(self, kdim, cdim, hdim, drop_p):
        super(Attention, self).__init__()
        att_hdim = hdim
        # layers
        self.key2att = nn.Linear(kdim, att_hdim)
        self.feat2att = nn.Linear(cdim, att_hdim)
        self.to_alpha = nn.Linear(att_hdim, 1)
        self.drop = nn.Dropout(drop_p)

    def forward(self, key, feats, feat_masks=None, return_weight=True):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2, "{} != 2".format(len(key.size()))
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dnsion 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # compute attention weights
        logits = self.compute_att_logits(key, feats, feat_masks) # [B,A]
        weight = self.drop(F.softmax(logits, dim=1))             # [B,A]

        # compute weighted sum: bmm working on (B,1,A) * (B,A,D) -> (B,1,D)
        att_feats = torch.bmm(weight.unsqueeze(1), feats).squeeze(1) # B * D
        if return_weight:
            return att_feats, weight
        return att_feats

    def compute_att_logits(self, key, feats, feat_masks=None):
        """ Compute attention weights
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dnsion 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)
        A = feats.size(1)

        # embedding key and feature vectors
        att_f = net_utils.apply_on_sequence(self.feat2att, feats)   # B * A * att_hdim
        att_k = self.key2att(key)                                   # B * att_hdim
        att_k = att_k.unsqueeze(1).expand_as(att_f)                 # B * A * att_hdim

        # compute attention weights
        dot = torch.tanh(att_f + att_k)                             # B * A * att_hdim
        alpha = net_utils.apply_on_sequence(self.to_alpha, dot)     # B * A * 1
        alpha = alpha.view(-1, A)                                   # B * A
        if feat_masks is not None:
            alpha = alpha.masked_fill(feat_masks.float().eq(0), -1e9)

        return alpha


class SequentialQueryAttention(nn.Module):
    def __init__(self, num_phrase, qdim):
        super(SequentialQueryAttention, self).__init__()

        self.nse = num_phrase
        self.qdim = qdim # 512
        self.global_emb_fn = nn.ModuleList( # W_q^(n) in Eq. (4)
                [nn.Linear(self.qdim, self.qdim) for i in range(self.nse)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2*self.qdim, self.qdim), # W_g in Eq. (4)
            nn.ReLU()
        ])
        # 기존 additive style attention (나중에 사용할 수 있도록 주석 처리)
        # self.att_fn = Attention(qdim, qdim, qdim, 0.1)
        
        # Cross attention 모듈 사용
        self.cross_att = CrossAttention(qdim, nheads=8, dropout=0.1)

    def forward(self, q_feats, w_feats, w_mask=None):
        """ extract N (=nse) semantic entity features from query
        Args:
            q_feats: sentence-level feature; [B,qdim]
            w_feats: phrase-level features; [B,L,qdim]
            w_mask: mask for effective phrases; [B,L]
        Returns:
            se_feats: semantic entity features; [B,N,qdim]
            se_attw: attention weight over phrases; [B,N,L]
        """
        q_feats = q_feats.squeeze(1)
        B = w_feats.size(0)
        prev_se = w_feats.new_zeros(B, self.qdim)
        se_feats, se_attw = [], []
        
        # compute semantic entity features sequentially
        for n in range(self.nse):
            # perform Eq. (4)
            q_n = self.global_emb_fn[n](q_feats) # [B,qdim] -> [B,qdim]
            g_n = self.guide_emb_fn(torch.cat([q_n, prev_se], dim=1)) # [B,2*qdim] -> [B,qdim]
            
            # Cross attention 수행
            g_n = g_n.unsqueeze(1)  # [B, 1, C]
            att_f, att_w = self.cross_att(g_n, w_feats, w_mask)
            att_f = att_f.squeeze(1)  # [B, C]
            
            # 기존 additive style attention (주석 처리)
            # att_f, att_w = self.att_fn(g_n, w_feats, w_mask)

            prev_se = att_f
            se_feats.append(att_f)
            se_attw.append(att_w.squeeze(1))  # [B, L]

        return torch.stack(se_feats, dim=1), torch.stack(se_attw, dim=1)


class Phrase_Generate(nn.Module):
    def __init__(self, num_phrase, hdim, num_heads, drop_p, num_layers):
        super(Phrase_Generate, self).__init__()
        self.num_layers = num_layers
        self.num_phrase = num_phrase
        self.hdim = hdim
        
        # Word-Video Cross Attention을 위한 projection
        self.word_proj = nn.Linear(hdim, hdim)
        self.video_proj = nn.Linear(hdim, hdim)
        
        # Phrase formation을 위한 cross attention
        self.phrase_att = nn.ModuleList([CrossAttention(hdim, num_heads, drop_p) for _ in range(num_layers)])
        
        # Position embedding
        self.pos = PositionEmbeddingSine(hdim)
        
        # EOS token
        self.eos_slot = nn.Parameter(torch.randn(1, 1, hdim))
        
        # Word selection을 위한 hyperparameter
        self.min_word_distance = 2  # 선택된 단어들 사이의 최소 거리
        
    def compute_word_importance(self, word_feats, video_feats, video_mask):
        """단어의 중요도를 계산"""
        B, L, C = word_feats.shape
        T = video_feats.shape[1]
        
        # Project features
        word_proj = self.word_proj(word_feats)  # [B, L, C]
        video_proj = self.video_proj(video_feats)  # [B, T, C]
        
        # Compute similarity matrix
        sim_matrix = torch.bmm(word_proj, video_proj.transpose(1, 2))  # [B, L, T]
        
        # Apply mask
        if video_mask is not None:
            sim_matrix = sim_matrix.masked_fill(~video_mask.unsqueeze(1), float('-inf'))
        
        # Softmax along T dimension
        attn_weights = F.softmax(sim_matrix, dim=2)  # [B, L, T]
        
        # Compute entropy
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-6), dim=2)  # [B, L]
        
        return entropy, attn_weights
    
    def select_important_words(self, entropy, word_feats, word_mask):
        """중요한 단어들을 선택"""
        B, L = entropy.shape
        selected_indices = []
        selected_feats = []
        
        for b in range(B):
            # Get valid word indices
            valid_indices = torch.where(word_mask[b])[0]
            valid_entropy = entropy[b, valid_indices]
            
            # Sort by entropy
            sorted_indices = torch.argsort(valid_entropy, descending=True)
            sorted_valid_indices = valid_indices[sorted_indices]
            
            # Select words with minimum distance constraint
            curr_indices = []
            for idx in sorted_valid_indices:
                if not curr_indices or min(abs(idx - i) for i in curr_indices) >= self.min_word_distance:
                    curr_indices.append(idx)
                    if len(curr_indices) == self.num_phrase:
                        break
            
            # Pad if necessary
            while len(curr_indices) < self.num_phrase:
                curr_indices.append(curr_indices[-1])
            
            selected_indices.append(curr_indices)
            selected_feats.append(word_feats[b, curr_indices])
        
        return torch.stack(selected_feats), torch.tensor(selected_indices, device=entropy.device)
    
    def forward(self, txt_emb, txt_mask, video_feats=None, video_mask=None):
        """
        Args:
            txt_emb: [B, L, C] - 전체 텍스트 임베딩
            txt_mask: [B, L] - 텍스트 마스크
            video_feats: [B, T, C] - 비디오 임베딩
            video_mask: [B, T] - 비디오 마스크
        """
        B, L, C = txt_emb.shape
        stc_emb, word_emb = torch.split(txt_emb, [1, L-1], dim=1)
        word_mask = txt_mask[:, 1:]
        
        # Position embedding 적용
        word_pos = self.pos(word_emb, word_mask)
        word_pe = word_emb + word_pos
        
        # 1. 단어 중요도 계산
        entropy, word_video_attn = self.compute_word_importance(word_pe, video_feats, video_mask)
        
        # 2. 중요 단어 선택
        phrase_slot, selected_indices = self.select_important_words(entropy, word_pe, word_mask)
        
        # 3. Cross attention으로 구 형성
        slot_sim = None
        for i in range(self.num_layers):
            phrase_slot, slot_sim = self.phrase_att[i](phrase_slot, word_pe, word_mask)
        
        # EOS token 처리
        eos_slot = self.eos_slot.expand(B, 1, C)
        
        return phrase_slot, word_video_attn, slot_sim, eos_slot

class SlotAttention(nn.Module):
    def __init__(self, hdim, num_heads, drop_p=0.1):
        super(SlotAttention, self).__init__()
        self.nh = num_heads
        self.C = hdim
        self.norm = nn.LayerNorm(hdim)
        
        # Phrase-Word Cross Attention
        self.phrase_word_att = CrossAttention(hdim, num_heads, drop_p)
        
        # EOS-Phrase Cross Attention
        self.eos_phrase_att = CrossAttention(hdim, num_heads, drop_p)

    def forward(self, txt_feat, txt_mask, phrase_slot, eos_token):
        """
        Inputs:
            txt_feat: [B, L, C] - word features
            txt_mask: [B, L] - word mask (1: valid, 0: padding)
            phrase_slot: [B, N, C] - phrase slots
            eos_token: [B, 1, C] - learnable EOS token
        Returns:
            updated_phrase: [B, N, C] - updated phrase slots
            updated_eos: [B, 1, C] - updated EOS token
            slot_sim: [B, N, L] - attention weights
        """
        B, N, C = phrase_slot.shape
        
        # 1. Phrase-Word Cross Attention
        phrase_updated, slot_sim = self.phrase_word_att(phrase_slot, txt_feat, txt_mask)
        phrase_updated = self.norm(phrase_updated + phrase_slot)  # Residual connection
        
        # 2. EOS-Phrase Cross Attention
        eos_updated, _ = self.eos_phrase_att(eos_token, phrase_updated)
        eos_updated = self.norm(eos_updated + eos_token)  # Residual connection
        
        return phrase_updated, eos_updated, slot_sim

class LowRankDynamicConv(nn.Module):
    def __init__(self, hdim, num_phrase, rank=32, t_kernels=(1, 3, 5)):
        super().__init__()
        self.hdim = hdim
        self.num_phrase = num_phrase
        self.rank = rank
        self.t_kernels = t_kernels

        # EOS slot과 phrase를 결합하기 위한 projection
        self.combine_proj = nn.Sequential(
            nn.Linear(2 * hdim, 4 * hdim),
            nn.ReLU(),
            nn.Linear(4 * hdim, hdim * rank)
        )

        # for each kernel we now keep a separate output dimension
        self.out_dims = [
            hdim for i in range(len(t_kernels))
        ]

        # dynamic kernel parameters: (rank, out_dim, kernel_size)
        self.kernel_params = nn.ParameterDict({
            f'k{k}': nn.Parameter(torch.randn(rank, out_dim, k))
            for (i, k), out_dim in zip(enumerate(t_kernels), self.out_dims)
        })

        # final linear from sum(out_dims) back to hdim
        self.linear_out = nn.Linear(sum(self.out_dims), hdim)
        self.norm = nn.LayerNorm(hdim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def temporal_unfold(self, x, kernel_size, padding):
        B, T, N, C = x.shape
        pad = kernel_size // 2 if padding else 0
        x_padded = F.pad(x, (0,0,0,0,pad,pad))
        stride = x_padded.stride()
        return x_padded.as_strided(
            size=(B, T, kernel_size, N, C),
            stride=(stride[0], stride[1], stride[1], stride[2], stride[3])
        )

    def forward(self, context_emb, phrase_slot, eos_slot):
        B, T, N, C = context_emb.shape
        
        # EOS slot을 phrase_slot과 같은 크기로 확장
        eos_slot = eos_slot.expand(B, N, C)  # [B, N, C]
        
        # phrase_slot과 eos_slot을 결합
        combined = torch.cat([phrase_slot, eos_slot], dim=-1)  # [B, N, 2C]
        
        # 비선형 projection을 통한 feature 추출
        phrase_proj = self.combine_proj(combined)  # [B, N, C*r]
        phrase_proj = rearrange(phrase_proj, 'b n (c r) -> b n c r', c=C, r=self.rank)

        outputs = []
        for i, k in enumerate(self.t_kernels):
            out_dim = self.out_dims[i]
            kernel_param = self.kernel_params[f'k{k}']  # (r, out_dim, k)

            # dynamic kernel: [B, N, C, k, out_dim]
            dyn_kernel = torch.einsum('bncr, rdk -> bnckd', phrase_proj, kernel_param)

            # local windows: [B, T, k, N, C]
            x_window = self.temporal_unfold(context_emb, k, padding=True)

            # reshape for batched matmul
            x_window = rearrange(x_window, 'b t k n c -> b t (k n c)')
            dyn_kernel = rearrange(dyn_kernel, 'b n c k d -> b (k n c) d')

            # out: [B, T, out_dim]
            out = torch.bmm(x_window, dyn_kernel)
            outputs.append(out)

        # concat along channel: [B, T, sum(out_dims)]
        feat = torch.cat(outputs, dim=-1)

        # project back to hdim
        out = self.linear_out(feat)
        out = self.dropout(out)
        out = self.act(self.norm(out))

        return out


    
class PhraseContextLayer(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1):
        super(PhraseContextLayer, self).__init__()
        self.t_att = SelfAttention(hdim, nheads, dropout=dropout)

        self.fc_t = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm_t = nn.LayerNorm(hdim)

    def forward(self, context_emb, phrase_emb, vid_mask, shape):
        """
        context_emb: [B*N, T, C]
        """
        B, N, T, C = shape
        
        # T-axis self-attention (시간적 맥락)
        context_emb, _ = self.t_att(context_emb, vid_mask) # [B*N, T, C]
        t_update = self.fc_t(context_emb)
        context_emb = self.norm_t(context_emb + t_update)

        return context_emb


class Phrase_Context(nn.Module):
    def __init__(self, hdim, num_layers, nheads, dropout=0.1, num_phrase=3, rank=32, t_kernels=(1,3,5)):
        super(Phrase_Context, self).__init__()
        self.hdim = hdim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([PhraseContextLayer(hdim, nheads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.rank = rank

        self.product = HadamardProduct(hdim, hdim, hdim)
        self.pos = PositionEmbeddingSine(hdim)
        self.local_context = LowRankDynamicConv(hdim, num_phrase, rank, t_kernels)

    def forward(self, phrase_slot, vid_feat, vid_mask, eos_slot):
        """
        Args:
            phrase_slot: [B, N, C] phrase slots
            vid_feat: [B, T, C] video-level features
            vid_mask: [B, T] video mask
            eos_slot: [B, 1, C] EOS token
        Returns:
            updated_phrase: [B, T, N, C]
        """
        B, T, C = vid_feat.shape
        N = phrase_slot.shape[1]
        
        context_emb = self.product([phrase_slot, vid_feat]) # [ B, N, T, C]
        context_emb_out = context_emb
        context_emb = rearrange(context_emb, 'b n t c -> (b n) t c') # [B*N, T, C]
        
        vid_mask = repeat(vid_mask, 'b t -> (b n) t', n=N) # [B*N, T]
        pos = self.pos(context_emb, vid_mask) # [B*N, T, C]
        context_emb = context_emb + pos
        
        for layer in self.layers:
            context_emb = layer(context_emb, phrase_slot, vid_mask, (B, N, T, C))
        context_emb = rearrange(context_emb, '(b n) t c -> b t n c', b=B, n=N)
        context_refine_out = context_emb.permute(0, 2, 1, 3)
        context_agg = self.local_context(context_emb, phrase_slot, eos_slot)
        return context_agg, context_emb_out, context_refine_out

class HadamardProduct(nn.Module):
    def __init__(self, idim_1, idim_2, hdim):
        super(HadamardProduct, self).__init__()

        self.fc_1 = nn.Linear(idim_1, hdim)
        self.fc_2 = nn.Linear(idim_2, hdim)
        self.fc_3 = nn.Linear(hdim, hdim)
        self.norm = nn.LayerNorm(hdim)
        self.norm1 = nn.LayerNorm(hdim)
    def forward(self, inp):
        """
        Args:
            inp0: Phrase [B,N,C]
            inp1: Vid [B,T,C]
        """
        x1, x2 = inp[0], inp[1]
        x1 = torch.relu(self.fc_1(x1)).unsqueeze(2) # [B, N, 1, hdim]
        x2 = torch.relu(self.fc_2(x2)).unsqueeze(1) # [B, 1, T, hdim]
        x = self.norm(x1 * x2) # [B, N, T, hdim]
        return torch.relu(self.norm1(self.fc_3(x))) # [B, N, T, hdim]
    
class SelfAttention(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.hdim = hdim
        self.nheads = nheads
        self.att = nn.MultiheadAttention(hdim, nheads, batch_first=True, dropout=dropout)
        self.q_proj = nn.Linear(hdim, hdim)
        self.k_proj = nn.Linear(hdim, hdim)
        self.v_proj = nn.Linear(hdim, hdim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hdim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, C]
            mask: [B, L]
        """
        B, L, C = x.shape
        q = self.q_proj(x) # [B, L, C]
        k = self.k_proj(x) # [B, L, C]
        v = self.v_proj(x) # [B, L, C]

        if mask is None:
            update, attn = self.att(q, k, v)
        else:
            update, attn = self.att(q, k, v, key_padding_mask=~(mask.bool()))
        update = self.dropout(update)
        x = self.norm(x + update)
        return x, attn

class SelfAttention_Dynamicv2(nn.Module):
    """
    Self-Attention with Dynamic Projection v2
    Mixture of Experts (MoE) 방식으로 Q, K에 dynamic projection 적용
    """
    def __init__(self, hdim, nheads, k, dropout=0.1):
        super(SelfAttention_Dynamicv2, self).__init__()
        self.hdim = hdim
        self.nheads = nheads
        self.k = k

        self.att = nn.MultiheadAttention(hdim, nheads, batch_first=True, dropout=dropout)

        self.q_experts = nn.Parameter(torch.randn(k, hdim, hdim))
        self.k_experts = nn.Parameter(torch.randn(k, hdim, hdim))

        # gating projection (from phrase embedding)
        self.dyn_proj = nn.Linear(hdim, k)  # phrase -> expert weight

        # v는 고정 projection
        self.v_proj = nn.Linear(hdim, hdim)

        self.norm = nn.LayerNorm(hdim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        """
        x: [B*T, N, C] (video-phrase context)
        y: [B, N, C]   (phrase)
        mask: [B*T, N]
        """
        BT, N, C = x.shape
        B = y.shape[0]
        T = BT // B

        # 1. Gating: [B, N, k]
        dyn = self.dyn_proj(y)        # [B, N, k]
        dyn = dyn.softmax(dim=-1)     # [B, N, k]

        # 2. Weighted sum of expert weights to get [B, N, C, C]
        q_dyn = torch.einsum('bnk,kcd->bncd', dyn, self.q_experts)  # [B, N, C, C]
        k_dyn = torch.einsum('bnk,kcd->bncd', dyn, self.k_experts)  # [B, N, C, C]

        # 3. Apply projection to input
        x_reshape = x.view(B, T, N, C)  # [B, T, N, C]
        q_input = torch.einsum('btnd,bncd->btnd', x_reshape, q_dyn)
        k_input = torch.einsum('btnd,bncd->btnd', x_reshape, k_dyn)
        v_input = self.v_proj(x_reshape)  # [B, T, N, C]

        # 4. Flatten for MultiheadAttention
        q = q_input.view(BT, N, C)
        k = k_input.view(BT, N, C)
        v = v_input.view(BT, N, C)

        attn_out, attn = self.att(q, k, v, key_padding_mask=~mask.bool() if mask is not None else None)
        out = self.norm(x + self.dropout(attn_out))
        return out, attn

class CrossAttention(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.hdim = hdim
        self.nheads = nheads
        self.att = nn.MultiheadAttention(hdim, nheads, batch_first=True, dropout=dropout)
        self.q_proj = nn.Linear(hdim, hdim)
        self.kv_proj = nn.Linear(hdim, 2 * hdim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hdim)
        
        self.linear = nn.Linear(hdim, hdim)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hdim)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: [B, L, C]
            mask: [B, L]
        """
        B, L, C = x.shape
        q = self.q_proj(x)
        k,v = torch.split(self.kv_proj(y), self.hdim, dim=-1)

        if mask is None:
            q_att, attn = self.att(q, k, v)
        else:
            q_att, attn = self.att(q, k, v, key_padding_mask=~(mask.bool()))
        q_att = self.dropout(q_att)
        x = self.norm(x + q_att)

        # linear block with residual connection
        update = self.dropout1(self.act(self.linear(x)))
        x = self.norm1(x + update)
        return x, attn

    
class LowRankDynamicProjector(nn.Module):
    def __init__(self, hdim, r):
        super().__init__()
        self.r = r
        self.proj_phrase = nn.Linear(hdim, hdim * r)   # Phrase -> Low-rank
        self.shared_param = nn.Parameter(torch.randn(r, hdim))  # Learnable [r, C]
        self.bias = nn.Parameter(torch.zeros(hdim))  # Bias for the output
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hdim)
        self.act = nn.ReLU()

    def forward(self, phrase_emb, context_emb):
        """
        phrase_emb: [B, N, C]
        context_emb: [B, T, N*C]
        """
        B, N, C = phrase_emb.shape
        context_emb = rearrange(context_emb, 'b t n c -> b t (n c)')  # [B, T, N*C]
        # 1. Generate Low-Rank Kernel
        dyn_kernel = self.proj_phrase(phrase_emb).view(B, N, C, self.r)   # [B, N, C, r]
        dyn_kernel = torch.matmul(dyn_kernel, self.shared_param)          # [B, N, C, C]
        dyn_kernel = rearrange(dyn_kernel, 'b n c1 c2 -> b (n c1) c2')    # [B, N*C, C]

        # 2. Apply to Context Embedding
        # context_emb: already [B, T, N*C]
        projected = torch.bmm(context_emb, dyn_kernel)  # [B, T, C]
        projected = projected + self.bias

        return self.norm(self.dropout(self.act(projected)))  # [B, T, C]
    
class T_SA_layer(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1):
        super(T_SA_layer, self).__init__()
        self.t_att = SelfAttention(hdim, nheads, dropout=dropout)
        self.linear = nn.Linear(hdim, hdim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hdim)
        self.norm1 = nn.LayerNorm(hdim)
    def forward(self, src_emb, mask=None):
        """
        src_emb: [B, T, C]
        """
        # T-axis self-attention
        src_emb, _ = self.t_att(src_emb, mask) # [B*N, T, C]
        update = self.dropout(self.act(self.linear(src_emb)))
        src_emb = self.norm(src_emb + update)

        return src_emb

class T_SA(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1, num_layers=2):
        super(T_SA, self).__init__()
        self.hdim = hdim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([T_SA_layer(hdim, nheads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_emb, mask=None):
        """
        Args:
            src_emb: [B, T, C]
        Returns:
            updated_phrase: [B, T, C]
        """
        for layer in self.layers:
            src_emb = layer(src_emb, mask)
        return src_emb
    
class EntropyGating(nn.Module):
    def __init__(self, tau=0.1, scale=5.0, shift=2.5):
        """
        tau: softmax temperature
        scale, shift: entropy scaling
        """
        super().__init__()
        self.tau = tau
        self.scale = scale
        self.shift = shift

    def forward(self, score, mask):
        """
        Args:
            score: [B, T]   # t2v attention values (0~1)
            mask:  [B, T]   # valid=1, pad=0
        Returns:
            gate: [B, 1, 1]
        """
        # 1. Masked Softmax
        masked_score = score.masked_fill(mask == 0, float('-inf'))
        prob = F.softmax(masked_score / self.tau, dim=1)   # [B, T]
        entropy = -torch.sum(prob * torch.log(prob + 1e-6) * mask, dim=1)   # [B]
        valid_lengths = mask.sum(dim=1)
        entropy = entropy / torch.log(valid_lengths)
        entropy = torch.clamp(entropy, min=0.0, max=1.0)

        return entropy.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
    
class Saliency_proj(nn.Module):
    def __init__(self, hdim):
        super(Saliency_proj, self).__init__()
        self.proj1 = nn.Linear(hdim, hdim)
        self.proj2 = nn.Linear(hdim, hdim)
        self.hdim = hdim
    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        """
        B, T, C = x.shape
        x1 = self.proj1(x)
        x_global = x.mean(1)
        x2 = self.proj2(x_global).unsqueeze(1)
        intermediate_result = x1 * x2
        saliency_scores = torch.sum(intermediate_result, dim=-1) / self.hdim ** 0.5

        return saliency_scores