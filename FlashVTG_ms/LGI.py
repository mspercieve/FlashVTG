import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('utils/')
import net_utils
import math
from position_encoding import PositionEmbeddingSine
from einops import rearrange, repeat

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
        self.att_fn = Attention(qdim, qdim, qdim, 0.1)

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
            # perform Eq. (5), (6), (7)
            att_f, att_w = self.att_fn(g_n, w_feats, w_mask)

            prev_se = att_f
            se_feats.append(att_f)
            se_attw.append(att_w)

        return torch.stack(se_feats, dim=1), torch.stack(se_attw, dim=1)
    

class Phrase_Generate(nn.Module):
    def __init__(self, num_phrase, hdim, num_heads, drop_p, num_layers):
        super(Phrase_Generate, self).__init__()
        self.num_layers = num_layers
        self.phrase_att = nn.ModuleList([SlotAttention(num_phrase, hdim, num_heads, drop_p) for _ in range(num_layers)])
        self.sqan = SequentialQueryAttention(num_phrase, hdim)
        self.pos = PositionEmbeddingSine(hdim)
    def forward(self, txt_emb, txt_mask):
        """
        Args:
            video_emb: [B, T, C] video-level features
            video_mask: [B, T] boolean mask for valid frames (True for valid)
            phrase_slot: [B, N, C] initial phrase slot embedding (should expanded along T)
        Returns:
            updated_phrase: [B, T, N, C]
        """

        B, L, C = txt_emb.shape
        stc_emb, word_emb = torch.split(txt_emb, [1, L-1], dim=1)
        word_mask = txt_mask[:, 1:] # [B, L-1]
        phrase_slot, phrase_attn = self.sqan(stc_emb, word_emb, word_mask) # [B, N, C]
        word_pos = self.pos(word_emb, word_mask)
        word_pe = word_emb + word_pos
        for i in range(self.num_layers):
            phrase_slot = self.phrase_att[i](word_pe, word_mask, phrase_slot)
            
        return phrase_slot, phrase_attn

class SlotAttention(nn.Module):
    def __init__(self, num_phrase, hdim, num_heads, drop_p=0.1):
        """
        Args:
            num_phrase: number of phrase slots (N)
            hdim: hidden dnsion (C)
            drop_p: dropout probability
        """
        super(SlotAttention, self).__init__()
        self.nh = num_heads
        self.N = num_phrase
        self.C = hdim
        self.dropout = nn.Dropout(drop_p)
        self.dropout_s = nn.Dropout(drop_p)
        self.norm_s = nn.LayerNorm(hdim)
        self.norm = nn.LayerNorm(hdim)

        self.q_proj = nn.Linear(hdim, hdim)
        self.k_proj = nn.Linear(hdim, hdim)
        self.v_proj = nn.Linear(hdim, hdim)
        self.slot_att = nn.MultiheadAttention(hdim, self.nh, batch_first=True, dropout=drop_p)
        self.self_att = nn.MultiheadAttention(hdim, self.nh, batch_first=True, dropout=drop_p)
        
        #linear after cross attention fusion
        self.linear1 = nn.Linear(hdim, hdim)
        self.norm1 = nn.LayerNorm(hdim)
        self.dropout1 = nn.Dropout(drop_p)
        self.act = nn.ReLU()

    def forward(self, txt_feat, txt_mask, phrase_slot):
        """
        Args:
            txt_feat: [B, L, C] phrase-level features
            txt_mask: [B, L] boolean mask for valid phrases (True for valid)
            phrase_slot: [B, N, C] phrase slots
        """
        # check inputs
        B, L, C = txt_feat.shape

        Q = self.q_proj(phrase_slot) # [B, N, C]
        K = self.k_proj(txt_feat) # [B, L, C]
        V = self.v_proj(txt_feat) # [B, L, C]

        slot_update = self.slot_att(Q, K, V, key_padding_mask=~(txt_mask.bool()))[0]
        slot_update = self.dropout(slot_update)
        phrase_slot = self.norm(slot_update + phrase_slot)

        # self attention
        self_update = self.self_att(phrase_slot, phrase_slot, phrase_slot)[0]
        self_update = self.dropout_s(self_update)
        phrase_slot = self.norm_s(self_update + phrase_slot)
        # linear
        phrase_slot = phrase_slot + self.dropout1(self.act(self.linear1(phrase_slot)))
        phrase_slot = self.norm1(phrase_slot)
        return phrase_slot

    
class PhraseWeight_vid(nn.Module):
    def __init__(self, temperature=1.0):
        super(PhraseWeight_vid, self).__init__()
        self.tau = temperature

    def forward(self, phrase_slot, vid_feat, vid_mask):
    # Compute attention weights
        dot_product = torch.einsum('btc,bnc->btn', vid_feat, phrase_slot) # [B, T, N]
        empty_vid_mask = ~(vid_mask.bool()).unsqueeze(-1)
        masked_dot_product = torch.where(empty_vid_mask, 
                                        torch.tensor(float('-inf')).to(dot_product.device), 
                                        dot_product)
        masked_dot_product = masked_dot_product.permute(0, 2, 1) # [B, N, T]
        # Apply temperature-scaled softmax
        softmax_output = F.softmax(masked_dot_product / self.tau, dim=2)
        keyphrase_weight = torch.max(softmax_output, dim=2)[0]

        
        return keyphrase_weight
    

class PhraseWeight_eos(nn.Module):
    def __init__(self, hdim):
        super(PhraseWeight_eos, self).__init__()
        self.q = nn.Linear(hdim, hdim)
        self.k = nn.Linear(hdim, hdim)
    def forward(self, phrase_slot, eos_emb):
        # Compute attention weights
        c = phrase_slot.shape[2]
        eos_q = self.q(eos_emb) # [B, 1, C]
        phrase_k = self.k(phrase_slot) # [B, N, C]
        dot_product = torch.einsum('blc,bnc->bln', eos_q, phrase_k) # [B, 1 N]
        # Apply temperature-scaled softmax
        softmax_output = F.softmax(dot_product / math.sqrt(c), dim=2)
        return softmax_output.squeeze(1) # [B, N]
    
class PhraseContextLayer(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1):
        super(PhraseContextLayer, self).__init__()
        self.n_att = SelfAttention(hdim, nheads, dropout=dropout)
        self.t_att = SelfAttention(hdim, nheads, dropout=dropout)

        self.fc_t = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc_n = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm_t = nn.LayerNorm(hdim)
        self.norm_n = nn.LayerNorm(hdim)

    def forward(self, context_emb, vid_mask, shape):
        """
        context_emb: [B*N, T, C]
        """
        B, N, T, C = shape
        # T-axis self-attention
        context_emb, _ = self.t_att(context_emb, vid_mask) # [B*N, T, C]
        t_update = self.fc_t(context_emb) # [B*N, T, C]
        context_emb = self.norm_t(context_emb + t_update) # [B*N, T, C]
        context_emb = rearrange(context_emb, '(b n) t c -> (b t) n c', b=B, n=N) # [B*T, N, C]
        # N-axis self-attention
        context_att, _ = self.n_att(context_emb, None) # [B*T,N,C]
        context_emb = rearrange(context_att, '(b t) n c -> (b n) t c', b=B, n=N) # [B*N, T, C]
        n_update = self.fc_n(context_emb)
        context_emb = self.norm_n(context_emb + n_update)

        return context_emb


class Phrase_Context(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1, num_layers=2, product_idim_1=None, product_idim_2=None):
        super(Phrase_Context, self).__init__()
        self.hdim = hdim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([PhraseContextLayer(hdim, nheads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        if product_idim_1 is None:
            product_idim_1 = hdim
        if product_idim_2 is None:
            product_idim_2 = hdim

        self.product = HadamardProduct(product_idim_1, product_idim_2, hdim)
        self.pos = PositionEmbeddingSine(hdim)

    def forward(self, phrase_slot, vid_feat, vid_mask):
        """
        Args:
            phrase_slot: [B, N, C] phrase slots
            vid_feat: [B, T, C] video-level features
        Returns:
            updated_phrase: [B, T, N, C]
        """
        B, T, C = vid_feat.shape
        N = phrase_slot.shape[1]
        
        context_emb = self.product([phrase_slot, vid_feat]) # [ B, N, T, C]
        context_emb = rearrange(context_emb, 'b n t c -> (b n) t c') # [B*N, T, C]
        
        vid_mask = repeat(vid_mask, 'b t -> (b n) t', n=N) # [B*N, T]
        pos = self.pos(context_emb, vid_mask) # [B*N, T, C]
        context_emb = context_emb + pos

        for layer in self.layers:
            context_emb = layer(context_emb, vid_mask, (B, N, T, C))
        updated_phrase = rearrange(context_emb, '(b n) t c -> b n t c', b=B, n=N) # [B, N, T, C]
        return updated_phrase


class PhraseContextLayerv2(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1, type='t'):
        super(PhraseContextLayerv2, self).__init__()
        self.att = SelfAttention(hdim, nheads, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hdim)
        self.norm_n = nn.LayerNorm(hdim)

    def forward(self, context_emb, vid_mask=None):
        """
        context_emb: [B*N, T, C]
        """
        context_emb, _ = self.att(context_emb, vid_mask) # [B*N, T, C]
        update = self.fc(context_emb)
        context_emb = self.norm(context_emb + update)
        return context_emb


class Phrase_Contextv2(nn.Module):
    def __init__(self, hdim, nheads, dropout=0.1, num_layers=2, product_idim_1=None, product_idim_2=None):
        super(Phrase_Contextv2, self).__init__()
        self.hdim = hdim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.t_layers = nn.ModuleList([PhraseContextLayerv2(hdim, nheads, dropout, type='t') for _ in range(num_layers)])
        self.n_layers = nn.ModuleList([PhraseContextLayerv2(hdim, nheads, dropout, type='n') for _ in range(num_layers)])
        if product_idim_1 is None:
            product_idim_1 = hdim
        if product_idim_2 is None:
            product_idim_2 = hdim

        self.product = HadamardProduct(product_idim_1, product_idim_2, hdim)
        self.pos = PositionEmbeddingSine(hdim)
        
    def forward(self, phrase_slot, vid_feat, vid_mask):
        """
        Args:
            phrase_slot: [B, N, C] phrase slots
            vid_feat: [B, T, C] video-level features
        Returns:
            updated_phrase: [B, T, N, C]
        """
        B, T, C = vid_feat.shape
        N = phrase_slot.shape[1]
        
        context_emb = self.product([phrase_slot, vid_feat]) # [ B, N, T, C]
        context_emb = rearrange(context_emb, 'b n t c -> (b n) t c') # [B*N, T, C]
        
        vid_mask = repeat(vid_mask, 'b t -> (b n) t', n=N) # [B*N, T]
        pos = self.pos(context_emb, vid_mask) # [B*N, T, C]
        context_emb = context_emb + pos

        # t first, then n
        for t_layer in self.t_layers:
            context_emb = t_layer(context_emb, vid_mask)
        context_emb = rearrange(context_emb, '(b n) t c -> (b t) n c', b=B, n=N) # [B, T, N, C]')
        for n_layer in self.n_layers:
            context_emb = n_layer(context_emb, None)
        updated_phrase = rearrange(context_emb, '(b t) n c -> b n t c', b=B, t=T) # [B, N, T, C]
        return updated_phrase

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
        context_emb = rearrange(context_emb, 'b n t c -> b t (n c)')  # [B, T, N*C]
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
        print('entropy', entropy[:3])

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