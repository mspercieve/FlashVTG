import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('/SSD1/minseok/MR_HD/FlashVTG/utils/')
import net_utils
import math
from position_encoding import PositionEmbeddingSine

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
        stc_emb, phrase_emb = torch.split(txt_emb, [1, L-1], dim=1)
        phrase_mask = txt_mask[:, 1:] # [B, L-1]
        phrase_slot, phrase_attn = self.sqan(stc_emb, phrase_emb, phrase_mask) # [B, N, C]
        
        for i in range(self.num_layers):
            phrase_slot = self.phrase_att[i](txt_emb, txt_mask, phrase_slot)
            
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

    '''
class PhraseWeight(nn.Module):
    def __init__(self, temperature=1.0):
        super(PhraseWeight, self).__init__()
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
    '''

class PhraseWeight(nn.Module):
    def __init__(self, hdim):
        super(PhraseWeight, self).__init__()
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
        self.linear = nn.Linear(hdim, hdim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hdim)
        self.norm1 = nn.LayerNorm(hdim)
    def forward(self, context_emb):
        # N-axis self-attention
        context_att, _ = self.n_att(context_emb, None) # [B*T,N,C]
        context_emb = self.norm1(context_emb + context_att)
        # feedforward with residual connection
        update = self.dropout(self.act(self.linear(context_emb)))
        context_emb = self.norm(context_emb + update)

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
        context_emb = context_emb.transpose(1, 2).contiguous().view(B*T, N, C) # [B*T, N, C]

        for layer in self.layers:
            context_emb = layer(context_emb)
        updated_phrase = context_emb.view(B, T, N, C).transpose(1, 2).contiguous() # [B, N, T, C]
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
        self.proj = nn.Linear(hdim, 3 * hdim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hdim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, C]
            mask: [B, L]
        """
        B, L, C = x.shape
        x = self.proj(x)
        q, k, v = torch.split(x, C, dim=-1) # [B, L, C]

        if mask is None:
            x, attn = self.att(q, k, v)
        else:
            x, attn = self.att(q, k, v, key_padding_mask=~(mask.bool()))
        x = self.dropout(x)
        x = self.norm(x + v)
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
    
class AttentivePooling(nn.Module):
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim (int): embedding dimension of the phrases
        """
        super(AttentivePooling, self).__init__()
        self.attn_weight = nn.Parameter(torch.randn(embed_dim, 1))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, T, C]
        Returns:
            out: Tensor of shape [B, T, C]
            weighted sum of phrases
        """
        B, N, T, C = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, N, C]
        x_reshaped = x.view(B * T, N, C)  # [B*T, N, C]

        scores = torch.matmul(x_reshaped, self.attn_weight) # [B*T, N, 1]
        scores = scores.squeeze(-1)  # [B*T, N]

        attn_scores = F.softmax(scores, dim=-1)  # [B*T, N]
        attn_scores = attn_scores.unsqueeze(-1)   # [B*T, N, 1]
        pooled = torch.sum(x_reshaped * attn_scores, dim=1)  # [B*T, C]

        out = pooled.view(B, T, C)
        return out
    
