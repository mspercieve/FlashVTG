import torch
from torch.utils.data import Dataset
import numpy as np
import random
import logging
from os.path import join
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from FlashVTG.span_utils import span_xx_to_cxw
from torchtext import vocab
import torch.nn as nn

logger = logging.getLogger(__name__)

TVSUM_SPLITS = {
    'BK': {
        'train': ['WxtbjNsCQ8A', 'EE-bNr36nyA', 'oDXZc0tZe04', 'uGu_10sucQo'],
        'val': ['Se3oxnaPsz0']
    },
    'BT': {
        'train': ['eQu1rNs0an0', 'qqR6AEXwxoQ', 'EYqVtI9YWJA', 'iVt07TCkFM0'],
        'val': ['JgHubY5Vw3Y']
    },
    'DS': {
        'train': ['kLxoNp-UchI', 'NyBmCxDoHJU', 'jcoYJXDG9sw', '-esJrBWj2d8'],
        'val': ['E11zDS9XGzg']
    },
    'FM': {
        'train': ['_xMr-HKMfVA', 'byxOvuiIJV0', 'VuWGsYPqAX8', 'xmEERLqJ2kU'],
        'val': ['JKpqYvAdIsw']
    },
    'GA': {
        'train': ['xxdtq8mxegs', 'i3wAGJaaktw', '0tmA_C6XwfM', '3eYKfiOEJNs'],
        'val': ['Bhxk-O1Y7Ho']
    },
    'MS': {
        'train': ['Hl-__g2gn_A', 'WG0MBPpPC6I', 'LRw_obCPUt0', '37rzWOQsNIw'],
        'val': ['Yi4Ij2NM7U4']
    },
    'PK': {
        'train': ['GsAD1KT1xo8', 'XkqCExn6_Us', 'b626MiF1ew4', 'PJrm840pAUI'],
        'val': ['cjibtmSLxQ4']
    },
    'PR': {
        'train': ['RBCABdttQmI', 'z_6gVvQb2d0', '4wU_LUjG5Ic', '91IHQYk1IQM'],
        'val': ['fWutDQy1nnY']
    },
    'VT': {
        'train': ['gzDbaEs1Rlg', 'XzYM3PfTM4w', '98MoyGZKHXc', 'AwmHb44_ouw'],
        'val': ['J0nA4VgnoCo']
    },
    'VU': {
        'train': ['akI8YFjEmUw', 'HT5vyqe0Xaw', 'vdmoEJ5YbrQ', 'xwqBXPGE9pQ'],
        'val': ['sTEELN-vY30']
    }
}
class StartEndDataset(Dataset):
    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state", "features"]
    """One line in data loaded from data_path."
    {
      "qid": 7803,
      "query": "Man in gray top walks from outside to inside.",
      "duration": 150,
      "vid": "RoripwjYFp8_360.0_510.0",
      "relevant_clip_ids": [13, 14, 15, 16, 17],
      "relevant_windows": [[26, 36]]
    }
    """

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs \
            if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        if max_v_l == -1:
            max_v_l = 100000000
        if max_q_l == -1:
            max_q_l = 100
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows  # maximum number of windows to use as labels
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()
        
        # load specific domain data for tvsum dataset
        if self.dset_name in ['tvsum', 'tvsum_sfc']:
            target_domain = dset_domain
            assert target_domain in ["BK", "BT", "DS", "FM", "GA", "MS", "PK", "PR", "VT", "VU"]

            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data
            
        # load specific domain data for youtube-hl dataset
        if self.dset_name == 'youtube_uni':
            target_domain = dset_domain
            assert target_domain in ["dog", "gymnastics", "parkour", "skating", "skiing", "surfing"]
            
            new_data = []
            for d in self.data:
                if target_domain == d['domain']:
                    new_data.append(d)
            self.data = new_data    
        
        self.use_glove = False
        self.use_glove = 'vgg' in self.v_feat_dirs[0]

        if self.dset_name == 'charadesSTA' and self.use_glove:
            self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
            self.vocab.itos.extend(['<unk>'])
            self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
            self.vocab.vectors = torch.cat(
                (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
            self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)
        
        # Load all data into memory
        self._preload_data()

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def _preload_data(self):
        self.preloaded_data = []
        for index in range(len(self.data)):
            meta = self.data[index]
            model_inputs = self._load_model_inputs(meta)
            self.preloaded_data.append((meta, model_inputs))

    def _load_model_inputs(self, meta):
        model_inputs = dict()

        if self.use_glove:
            model_inputs["query_feat"] = self.get_query(meta["query"])
        else:
            model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)
            
        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.dset_name in ['tvsum']:
            model_inputs["span_labels"] = torch.tensor([[0., 0.]])
            meta_label = meta['label']
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all_tvsum(meta_label, ctx_l)
            if len(model_inputs["saliency_all_labels"]) != len(model_inputs["video_feat"]):
                model_inputs["video_feat"] = model_inputs["video_feat"][:len(model_inputs["saliency_all_labels"])]

        elif self.dset_name == 'youtube_uni':
            model_inputs["span_labels"] = torch.tensor([[0., 0.]])
            meta_label = meta['label']
            model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all_youtube(meta_label, ctx_l)
        else:
            if "relevant_windows" in meta: ## For Qvhighlights test set
                model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
                if self.dset_name in ['charadesSTA', 'tacos', 'activitynet']: ## charades, tacos, nlq
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l)  # only one gt
                elif self.dset_name in ['nlq']:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l, 2)  # only one gt
                elif "subs_train" not in self.data_path:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                        self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
                else:
                    model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs[
                        "saliency_all_labels"] = \
                        self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l)  # only one gt

        if 'qvhighlight' in self.data_path:
            model_inputs["relevant_clip_ids"] = meta["relevant_clip_ids"]
        model_inputs["vid"] = meta["vid"]
        model_inputs["qid"] = meta["qid"]
        return model_inputs

    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, index):
        return self.preloaded_data[index]

    def get_query(self, query):
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def get_saliency_labels_sub_as_query(self, gt_window, duration, ctx_l, max_n=2):
        clip_len = duration / ctx_l
        gt_st = int(gt_window[0] / clip_len)
        gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
        if gt_st > gt_ed:
            gt_st = gt_ed

        if gt_st != gt_ed:
            pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)
        else:
            if self.dset_name == 'nlq':
                pos_clip_indices = [gt_st] * 2
            else:
                pos_clip_indices = [gt_st, gt_st]

        neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l))
        try:
            neg_clip_indices = random.sample(neg_pool, k=max_n)
        except:
            neg_clip_indices = pos_clip_indices

        # For charades_sta
        score_array = np.zeros(ctx_l)
        score_array[gt_st:gt_ed + 1] = 1

        return pos_clip_indices, neg_clip_indices, score_array
        

    def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_tvsum(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        
        agg_scores = np.sum(labels - np.ones_like(labels), axis=-1)[:ctx_l] # start from 1, so minus 1
        score_array = agg_scores / 80 * 12
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array

    def get_saliency_labels_all_youtube(self, labels, ctx_l, max_n=1, add_easy_negative=False):
        
        # Youtube-hl only have binary score
        agg_scores = np.array(labels)[:, 0] # (L, 1) --> (L, )
        score_array = agg_scores * 1
        
        sort_indices = np.argsort(agg_scores)  # increasing

        hard_pos_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(idx, ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices

        return pos_clip_indices, neg_clip_indices, score_array
    
    
    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        if self.dset_name == 'tvsum':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid))) # 'token', 'text'
            return torch.from_numpy(q_feat['last_hidden_state'])
            # return torch.from_numpy(q_feat['token'])
        # youtube-hl
        elif self.dset_name == 'youtube_uni':
            q_feat = np.load(join(self.q_feat_dir, "{}.npz".format(qid)))
            return torch.from_numpy(q_feat['last_hidden_state'])
        
        elif self.dset_name in ['tacos', 'nlq']:
            q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
            q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
        else:
            try:
                # QVhighlight dataset
                q_feat_path = join(self.q_feat_dir, f"qid{qid}.npz")
                q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
                if self.q_feat_type == "last_hidden_state":
                    q_feat = q_feat[:self.max_q_l]
                if self.normalize_t:
                    q_feat = l2_normalize_np_array(q_feat)
                if self.txt_drop_ratio > 0:
                    q_feat = self.random_drop_rows(q_feat)
            except:
                q_feat_path = join(self.q_feat_dir, f"{qid}.npy")
                q_feat = np.load(q_feat_path).astype(np.float32)
                q_feat = np.concatenate([q_feat[-1:], q_feat[4:-1]], axis=0) # Ensure the first token is the [EOS] token
                if self.q_feat_type == "last_hidden_state":
                    q_feat = q_feat[:self.max_q_l]
                if self.normalize_t:
                    q_feat = l2_normalize_np_array(q_feat)
                if self.txt_drop_ratio > 0:
                    q_feat = self.random_drop_rows(q_feat)
        return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

    def random_drop_rows(self, embeddings):
        """randomly mask num_drop rows in embeddings to be zero.
        Args:
            embeddings: np.ndarray (L, D)
        """
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        if self.dset_name == 'tvsum':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                try:
                    _feat_path = join(_feat_dir, f"{vid}_rgb.npy")
                    _feat_rgb = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                    _feat_path = join(_feat_dir, f"{vid}_opt.npy")
                    _feat_opt = np.load(_feat_path)[:self.max_v_l].astype(np.float32)

                    _feat = np.concatenate([_feat_rgb, _feat_opt], axis=-1)
                except:
                    try:
                        _feat_path = join(_feat_dir, f"{vid}.npy")
                        _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                    except:
                        _feat_path = join(_feat_dir, f"{vid}.npz")
                        _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                
                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)

        elif self.dset_name == 'youtube_uni':
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                # Only single npz files per directory
                try:
                    _feat_path = join(_feat_dir, f"{vid}.npz")
                    _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                except:
                    _feat_path = join(_feat_dir, f"{vid}.npy")
                    _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                
                # _feat = _feat_rgb
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list] # TODO do we need to cut the length over the min_len?
            v_feat = np.concatenate(v_feat_list, axis=1)

        else:
            v_feat_list = []
            for _feat_dir in self.v_feat_dirs:
                try:
                    _feat_path = join(_feat_dir, f"{vid}.npz")
                    _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                except:
                    try:
                        _feat_path = join(_feat_dir, f"{vid}.pt")
                        _feat = torch.load(_feat_path)[:self.max_v_l].float().numpy()
                    except:
                        _feat_path = join(_feat_dir, f"{vid}.npy")
                        _feat = np.load(_feat_path)[:self.max_v_l].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            # some features are slightly longer than the others
            min_len = min([len(e) for e in v_feat_list])
            v_feat_list = [e[:min_len] for e in v_feat_list]
            v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)


def start_end_collate(batch):
    # batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?
    batch_meta = [e[0] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0][1].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e[1]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e[1][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e[1][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue
        if k == 'qid':
            batched_data[k] = [e[1][k] for e in batch]
            continue
        if k == 'vid':
            batched_data[k] = [e[1][k] for e in batch]
            continue
        batched_data[k] = pad_sequences_1d(
            [e[1][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
        vid=batched_model_inputs["vid"],
        qid=batched_model_inputs["qid"],
    )
    targets = {}

    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]

    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
        targets["relevant_clips"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
    
    targets = None if len(targets) == 0 else targets
    return model_inputs, targets