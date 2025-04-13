import sys
sys.path.append("/SSD1/minseok/MR_HD/FlashVTG")
sys.path.append("/SSD1/minseok/MR_HD/QD-DETR")
from FlashVTG_ms.config import BaseOptions
import qd_detr.config as qdconfig

from FlashVTG_ms.start_end_dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from FlashVTG.inference import eval_epoch, start_inference, setup_model
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import argparse
from easydict import EasyDict as edict
from transformers import LlamaTokenizer, PreTrainedTokenizerFast, AutoTokenizer
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import glob
from vis_utils import plot_chunk_figure, get_chunk_ranges, visualize_similarity_matrix
# tokenizer
class Tokenizer(nn.Module):
    def __init__(self, tokenizer_path="/SSD1/minseok/MR_HD/InternVideo/InternVideo2/multi_modality/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b/"):
        super(Tokenizer, self).__init__()
        self.tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_path, 
            local_files_only=True,
            legacy=False
        )
        self.tokenizer.pad_token = " "  # allow padding
        self.tokenizer.add_eos_token = True

    def forward(self, text):
        text = ["summarize: " + text]
        encoding = self.tokenizer(text, return_tensors="pt", max_length=80, truncation=True, padding="max_length")
        text = encoding.input_ids
        token_ids = text.tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids[0])
        token_lst = [token for token in tokens if token != "<unk>"]
        token_lst = token_lst[:-1]
        return text, token_lst


def cosine_similarity_matrix(features):
    """
    features: torch.Tensor, shape (L, 4096)
    반환: torch.Tensor, shape (L, L) - 각 단어 임베딩 간 cosine similarity
    """
    norms = torch.norm(features, dim=1, keepdim=True) + 1e-8
    normalized_features = features / norms
    sim_matrix = torch.mm(normalized_features, normalized_features.t())
    return sim_matrix

###################################FLASHVTG###################################
# Your configuration dictionary
config = {
    "dset_name": "hl",
    "ctx_mode": "video_tef",
    "v_feat_types": "internvideo2",
    "t_feat_type": "internvideo2",
    "results_root": "vis",
    "exp_id": "vis",  # use a dummy exp_id for visualization
    "device": 0,
    "train_path": "data/highlight_train_release.jsonl",
    "eval_path": "data/highlight_val_release.jsonl",
    "eval_split_name": "val",
    "feat_root": "/SSD1/minseok/MR_HD/DB",
    "v_feat_dim": 0,
    "v_feat_dirs": [],
    "t_feat_dir": "",
    "t_feat_dim": 0,
    "bsz": 1,
    "max_v_l": 75,
    "max_q_l": 40,
    "eval_epoch": 3,
    "weight_decay": 0.0001,
    "eval_bsz": 1,
    "enc_layers": 3,
    "t2v_layers": 6,
    "dummy_layers": 2,
    "num_dummies": 10,
    "kernel_size": 5,
    "num_conv_layers": 1,
    "num_mlp_layers": 5,
    "lw_reg": 1,
    "lw_cls": 5,
    "lw_sal": 0.1,
    "lw_saliency": 0.8,
    "label_loss_coef": 4,
    "num_workers": 4,
    "pin_memory": True,
    "cfg": {
        "_base_": ['blocks'],
        "model": {
            "strides": (1, 2, 4, 8),
            "buffer_size": 1024,
            "max_num_moment": 50,
            "pyramid_cfg": {"type": "ConvPyramid"},
            "pooling_cfg": {"type": "AdaPooling"},
            "class_head_cfg": {"type": "ConvHead", "kernal_size": 3},
            "coord_head_cfg": {"type": "ConvHead", "kernal_size": 3},
            "loss_cfg": {
                "type": 'BundleLoss',
                "sample_radius": 1.5,
                "loss_cls": {"type": 'FocalLoss'},
                "loss_reg": {"type": 'L1Loss'},
                "loss_sal": {"type": 'SampledNCELoss'},
            },
        },
    },
}

if "internvideo2" in config["v_feat_types"]:
    config["v_feat_dirs"].append(f"{config['feat_root']}/features/internvid_features/qvhighlights/stage2_video/fps2/l40_vid_pool/")
    config["v_feat_dim"] += 768

if config["t_feat_type"] == "internvideo2":
    config["t_feat_dir"] = f"{config['feat_root']}/features/internvid_features/qvhighlights/text/"
    config["t_feat_dim"] = 4096
else:
    raise ValueError("Wrong arg for t_feat_type.")

# Setup BaseOptions using your config as defaults
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0], "dummy_config"]
base_options = BaseOptions()
base_options.initialize()
base_options.parser.set_defaults(**config)

# Monkey-patch display_save to disable folder creation and code saving.
base_options.display_save = lambda opt: None

opt = base_options.parse()
opt.cfg = edict(opt.cfg)  # convert cfg to EasyDict if needed

dataset_config = dict(
    dset_name=opt.dset_name,
    data_path=opt.train_path,
    v_feat_dirs=opt.v_feat_dirs,
    q_feat_dir=opt.t_feat_dir,
    q_feat_type="last_hidden_state",
    max_q_l=opt.max_q_l,
    max_v_l=opt.max_v_l,
    ctx_mode=opt.ctx_mode,
    data_ratio=1.0,
    normalize_v=not getattr(opt, "no_norm_vfeat", False),
    normalize_t=not getattr(opt, "no_norm_tfeat", False),
    clip_len=0,
    max_windows=0,
    span_loss_type="ce",
    txt_drop_ratio=0.0,
    dset_domain="",
)
dataset_config["data_path"] = opt.train_path

train_dataset = StartEndDataset(**dataset_config)
train_loader = DataLoader(
    train_dataset,
    collate_fn=start_end_collate,
    batch_size=opt.bsz,
    num_workers=opt.num_workers,
    shuffle=True,
    pin_memory=opt.pin_memory,
)

device = opt.device
from FlashVTG_ms.model import build_model1
model, criterion = build_model1(opt)
model = model.to(device)
model.eval()

# load pretrained model
model_pth = "results/hl-video_tef-exp-2025-04-01-17-09-36/model_best.ckpt"
state_dict = torch.load(model_pth, map_location="cpu")["model"]
model.load_state_dict(state_dict)
print("Model loaded from", model_pth)
###############################################################################

# load pretrained tokenizer
tokenizer = Tokenizer()
print("Tokenizer loaded from", tokenizer.tokenizer.name_or_path)
tokenizer = tokenizer.to(device)

# frame directory
frame_dir = "/SSD1/minseok/MR_HD/DB/frames/QVHighlights/"
save_path = "/SSD1/minseok/MR_HD/FlashVTG/visualize/train_noEOS"

for i, batch in enumerate(train_loader):
    model_inputs, targets = prepare_batch_inputs(batch[1], device, non_blocking=opt.pin_memory)
    targets["label"] = batch[0]
    text_query = targets["label"][0]["query"]
    text, token = tokenizer(text_query)
    outputs = model(**model_inputs, targets=targets)
    
    

    print(targets["label"][0])
    vid = targets["label"][0]["vid"]
    query = targets["label"][0]["query"]
    moment_gt = targets["label"][0]["relevant_windows"]
    moment_gt_divided = [[int(x / 2) for x in sublist] for sublist in moment_gt]
    attn = outputs["t2vattnvalues"][0]
    
    frame_np = np.load(os.path.join(frame_dir, vid + ".npy"))
    video_length = frame_np.shape[0]

    t_feat = model_inputs["src_txt"][0]
    t_sim = cosine_similarity_matrix(t_feat)

    t_proj = outputs["src_txt_out"][0]
    t_proj_sim = cosine_similarity_matrix(t_proj)
    qd_t_proj_sim = torch.zeros_like(t_proj_sim)
    # visualize
    vis_path = os.path.join(save_path, vid)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    chunk_ranges = get_chunk_ranges(video_length, num_chunks=3)
    
    visualize_similarity_matrix(token, t_sim, t_proj_sim, qd_t_proj_sim, query_text=query, save_path=vis_path)

    #Flash-VTG
    for idx, crange in enumerate(chunk_ranges):
        print('token:', token)
        print('attn:', attn.shape)
        vtg_vis_path = os.path.join(vis_path, "vtg")
        plot_chunk_figure(query, moment_gt_divided, frame_np, token, attn, crange, idx, vtg_vis_path, vid)
        

    

