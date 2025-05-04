import sys
sys.path.append("/SSD1/minseok/MR_HD/FlashVTG")
from FlashVTG_ms.config import BaseOptions
from FlashVTG_ms.start_end_dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from FlashVTG_ms.inference import eval_epoch, start_inference, setup_model
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
from vis_utils import visualize_phrase_clusters
from FlashVTG_ms.model import build_model1

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
        return text, token_lst

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
    "enc_layers": 3,
    "t2v_layers": 6,
    "dummy_layers": 2,
    "num_dummies": 3,
    "kernel_size": 5,
    "num_conv_layers": 1,
    "num_mlp_layers": 5,
    "num_phrase=":3,
    "context_layers": 2,
    "phrase_layers": 2,
    "rank": 64,
    "t_sa" : 1,
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
model, criterion = build_model1(opt)
model = model.to(device)
model.eval()

model_path = "results/hl-video_tef-exp-2025-04-29-21-21-54/model_best.ckpt"
state_dict = torch.load(model_path, map_location="cpu")["model"]
for name, param in model.named_parameters():
    if name in state_dict:
        print(f"Matching parameter: {name}")
    else:
        print(f"Missing parameter: {name}")
model.load_state_dict(state_dict, strict=False)
tokenizer = Tokenizer()
print("Tokenizer loaded from", tokenizer.tokenizer.name_or_path)
tokenizer = tokenizer.to(device)
frame_dir = "/SSD1/minseok/MR_HD/DB/frames/QVHighlights/"
save_path = "/SSD1/minseok/MR_HD/FlashVTG/visualize/phrase/train"

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
    # 모델 출력에서 attention score 가져오기 (Tensor → numpy)
    sqan_attn_np = outputs["sqan_att"][0].detach().cpu().numpy()  # [N, L]
    slot_attn_np = outputs["slot_att"][0].detach().cpu().numpy()  # [N, L]
    # attention 시각화
    visualize_phrase_clusters(
        query=query,
        tokens=token[4:-1],
        sqan_attn=sqan_attn_np,
        slot_attn=slot_attn_np,
        save_path=os.path.join(save_path, f"{vid}_phrase.png")
    )

    
    frame_np = np.load(os.path.join(frame_dir, vid + ".npy"))
    video_length = frame_np.shape[0]