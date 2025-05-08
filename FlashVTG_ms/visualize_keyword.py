import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import cv2
from einops import rearrange
import sys
import os
import argparse
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
import torch.nn as nn
from FlashVTG_ms.config import BaseOptions
from FlashVTG_ms.start_end_dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from FlashVTG_ms.model import build_model1
from easydict import EasyDict as edict

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize keyword attention and context')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                      help='Dataset split to visualize (train or val)')
    parser.add_argument('--model_path', type=str, 
                      default='results/hl-video_tef-exp-2025-05-07-18-24-05/model_best.ckpt',
                      help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, 
                      default='visualize/keyword',
                      help='Directory to save visualizations')
    parser.add_argument('--tokenizer_path', type=str,
                      default="/SSD1/minseok/MR_HD/InternVideo/InternVideo2/multi_modality/InternVL/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b/",
                      help='Path to the tokenizer')
    return parser.parse_args()

class Tokenizer(nn.Module):
    def __init__(self, tokenizer_path):
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

def visualize_word_video_attention(word_video_attn, selected_indices, word_tokens, entropy=None, save_path=None):
    """
    Word-Video attention 시각화
    Args:
        word_video_attn: [B, L, T] attention scores
        selected_indices: [B, N] selected word indices
        word_tokens: [B, L] word tokens
        entropy: [B, L] word entropy values
        save_path: 저장할 경로
    """
    B, L, T = word_video_attn.shape
    
    # 첫 번째 배치만 시각화
    attn = word_video_attn[0].detach().cpu().numpy()  # [L, T]
    indices = selected_indices[0].detach().cpu().numpy()  # [N]
    tokens = word_tokens[0]  # [L]
    if entropy is not None:
        entropy = entropy[0].detach().cpu().numpy()  # [L]
    
    plt.figure(figsize=(15, 8))
    
    # attention map 시각화
    plt.subplot(1, 1, 1)
    sns.heatmap(attn, cmap='Greys', xticklabels=5, yticklabels=tokens)
    plt.title('Word-Video Attention Map')
    plt.xlabel('Video Frames')
    plt.ylabel('Words')
    
    # 선택된 단어 표시
    for idx in indices:
        plt.axhline(y=idx, color='red', linestyle='-', alpha=0.3, linewidth=2)
    
    # entropy 값 표시
    if entropy is not None:
        for i, (token, ent) in enumerate(zip(tokens, entropy)):
            plt.text(T + 1, i, f'{ent:.3f}', va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_slot_attention(slot_att, word_tokens, selected_indices, save_path=None):
    """
    Slot-Word attention 시각화
    Args:
        slot_att: [B, N, L] attention scores
        word_tokens: [B, L] word tokens
        selected_indices: [B, N] selected word indices
        save_path: 저장할 경로
    """
    B, N, L = slot_att.shape
    
    # 첫 번째 배치만 시각화
    attn = slot_att[0].detach().cpu().numpy()  # [N, L]
    tokens = word_tokens[0]  # [L]
    indices = selected_indices[0].detach().cpu().numpy()  # [N]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(attn, cmap='Greys', xticklabels=tokens, yticklabels=[f'Phrase {i+1}' for i in range(N)])
    plt.title('Phrase-Word Attention Map')
    plt.xlabel('Words')
    plt.ylabel('Phrases')
    
    # 선택된 단어 표시
    for i, idx in enumerate(indices):
        plt.axvline(x=idx, color='red', linestyle='-', alpha=0.3, linewidth=2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_activation_maps(context_agg, vid_emb, save_path=None):
    """
    Context aggregation과 video embedding의 activation map 시각화
    Args:
        context_agg: [B, T, C] context aggregated features
        vid_emb: [B, T, C] video features
        save_path: 저장할 경로
    """
    B, T, C = context_agg.shape
    
    # 첫 번째 배치만 시각화
    context = context_agg[0].detach().cpu().numpy()  # [T, C]
    video = vid_emb[0].detach().cpu().numpy()  # [T, C]
    
    # L2 norm 계산
    context_norm = np.linalg.norm(context, axis=1)  # [T]
    video_norm = np.linalg.norm(video, axis=1)  # [T]
    
    plt.figure(figsize=(15, 5))
    
    # 1. Context aggregation activation
    plt.subplot(1, 2, 1)
    plt.plot(context_norm, color='red', alpha=0.7)
    plt.title('Context Aggregation Activation')
    plt.xlabel('Time Steps')
    plt.ylabel('L2 Norm')
    
    # 2. Video embedding activation
    plt.subplot(1, 2, 2)
    plt.plot(video_norm, color='red', alpha=0.7)
    plt.title('Video Embedding Activation')
    plt.xlabel('Time Steps')
    plt.ylabel('L2 Norm')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def visualize_all(model_output, word_tokens, save_dir):
    """
    모든 시각화를 수행
    Args:
        model_output: model의 output dictionary
        word_tokens: [B, L] word tokens
        save_dir: 저장할 디렉토리
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Word-Video attention 시각화
    visualize_word_video_attention(
        model_output['word_video_attn'],
        model_output['selected_indices'],
        word_tokens,
        entropy=model_output.get('entropy', None),
        save_path=os.path.join(save_dir, 'word_video_attention.png')
    )
    
    # Slot attention 시각화
    visualize_slot_attention(
        model_output['slot_att'],
        word_tokens,
        model_output['selected_indices'],
        save_path=os.path.join(save_dir, 'slot_attention.png')
    )
    
    # Activation maps 시각화
    visualize_activation_maps(
        model_output['context_agg'],
        model_output['vid_emb'],
        save_path=os.path.join(save_dir, 'activation_maps.png')
    )

def main():
    args = parse_args()
    
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
        "context_layers": 2,
        "phrase_layers": 2,
        "num_phrase": 4,
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
        data_path=opt.eval_path if args.split == 'val' else opt.train_path,
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

    dataset = StartEndDataset(**dataset_config)
    data_loader = DataLoader(
        dataset,
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

    state_dict = torch.load(args.model_path, map_location="cpu")["model"]
    for name, param in model.named_parameters():
        if name in state_dict:
            print(f"Matching parameter: {name}")
        else:
            print(f"Missing parameter: {name}")
    model.load_state_dict(state_dict, strict=False)
    
    tokenizer = Tokenizer(args.tokenizer_path)
    print("Tokenizer loaded from", tokenizer.tokenizer.name_or_path)
    tokenizer = tokenizer.to(device)
    
    save_dir = os.path.join(args.save_dir, args.split)
    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(data_loader):
        model_inputs, targets = prepare_batch_inputs(batch[1], device, non_blocking=opt.pin_memory)
        targets["label"] = batch[0]
        text_query = targets["label"][0]["query"]
        text, tokens = tokenizer(text_query)
        outputs = model(**model_inputs, targets=targets)
        
        print(f"\nProcessing video {i+1}:")
        print(f"Query: {text_query}")
        print(f"Tokens: {tokens[4:-1]}")  # Remove special tokens
        
        # 시각화 수행
        vid = targets["label"][0]["vid"]
        current_save_dir = os.path.join(save_dir, vid)
        visualize_all(outputs, tokens[4:-1], current_save_dir)
        
        if i >= 4:  # 처음 5개 비디오만 시각화
            break

if __name__ == "__main__":
    main() 