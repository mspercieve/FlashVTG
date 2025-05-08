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
from vis_utils import visualize_phrase_and_context
from FlashVTG_ms.model import build_model1
# results/hl-video_tef-exp-2025-05-07-11-40-38/model_latest.ckpt
def parse_args():
    parser = argparse.ArgumentParser(description='Visualize phrase attention and context')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                      help='Dataset split to visualize (train or val)')
    parser.add_argument('--model_path', type=str, 
                      default='results/hl-video_tef-exp-2025-05-08-13-30-42/model_latest.ckpt',
                      help='Path to the model checkpoint')
    parser.add_argument('--save_dir', type=str, 
                      default='/SSD1/minseok/MR_HD/FlashVTG/visualize/phrase',
                      help='Directory to save visualizations')
    return parser.parse_args()

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
    tokenizer = Tokenizer()
    print("Tokenizer loaded from", tokenizer.tokenizer.name_or_path)
    tokenizer = tokenizer.to(device)
    
    frame_dir = "/SSD1/minseok/MR_HD/DB/frames/QVHighlights/"
    save_path = os.path.join(args.save_dir, args.split)
    os.makedirs(save_path, exist_ok=True)

    for i, batch in enumerate(data_loader):
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
        context_emb_np = outputs["context_emb_out"][0].detach().cpu().numpy()  # [N, T, C]
        context_refine_np = outputs["context_refine_out"][0].detach().cpu().numpy()  # [N, T, C]
        context_agg_np = outputs["context_agg"][0].detach().cpu().numpy()  # [T, C]
        vid_emb_np = outputs["vid_emb"][0].detach().cpu().numpy()  # [T, C]
        
        # phrase 수를 context_emb의 shape에 맞춰서 설정
        num_phrases = context_emb_np.shape[0]
        
        # shape 확인을 위한 디버깅 출력
        print("Shapes:")
        print(f"sqan_attn_np: {sqan_attn_np.shape}")
        print(f"context_emb_np: {context_emb_np.shape}")
        print(f"context_refine_np: {context_refine_np.shape}")
        print(f"context_agg_np: {context_agg_np.shape}")
        print(f"vid_emb_np: {vid_emb_np.shape}")
        print(f"Number of phrases: {num_phrases}")
        
        # 값 확인을 위한 디버깅 출력
        print("\nValues:")
        print("context_agg_np mean:", np.mean(context_agg_np, axis=-1))
        print("vid_emb_np mean:", np.mean(vid_emb_np, axis=-1))
        
        # context_agg가 1차원이면 2차원으로 변환 (T, C)
        if len(context_agg_np.shape) == 1:
            context_agg_np = context_agg_np.reshape(-1, context_emb_np.shape[-1])  # [T, C]로 변환
        
        # 모델 예측 구간 가져오기
        pred_boundary = None
        if "_out" in outputs and "boundary" in outputs["_out"]:
            pred_boundary = outputs["_out"]["boundary"].detach().cpu().numpy()  # [N, 3] (start, end, score)
        
        # phrase attention과 context activation 시각화
        slot_att = outputs.get('slot_att', None)  # [N, L]
        if slot_att is not None:
            slot_att = slot_att[0].detach().cpu().numpy()
        visualize_phrase_and_context(
            query=query,
            tokens=token[4:-1],
            sqan_attn=sqan_attn_np,
            context_emb=context_emb_np,
            context_refine=context_refine_np,
            context_agg=context_agg_np,
            vid_emb=vid_emb_np,
            moment_gt=moment_gt_divided,
            pred_boundary=pred_boundary,
            save_path=os.path.join(save_path, f"{vid}_visualization.png"),
            slot_att=slot_att
        )
        
        frame_np = np.load(os.path.join(frame_dir, vid + ".npy"))
        video_length = frame_np.shape[0]

if __name__ == "__main__":
    main()