import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from FlashVTG_ms.config import BaseOptions
from FlashVTG_ms.start_end_dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from FlashVTG_ms.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown

import nncore
from datetime import datetime
import logging

def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()

    # init meters
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)

    # iteration loop
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(
        enumerate(train_loader), desc="Training Iteration", total=num_training_examples
    ):


        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        targets["label"] = batch[0]
        targets["fps"] = torch.full((256,), 1/opt.clip_length).to(opt.device) # if datasets is qv, fps is 0.5
        outputs = model(**model_inputs, targets=targets)

        loss_dict = criterion(batch, epoch_i, outputs, targets)
        #loss_dict = {k: v for k, v in outputs.items() if 'loss' in k}

        

        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )


        if torch.isnan(losses).any():
            print("Loss contains NaN values")

        optimizer.zero_grad()
        losses.backward()

        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), opt.grad_clip, error_if_nonfinite=False
            )
        optimizer.step()

        loss_dict["weighted_loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v)
            )

        # Output and log loss info every iteration
        current_loss = {k: v.avg for k, v in loss_meters.items()}
        for k, v in current_loss.items():
            tb_writer.add_scalar(f"Train/{k}", v, epoch_i * num_training_examples + batch_idx)

        tb_writer.add_scalar(
            "Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i * num_training_examples + batch_idx
        )

    # Write epoch-level logs to file
    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i + 1,
        loss_str=" ".join(
            ["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]
        ),
    )
    logger.info(to_write)
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)
    
    return losses, epoch_i * num_training_examples + batch_idx

def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, test_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    N_phrase = opt.num_phrase
    N_layer = opt.phrase_layers
    C_layer = opt.context_layers
    rank = opt.rank

    contribution = "T_kernel_channels"

    run_name = f"{contribution}_Nphrase{N_phrase}_Nlayer{N_layer}_Clayer{C_layer}_rank{rank}"

    wandb.init(project="FlashVTG", name = run_name, entity="msperceive", sync_tensorboard=True)
    tb_writer = SummaryWriter(log_dir=wandb.run.dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.0
    es_cnt = 0  # early stop counter
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            losses, iteration = train_epoch(
                model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer
            )
            lr_scheduler.step(losses)
        eval_epoch_interval = opt.eval_epoch

        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            # Validation
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = (
                    eval_epoch(
                        model,
                        val_dataset,
                        opt,
                        f"val_latest_{opt.dset_name}_preds.jsonl",
                        epoch_i,
                        criterion,
                        tb_writer,
                    )
                )

            # log validation results
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(
                    ["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]
                ),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info(
                "Validation metrics_no_nms {}".format(
                    pprint.pformat(metrics_no_nms["brief"], indent=4)
                )
            )
            if metrics_nms is not None:
                logger.info(
                    "Validation metrics_nms {}".format(
                        pprint.pformat(metrics_nms["brief"], indent=4)
                    )
                )
                with open(opt.eval_log_filepath, "a") as f:
                    f.write("Validation metrics_nms {}\n".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Val/{k}", float(v), epoch_i + 1)

            # Save best model
            if opt.dset_name in ["hl"]:
                stop_score = metrics["brief"]["MR-full-mAP"]
            elif opt.dset_name in ["tacos"]:
                stop_score = metrics["brief"]["MR-full-R1@0.3"]
            elif opt.dset_name in ["tvsum", "youtube_uni"]:
                stop_score = metrics["brief"]["mAP"]
            else:
                stop_score = (
                    metrics["brief"]["MR-full-R1@0.7"]
                    + metrics["brief"]["MR-full-R1@0.5"]
                ) / 2

            if stop_score > prev_best_score:
                prev_best_score = stop_score
                es_cnt = 0
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                logger.info("Saved best model checkpoint to [{}]".format(opt.ckpt_filepath.replace(".ckpt", "_best.ckpt")))
            else:
                es_cnt += 1
                if es_cnt > opt.max_es_cnt:
                    logger.info("Early stop at epoch {}".format(epoch_i))
                    break

        # Save latest model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_i,
            "opt": opt,
        }
        torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if opt.debug:
            break

    tb_writer.close()

    # Final evaluation on validation and test sets
    logger.info("\n\nEvaluating best model...")
    checkpoint = torch.load(opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
    model.load_state_dict(checkpoint["model"])
    
    # Validation set evaluation
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, _ = eval_epoch(
            model, val_dataset, opt, f"val_best_{opt.dset_name}_preds.jsonl", epoch_i, criterion, tb_writer
        )
    logger.info("Best model validation metrics_no_nms: {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("Best model validation metrics_nms: {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))
    
    # Test set evaluation
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, _ = eval_epoch(
            model, test_dataset, opt, f"test_best_{opt.dset_name}_preds.jsonl", epoch_i, criterion, tb_writer
        )
    logger.info("Best model test metrics_no_nms: {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("Best model test metrics_nms: {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

    logger.info("\n\nEvaluating latest model...")
    checkpoint = torch.load(opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))
    model.load_state_dict(checkpoint["model"])
    
    # Validation set evaluation
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, _ = eval_epoch(
            model, val_dataset, opt, f"val_latest_{opt.dset_name}_preds.jsonl", epoch_i, criterion, tb_writer
        )
    logger.info("Latest model validation metrics_no_nms: {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("Latest model validation metrics_nms: {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))
    
    # Test set evaluation
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, _ = eval_epoch(
            model, test_dataset, opt, f"test_latest_{opt.dset_name}_preds.jsonl", epoch_i, criterion, tb_writer
        )
    logger.info("Latest model test metrics_no_nms: {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("Latest model test metrics_nms: {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

def train_hl(
    model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, test_dataset, opt
):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    N_phrase = opt.num_phrase
    N_layer = opt.phrase_layers
    C_layer = opt.context_layers
    rank = opt.rank

    contribution = "T_kernel_channels"

    run_name = f"{contribution}_Nphrase{N_phrase}_Nlayer{N_layer}_Clayer{C_layer}_rank{rank}"

    wandb.init(project="FlashVTG", name = run_name, entity="msperceive", sync_tensorboard=True)
    tb_writer = SummaryWriter(log_dir=wandb.run.dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.0
    es_cnt = 0  # early stop counter
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            losses, iteration = train_epoch(
                model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer
            )
            lr_scheduler.step(losses)
        eval_epoch_interval = opt.eval_epoch

        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            # Validation
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = (
                    eval_epoch(
                        model,
                        val_dataset,
                        opt,
                        f"val_latest_{opt.dset_name}_preds.jsonl",
                        epoch_i,
                        criterion,
                        tb_writer,
                    )
                )

            # log validation results
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(
                    ["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]
                ),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info(
                "Validation metrics_no_nms {}".format(
                    pprint.pformat(metrics_no_nms["brief"], indent=4)
                )
            )
            if metrics_nms is not None:
                logger.info(
                    "Validation metrics_nms {}".format(
                        pprint.pformat(metrics_nms["brief"], indent=4)
                    )
                )
                with open(opt.eval_log_filepath, "a") as f:
                    f.write("Validation metrics_nms {}\n".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Val/{k}", float(v), epoch_i + 1)

            # Save best model
            if opt.dset_name in ["hl"]:
                stop_score = metrics["brief"]["MR-full-mAP"]
            elif opt.dset_name in ["tacos"]:
                stop_score = metrics["brief"]["MR-full-R1@0.3"]
            elif opt.dset_name in ["tvsum", "youtube_uni"]:
                stop_score = metrics["brief"]["mAP"]
            else:
                stop_score = (
                    metrics["brief"]["MR-full-R1@0.7"]
                    + metrics["brief"]["MR-full-R1@0.5"]
                ) / 2

            if stop_score > prev_best_score:
                prev_best_score = stop_score
                es_cnt = 0
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt,
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
                logger.info("Saved best model checkpoint to [{}]".format(opt.ckpt_filepath.replace(".ckpt", "_best.ckpt")))
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:
                    logger.info(
                        "Early stop, val score = {:.4f}".format(prev_best_score)
                    )
                    break

        # Test - only save predictions without evaluation
        if opt.test_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                _, _, _, latest_file_paths = (
                    eval_epoch(
                        model,
                        test_dataset,
                        opt,
                        f"test_latest_{opt.dset_name}_preds.jsonl",
                        epoch_i,
                        criterion,
                        tb_writer,
                        eval_mode=False  # Don't compute metrics
                    )
                )
            logger.info(f"Saved test predictions to {latest_file_paths[0]}")

        # Save latest model
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_i,
            "opt": opt,
        }
        torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if opt.debug:
            break

    tb_writer.close()

    # Final evaluation on validation set only
    logger.info("\n\nEvaluating best model...")
    checkpoint = torch.load(opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))
    model.load_state_dict(checkpoint["model"])
    
    # Validation set evaluation
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, _ = eval_epoch(
            model, val_dataset, opt, f"val_best_{opt.dset_name}_preds.jsonl", epoch_i, criterion, tb_writer
        )
    logger.info("Best model validation metrics_no_nms: {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("Best model validation metrics_nms: {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))
    
    # Test set - only save predictions
    if opt.test_path is not None:
        with torch.no_grad():
            _, _, _, _ = eval_epoch(
                model, test_dataset, opt, f"test_best_{opt.dset_name}_preds.jsonl", epoch_i, criterion, tb_writer, eval_mode=False
            )
        logger.info(f"Saved final test predictions to test_best_{opt.dset_name}_preds.jsonl")

def start_training():
    logger.info("Setup data and model...")

    # setup data
    train_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        dset_domain=opt.dset_domain,
    )

    if opt.eval_path is not None:
        val_dataset = StartEndDataset(
            dset_name=opt.dset_name,
            data_path=opt.eval_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_type=opt.q_feat_type,
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            load_labels=True,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=0,
            dset_domain=opt.dset_domain,
        )
    else:
        val_dataset = None

    if opt.test_path is not None:
        test_dataset = StartEndDataset(
            dset_name=opt.dset_name,
            data_path=opt.test_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_type=opt.q_feat_type,
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            load_labels=True,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=0,
            dset_domain=opt.dset_domain,
        )
    else:
        test_dataset = None

    # setup model
    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    params = []
    logger.info("Learnable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # logger.info(f"{name} - {param.shape}")
            params.append(param)

    train_params = sum(p.numel() for p in params)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = round(train_params / total_params * 100, 3)
    param = round(train_params / 1024 / 1024, 3)
    logger.info(f"Learnable Parameters: {param}M ({ratio}%)")

    logger.info("Start Training...")

    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum', 'youtube_uni']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, test_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, test_dataset, opt)
    return (
        opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"),
        opt.eval_split_name,
        opt.eval_path,
        opt.debug,
        opt,
    )


if __name__ == "__main__":
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.device.type == "cuda":
        torch.cuda.set_device(opt.device)
        cudnn.benchmark = True

    opt.cfg = nncore.Config.from_file(opt.config)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_directory = os.path.join(opt.results_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log')
    file_handler = logging.FileHandler(log_directory)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()

    if not debug:
        input_args = [
            opt.config,
            "--resume",
            best_ckpt_path,
            "--eval_split_name",
            eval_split_name,
            "--eval_path",
            eval_path,
        ]

        import sys

        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        
        start_inference(opt)
