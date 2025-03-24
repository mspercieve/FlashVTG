dset_name=charadesSTA
ctx_mode=video_tef
v_feat_types=internvideo2
t_feat_type=llama
results_root=results
exp_id=internvideo2

######## data paths
train_path=data/charades_sta/charades_sta_train_tvr_format.jsonl
eval_path=data/charades_sta/charades_sta_test_tvr_format.jsonl
eval_split_name=val

######## setup video+text features
# video features
v_feat_dim=768
v_feat_dirs=/home/userid/data_ssd/charades-sta/charades_internvideo2/charade_sta_6b
# text features
t_feat_dir=/home/userid/data_ssd/charades-sta/charades_internvideo2/charade_sta_llama_text_feature
t_feat_dim=4096

#### training
bsz=32
max_v_l=-1
max_q_l=23
eval_epoch=1
weight_decay=0.0001
eval_bsz=1

enc_layers=3
t2v_layers=6
dummy_layers=2
num_dummies=40
kernel_size=7
num_conv_layers=2
num_mlp_layers=3

lw_reg=1
lw_cls=5
lw_sal=0.01
lw_saliency=0.8
label_loss_coef=0.1
nms_type=normal

PYTHONPATH=$PYTHONPATH:. python FlashVTG/train.py \
data/MR.py \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--enc_layers ${enc_layers} \
--results_root ${results_root} \
--bsz ${bsz} \
--exp_id ${exp_id} \
--t2v_layers ${t2v_layers} \
--dummy_layers ${dummy_layers} \
--max_v_l ${max_v_l} \
--max_q_l ${max_q_l} \
--n_epoch 50 \
--lr_drop 50 \
--eval_epoch ${eval_epoch} \
--wd ${weight_decay} \
--eval_bsz ${eval_bsz} \
--lw_reg ${lw_reg} \
--lw_cls ${lw_cls} \
--lw_sal ${lw_sal} \
--lw_saliency ${lw_saliency} \
--nms_thd 0.7 \
--use_neg \
--num_dummies ${num_dummies} \
--kernel_size ${kernel_size} \
--num_conv_layers ${num_conv_layers} \
--num_mlp_layers ${num_mlp_layers} \
--label_loss_coef ${label_loss_coef} \
--nms_type ${nms_type} \
--clip_length 1 \
--lr 1.5e-4 \
${@:1}

