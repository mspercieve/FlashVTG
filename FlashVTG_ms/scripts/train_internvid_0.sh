dset_name=hl
ctx_mode=video_tef
v_feat_types=internvideo2
t_feat_type=internvideo2
results_root=results
exp_id=exp
device_id=0
######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root='/SSD/minseok/DB/'

# video features
v_feat_dim=0
v_feat_dirs=()

if [[ ${v_feat_types} == *"internvideo2"* ]]; then
  v_feat_dirs+=(${feat_root}/features/internvid_features/qvhighlights/stage2_video/fps2/l40_vid_pool/)
  (( v_feat_dim += 768))
fi

# text features
if [[ ${t_feat_type} == "internvideo2" ]]; then
  t_feat_dir=${feat_root}/features/internvid_features/qvhighlights/text/
  t_feat_dim=4096
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi
#### training
bsz=64
max_v_l=75
max_q_l=40
eval_epoch=3
weight_decay=0.0001
eval_bsz=1

enc_layers=3
t2v_layers=6
dummy_layers=2
num_dummies=3
kernel_size=5
num_conv_layers=1
num_mlp_layers=5

lw_reg=1
lw_cls=5
lw_sal=0.1
lw_saliency=0.8
label_loss_coef=4

PYTHONPATH=$PYTHONPATH:. python FlashVTG_ms/train.py \
data/MR.py \
--use_dfl \
--num_bins 16 \
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
--n_epoch 150 \
--lr_drop 400 \
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
--device ${device_id} \
--use_SRM \
${@:1}