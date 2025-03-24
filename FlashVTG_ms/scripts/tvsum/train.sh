dset_name=tvsum
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_tvsum
exp_id=demo


######## data paths
train_path=data/tvsum/tvsum_train.jsonl
eval_path=data/tvsum/tvsum_val.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/home/caozhuo/data_ssd/tvsum

# # video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_slowfast)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_clip)
  (( v_feat_dim += 512 ))
fi

# # text features
t_feat_dir=${feat_root}/txt_clip/ # maybe not used
t_feat_dim=512

#### training
bsz=4
lr=1e-3
enc_layers=3
t2v_layers=2
dummy_layers=2

kernel_size=5
num_conv_layers=2
num_mlp_layers=3

lw_cls=5
lw_sal=0.1
lw_saliency=0.8
label_loss_coef=4

num_dummies=3

######## TVSUM domain name
for dset_domain in BK BT DS FM GA MS PK PR VT VU
do
    for num_dummies in 3
    do
        PYTHONPATH=$PYTHONPATH:. python FlashVTG/train.py \
        data/HD.py \
        --dset_name ${dset_name} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dir ${t_feat_dir} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}/${dset_domain} \
        --exp_id ${exp_id} \
        --max_v_l 1000 \
        --n_epoch 600 \
        --lr_drop 3000 \
        --max_es_cnt -1 \
        --seed 2024 \
        --lr ${lr} \
        --dset_domain ${dset_domain} \
        --enc_layers ${enc_layers} \
        --t2v_layers ${t2v_layers} \
        --dummy_layers ${dummy_layers} \
        --num_dummies ${num_dummies} \
        --lw_cls ${lw_cls} \
        --lw_sal ${lw_sal} \
        --lw_saliency ${lw_saliency} \
        --eval_epoch 1 \
        --dropout 0.1 \
        --wd 0.05 \
        --use_neg \
        ${@:1}
    done
done