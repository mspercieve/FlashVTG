tfl_config_path=$1
ckpt_path=$2
eval_split_name=$3
# eval_path=data/highlight_${eval_split_name}_release.jsonl
eval_path=data/charades_sta/charades_sta_${eval_split_name}_tvr_format.jsonl
PYTHONPATH=$PYTHONPATH:. python FlashVTG/inference.py \
${tfl_config_path} \
--resume ${ckpt_path} \
--eval_split_name ${eval_split_name} \
--eval_path ${eval_path} \
${@:4}
